"""Test-persona discovery endpoint.

Exposes the canned personas to the frontend so the UI can offer a persona
picker. The persona list is resolved from one of two sources, in order:

1. **Volume override** — ``{DATABRICKS_VOLUME}/config/test_personas.yaml``.
   When this file exists it is used *exclusively*. This lets teams that deploy
   Ontos via DABs from a git repo edit the persona list at runtime (drop/update
   the file on the Unity Catalog Volume) without changing the repo or
   restarting the app.
2. **Repo-bundled fallback** — ``src/backend/src/data/test_personas.yaml``
   (checked into git). Used when no volume override is present, e.g. local dev.

The endpoint is enabled only when ``TEST_USER_TOKEN`` is configured server-side
(otherwise the entire header-override mechanism is dormant and the frontend
should not show any persona UI). When disabled, the endpoint returns 404 so
clients can probe without learning that the feature exists.

Freshness: the resolved list is cached with a short TTL (see
``_CACHE_TTL_SECONDS``) so edits to the volume file surface within seconds and
without a restart, while bounding how often we hit the Volume/Files API.

Security:
- The endpoint does NOT echo back the ``TEST_USER_TOKEN`` itself. The token
  is provisioned to the dev/test client out-of-band (e.g. a frontend
  ``VITE_TEST_USER_TOKEN`` env var or a curl flag). This prevents an
  accidental information leak via a misconfigured prod deployment.
- The persona list itself is harmless metadata, but we still gate it on the
  token being configured to avoid surfacing the feature in production.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.common.authorization import (
    TEST_TOKEN_HEADER,
    TEST_USER_EMAIL_HEADER,
    TEST_USER_GROUPS_HEADER,
    TEST_USER_IP_HEADER,
    TEST_USER_NAME_HEADER,
    TEST_USER_USERNAME_HEADER,
)
from src.common.config import Settings, get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/test", tags=["Test"])

# Repo-bundled fallback list (checked into git).
PERSONAS_YAML_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "test_personas.yaml"
)

# Relative location of the optional volume override, under DATABRICKS_VOLUME.
PERSONAS_VOLUME_SUBPATH = "config/test_personas.yaml"

# How long a resolved persona list is cached before we re-check the source.
# Small enough that volume edits appear "live"; large enough to avoid hitting
# the Files API on every picker load.
_CACHE_TTL_SECONDS = 30.0


class TestPersona(BaseModel):
    id: str
    label: str
    email: str
    # Groups are optional: when omitted, the backend falls back to a SCIM
    # lookup so the persona reflects real workspace state.
    groups: Optional[List[str]] = None
    description: Optional[str] = None


class TestPersonasResponse(BaseModel):
    personas: List[TestPersona]
    # Names of the headers the frontend must send. Surfacing them here keeps
    # the client and server agreed on the wire format.
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "token": TEST_TOKEN_HEADER,
            "email": TEST_USER_EMAIL_HEADER,
            "groups": TEST_USER_GROUPS_HEADER,
            "username": TEST_USER_USERNAME_HEADER,
            "name": TEST_USER_NAME_HEADER,
            "ip": TEST_USER_IP_HEADER,
        }
    )


_cached_personas: Optional[List[TestPersona]] = None
_cache_expires_at: float = 0.0


def _invalidate_persona_cache() -> None:
    """Drop the cached persona list (used by tests and future refresh hooks)."""
    global _cached_personas, _cache_expires_at
    _cached_personas = None
    _cache_expires_at = 0.0


def _read_volume_override(settings: Settings) -> Optional[str]:
    """Return the raw YAML text of the volume override, or None if absent.

    ``DATABRICKS_VOLUME`` is injected as a full ``/Volumes/...`` path in a
    Databricks Apps deployment. ``/Volumes`` is NOT a real filesystem mount in
    that runtime, so reads must go through the SDK Files API — mirroring the
    pattern used for PDF/document downloads elsewhere. For local development
    ``DATABRICKS_VOLUME`` is typically a dotted placeholder (or unset), in which
    case we treat it as a plain filesystem path (and it simply won't exist, so
    we fall back to the bundled list).
    """
    base = (settings.DATABRICKS_VOLUME or "").rstrip("/")
    if not base:
        return None

    if base.startswith("/Volumes/"):
        volume_path = f"{base}/{PERSONAS_VOLUME_SUBPATH}"
        try:
            # Imported lazily so this module has no hard dependency on the SDK
            # client at import time (and so tests can patch it cleanly).
            from src.common.workspace_client import get_workspace_client

            ws = get_workspace_client()
            resp = ws.files.download(file_path=volume_path)
            contents = getattr(resp, "contents", None)
            raw = contents.read() if contents is not None else resp.read()  # type: ignore[union-attr]
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            logger.info("Loaded test personas from volume override: %s", volume_path)
            return text
        except Exception as e:
            # Missing file / permission / SDK error → fall back to the bundled
            # list. Debug-level: absence of an override is the common case.
            logger.debug(
                "No usable test-persona override on volume at %s (%s)",
                volume_path,
                e,
            )
            return None

    # Local/dotted DATABRICKS_VOLUME: treat as a filesystem path (dev only).
    local_path = Path(base) / PERSONAS_VOLUME_SUBPATH
    if local_path.is_file():
        try:
            text = local_path.read_text()
            logger.info("Loaded test personas from local override: %s", local_path)
            return text
        except OSError as e:
            logger.warning("Failed reading persona override %s: %s", local_path, e)
    return None


def _read_bundled() -> Optional[str]:
    """Return the raw YAML text of the repo-bundled persona list, or None."""
    if not PERSONAS_YAML_PATH.is_file():
        logger.warning(
            "test_personas.yaml not found at %s; returning empty persona list",
            PERSONAS_YAML_PATH,
        )
        return None
    try:
        return PERSONAS_YAML_PATH.read_text()
    except OSError as e:
        logger.error("Failed reading bundled test_personas.yaml: %s", e)
        return None


def _parse_personas(raw_text: Optional[str], source: str) -> List[TestPersona]:
    """Parse persona YAML text into validated models, skipping malformed entries."""
    if not raw_text:
        return []
    try:
        raw = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as e:
        logger.error("Failed to parse test personas from %s: %s", source, e)
        return []

    items = raw.get("personas") or []
    personas: List[TestPersona] = []
    for item in items:
        try:
            personas.append(TestPersona(**item))
        except Exception as e:
            logger.warning("Skipping malformed test persona %r: %s", item, e)
    return personas


def _load_personas(settings: Settings) -> List[TestPersona]:
    """Resolve and cache the persona list (volume override, else bundled)."""
    global _cached_personas, _cache_expires_at

    now = time.monotonic()
    if _cached_personas is not None and now < _cache_expires_at:
        return _cached_personas

    raw_text = _read_volume_override(settings)
    source = "volume override"
    if raw_text is None:
        raw_text = _read_bundled()
        source = "repo-bundled file"

    personas = _parse_personas(raw_text, source)
    _cached_personas = personas
    _cache_expires_at = now + _CACHE_TTL_SECONDS
    return personas


@router.get("/personas", response_model=TestPersonasResponse)
def list_test_personas(settings: Settings = Depends(get_settings)) -> TestPersonasResponse:
    """List the canned test personas.

    Returns 404 when ``TEST_USER_TOKEN`` is unset (feature disabled).
    """
    if not settings.TEST_USER_TOKEN:
        raise HTTPException(status_code=404, detail="Test mode not enabled")

    return TestPersonasResponse(personas=_load_personas(settings))


def register_routes(app):
    """Register the test-personas router with the FastAPI app."""
    app.include_router(router)
    logger.info("Test-personas routes registered")

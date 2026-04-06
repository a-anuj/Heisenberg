"""Server module for Clinical Triage Agent."""

try:
    from .app import app
    __all__ = ["app"]
except ImportError:
    pass

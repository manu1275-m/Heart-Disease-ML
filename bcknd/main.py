"""Compatibility shim for older imports.

Keeps `from bcknd.main import app` working by re-exporting from `backend.main`.
"""

from backend.main import *  # noqa: F401,F403

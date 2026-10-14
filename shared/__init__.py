"""
Shared utilities and services for the emb-explorer applications.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Single source of truth: the version declared in pyproject.toml,
    # read from the installed package metadata.
    __version__ = _version("emb-explorer")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

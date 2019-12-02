import pathlib
from pathlib import Path


def checkmate_root_dir() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def checkmate_data_dir() -> Path:
    """Returns project root folder."""
    return checkmate_root_dir() / "data"


def checkmate_cache_dir() -> Path:
    """Returns cache dir"""
    return pathlib.Path('/tmp') / 'remat_cache'
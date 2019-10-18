from pathlib import Path


def remat_root_dir() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def remat_data_dir() -> Path:
    """Returns project root folder."""
    return remat_root_dir() / "data"

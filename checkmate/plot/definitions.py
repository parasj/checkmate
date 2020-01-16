import pathlib
from pathlib import Path

PLATFORM_CHOICES = ["p32xlarge", "p32xlarge_fp16", "p2xlarge", "c524xlarge", "flops"]


def pretty_platform_name(platform: str):
    mapping = {"p32xlarge": "V100", "p32xlarge_fp16": "V100, fp16", "p2xlarge": "K80", "flops": "FLOPs"}
    if platform in mapping:
        return mapping[platform]
    return platform


def platform_memory(platform: str):
    mapping = {
        "p32xlarge": 16 * 1000 * 1000 * 1000,
        "p32xlarge_fp16": 16 * 1000 * 1000 * 1000,
        "p2xlarge": 12 * 1000 * 1000 * 1000,
        "flops": 12 * 1000 * 1000 * 1000,
    }
    if platform in mapping:
        return mapping[platform]
    return platform


def checkmate_root_dir() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def checkmate_data_dir() -> Path:
    """Returns project root folder."""
    return checkmate_root_dir() / "data"


def checkmate_cache_dir() -> Path:
    """Returns cache dir"""
    return pathlib.Path("/tmp") / "remat_cache"

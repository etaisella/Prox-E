"""Prox-E inference package."""

from __future__ import annotations

import os

# Trellis matches paper-style FlashAttention when the flash_attn package is installed (see setup_environment.sh).
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
import tempfile
from pathlib import Path


def _cache_suffix() -> str:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return "user"
    return str(getuid())


def _first_writable_dir(candidates: list[Path]) -> Path:
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    fallback = Path(tempfile.gettempdir()) / f"prox-e-cache-{_cache_suffix()}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def configure_runtime_environment() -> None:
    """
    Set cache defaults before CUDA extensions, numba, or matplotlib initialize.

    A normal user environment has a writable home cache, but cluster jobs,
    containers, and read-only launchers often do not. Keeping these defaults
    under a Prox-E-specific cache also avoids reusing Torch extensions compiled
    for another GPU architecture.
    """
    cache_root = _first_writable_dir(
        [
            Path(os.environ.get("XDG_CACHE_HOME", "")) / "prox-e"
            if os.environ.get("XDG_CACHE_HOME")
            else Path.home() / ".cache" / "prox-e",
            Path(tempfile.gettempdir()) / f"prox-e-cache-{_cache_suffix()}",
        ]
    )

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

    for env_var in ("TORCH_EXTENSIONS_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)

    if "XDG_RUNTIME_DIR" not in os.environ:
        runtime_dir = cache_root / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime_dir.chmod(0o700)
        except OSError:
            pass
        os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)


configure_runtime_environment()

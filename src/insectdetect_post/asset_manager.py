"""Download and verify checksummed release assets defined in JSON registries.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Downloads assets listed in JSON registries and verifies them by SHA-256 on every use,
so that updating a registry entry replaces an outdated local copy.

Functions:
    compute_sha256(): Compute and return the SHA-256 checksum of a file.
    download_file(): Download a file from the given URL to a local destination path.
    list_registered_filenames(): Return the filenames of all assets in a registry.
    ensure_asset(): Resolve a registered asset's local path, downloading and verifying it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Create module-level logger
logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute and return the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = 60,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> None:
    """Download a file from the given URL to a local destination path."""
    request = urllib.request.Request(url, headers={"User-Agent": "insect-detect-post"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            dest_path.open("wb") as f
        ):
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_logged_pct = -1
            while chunk := response.read(1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded / total * 100)
                    if pct >= last_logged_pct + 10:
                        msg = f"{downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct}%)"
                        logger.info("  %s", msg)
                        if progress_callback:
                            progress_callback(pct, 100, f"Downloading {dest_path.name}: {msg}")
                        last_logged_pct = pct
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise


def _load_registry(registry_path: Path) -> list[dict]:
    """Load the list of asset entries from a JSON registry file."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Asset registry not found: '{registry_path}'")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return registry.get("assets", [])


def _resolve_and_verify(
    name: str,
    url: str,
    expected_sha256: str,
    dest_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Path:
    """Ensure dest_dir holds the file registered at url, verified by SHA-256.

    An existing local file is re-verified on every call and replaced if it does not match
    the registered checksum, so updating a registry entry propagates to installs that
    already downloaded the previous version. Downloads are staged to a '.part' file and
    swapped in only after verification, so a failed download leaves any existing copy intact.

    Raises ValueError and deletes the staged file on checksum mismatch. Returns the local path.
    """
    archive_name = url.split("/")[-1]
    archive_path = dest_dir / archive_name

    if archive_path.exists():
        actual = compute_sha256(archive_path)
        if actual == expected_sha256:
            logger.debug("Asset '%s' is already present and matches the registry.", name)
            return archive_path
        logger.warning(
            "Local '%s' does not match the checksum registered for asset '%s' "
            "(registered %s..., found %s...). Replacing it with the registered version.",
            archive_name, name, expected_sha256[:12], actual[:12]
        )

    # Download to a temporary file so a failed download cannot destroy a usable local copy
    staged_path = archive_path.with_name(f"{archive_name}.part")
    logger.info("Downloading '%s'...", name)
    download_file(url, staged_path, progress_callback=progress_callback)

    logger.info("Verifying checksum for '%s'...", name)
    actual = compute_sha256(staged_path)
    if actual != expected_sha256:
        staged_path.unlink()
        raise ValueError(
            f"SHA-256 mismatch for '{archive_name}': expected {expected_sha256}, got {actual}"
        )
    staged_path.replace(archive_path)

    logger.info("Asset '%s' downloaded and verified.", name)
    return archive_path


def list_registered_filenames(registry_path: Path) -> list[str]:
    """Return the filenames of all assets in a registry."""
    return [asset["url"].split("/")[-1] for asset in _load_registry(registry_path)]


def ensure_asset(
    filename: str,
    registry_path: Path,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Path:
    """Resolve a registered asset's local path, downloading and verifying it.

    The local file is verified against the registry on every call,
    so a changed registry entry replaces an outdated local copy.

    Args:
        filename: Filename of the asset (URL basename), as referenced by a registry entry.
        registry_path: Path to the JSON registry. Asset is downloaded into parent directory.
        progress_callback: Optional callback for download progress reporting.

    Raises:
        FileNotFoundError: If registry_path does not exist.
        KeyError: If filename is not a registered asset in registry_path.
        ValueError: If the downloaded file fails checksum verification.

    Returns:
        Path to the verified local asset file.
    """
    for asset in _load_registry(registry_path):
        if asset["url"].split("/")[-1] == filename:
            return _resolve_and_verify(
                asset["name"], asset["url"], asset["sha256"], registry_path.parent, progress_callback
            )
    raise KeyError(f"'{filename}' not found in '{registry_path}'.")

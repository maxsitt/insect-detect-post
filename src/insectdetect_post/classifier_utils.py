"""Shared utilities for the classifier modules.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Utilities for parsing crop filenames, validating metadata,
and merging classification results into the metadata CSV.

Functions:
    parse_crop_name(): Parse crop filename to extract timestamp and track ID.
    format_time(): Format seconds into human-readable time.
    validate_metadata(): Validate that metadata has required columns.
    save_classification_results(): Join classification results with metadata and save to CSV.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import polars as pl

from insectdetect_post.constants import OutputLayout
from insectdetect_post.dataset_harmonizer import parse_crop_stem

# Create module-level logger
logger = logging.getLogger(__name__)

# Regular expressions to extract timestamp components from crop filenames
_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{6})')
_TS_RE_LEGACY = re.compile(r'(\d{4})(\d{2})(\d{2})_(\d{2})-(\d{2})-(\d{2})\.(\d{6})')


def parse_crop_name(name: str) -> tuple[str, int | None]:
    """Parse crop filename to extract timestamp and track ID.

    Args:
        name: Crop filename (current or legacy format).

    Returns:
        Tuple of (timestamp in ISO format, track ID). Falls back to the raw
        filename stem and a None track ID if the name cannot be parsed.
    """
    stem = Path(name).stem
    try:
        parsed = parse_crop_stem(stem)
        if parsed is None:
            return stem, None
        ts_part, track_id = parsed

        m = _TS_RE.search(ts_part)
        if m:
            date, hour, minute, second, micros = m.groups()
            ts_iso = f"{date}T{hour}:{minute}:{second}.{micros}"
            return ts_iso, track_id

        m_legacy = _TS_RE_LEGACY.search(ts_part)
        if m_legacy:
            year, month, day, hour, minute, second, micros = m_legacy.groups()
            ts_iso = f"{year}-{month}-{day}T{hour}:{minute}:{second}.{micros}"
            return ts_iso, track_id

        return ts_part, track_id
    except ValueError:
        logger.warning("Could not parse crop filename %r, using raw stem", name)
        return stem, None


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string (e.g., "2m 30s" or "45s").
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def validate_metadata(
    metadata_path: Path,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> None:
    """Validate that metadata has required columns.

    Args:
        metadata_path: Path to metadata CSV.
        progress_callback: Optional progress callback.

    Raises:
        ValueError: If required columns are missing.
    """
    if progress_callback:
        progress_callback(0, 100, "Validating metadata...")

    df_header = pl.read_csv(metadata_path, n_rows=0)
    required = {"timestamp", "track_id"}
    missing = required - set(df_header.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")


def save_classification_results(
    df_cls: pl.DataFrame,
    metadata_path: Path,
    output_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Path:
    """Join classification results with metadata and save to CSV.

    Args:
        df_cls: DataFrame with classification results.
        metadata_path: Path to original metadata CSV.
        output_dir: Output directory for results.
        progress_callback: Optional progress callback.

    Returns:
        Path to saved metadata CSV with classification results.
    """
    if progress_callback:
        progress_callback(98, 100, "Saving results...")

    df_full = pl.read_csv(metadata_path)

    if len(df_cls) > 0:
        matched = df_full.join(
            df_cls.select(["timestamp", "track_id"]), on=["timestamp", "track_id"], how="semi"
        ).height
        if matched == 0:
            logger.warning(
                "Classification results matched 0 of %d metadata rows (had %d classified crops) - "
                "check crop filename/timestamp parsing", len(df_full), len(df_cls)
            )

    out = df_full.join(df_cls, on=["timestamp", "track_id"], how="left", maintain_order="left")

    if "crop_path" in out.columns:
        out = out.select([c for c in out.columns if c != "crop_path"] + ["crop_path"])

    out_dir = OutputLayout(output_dir).metadata_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metadata_path.stem}_classified.csv"
    out.write_csv(out_path)

    return out_path

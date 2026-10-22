"""Harmonize metadata columns, timestamps and crop filenames across insect-detect versions.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    rename_legacy_columns(): Lowercase and rename legacy metadata columns to their canonical form.
    parse_timestamp_column(): Parse a timestamp column, while handling legacy format and UTC offsets.
    timestamp_offset(): Extract the UTC offset suffix of a timestamp column.
    parse_crop_stem(): Parse a crop filename stem to (timestamp, track_id), while handling legacy format.
"""

from __future__ import annotations

import re

import polars as pl

# Mapping of legacy metadata column names (lowercase) to their canonical equivalents
LEGACY_COLUMNS: dict[str, str] = {
    "cam_id": "device_id",
    "rec_id": "session_id",
}

# ISO timestamp format written to metadata
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%.f"

# Legacy timestamp format written to metadata, identical to the filename stem in that era
LEGACY_TIMESTAMP_FORMAT = "%Y%m%d_%H-%M-%S.%6f"

# Canonical timestamp format for normalizing regardless of source version
CANONICAL_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%.6f"

# Timestamp format embedded in image filenames, polars strftime dialect (%6f = zero-padded)
FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S-%6f"

# Same format as FILENAME_TIMESTAMP_FORMAT, Python strftime dialect (%f, no "6" width modifier)
FILENAME_TIMESTAMP_FORMAT_PY = "%Y-%m-%d_%H-%M-%S-%f"

# Marker preceding the track ID in crop filename stems, and the digits that follow it
_ID_MARKER = "_ID"
_TRACK_ID_RE = re.compile(r"\d+")

# UTC offset suffix of ISO timestamps written from timezone-aware datetimes (e.g. '+00:00')
_UTC_OFFSET_PATTERN = r"(?:[+-]\d{2}:\d{2}|Z)$"


def rename_legacy_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase and rename legacy metadata columns to their canonical form."""
    col_map = {col: LEGACY_COLUMNS.get(col.lower(), col.lower()) for col in df.columns}
    return df.rename(col_map)


def parse_timestamp_column(col: str = "timestamp") -> tuple[pl.Expr, pl.Expr]:
    """Parse a timestamp column, while handling legacy format and UTC offsets.

    A UTC offset is dropped without conversion: the time is kept as written, so it still
    matches the timestamp embedded in the full-frame and crop filenames.
    """
    iso = pl.col(col).str.replace(_UTC_OFFSET_PATTERN, "").str.strptime(pl.Datetime, ISO_TIMESTAMP_FORMAT, strict=False)
    legacy = pl.col(col).str.strptime(pl.Datetime, LEGACY_TIMESTAMP_FORMAT, strict=False)
    parsed = pl.coalesce([iso, legacy])
    is_legacy_format = iso.is_null() & legacy.is_not_null()
    return parsed, is_legacy_format


def timestamp_offset(col: str = "timestamp") -> pl.Expr:
    """Extract the UTC offset suffix of a timestamp column (e.g. '+00:00'), null if there is none."""
    return pl.col(col).str.extract(f"({_UTC_OFFSET_PATTERN})", 1)


def parse_crop_stem(stem: str) -> tuple[str, int] | None:
    """Parse a crop filename stem to (timestamp, track_id), while handling legacy format."""
    id_idx = stem.find(_ID_MARKER)
    if id_idx != -1:
        ts_part = stem[:id_idx]
        m = _TRACK_ID_RE.match(stem, id_idx + len(_ID_MARKER))
        return (ts_part, int(m.group())) if m else None

    # Legacy format: no "_ID" marker - strip suffix, trailing number after underscore is the track_id
    body = None
    for suffix in ("_cropped", "_crop"):
        if stem.endswith(suffix):
            body = stem.removesuffix(suffix)
            break
    if body is None:
        return None

    last_us = body.rfind("_")
    if last_us == -1:
        return None
    ts_part, track_str = body[:last_us], body[last_us + 1:]
    return (ts_part, int(track_str)) if track_str.isdecimal() else None

"""Inspect a dataset directory and cache its metadata and image mappings for pipeline processing.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Harmonizes metadata across insect-detect versions and links detections from metadata
to their corresponding full frame and crop images if available.

Classes:
    CropOnlyEntry: A crop-only detection: an existing crop file with no matching detection frame.
    CameraInfo: Aggregated metadata stats for a single camera/device.
    DatasetContext: Cached dataset information for efficient pipeline processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import polars as pl

from insectdetect_post.constants import SETTING_COLUMN
from insectdetect_post.dataset_harmonizer import (
    CANONICAL_TIMESTAMP_FORMAT,
    FILENAME_TIMESTAMP_FORMAT,
    LEGACY_COLUMNS,
    parse_crop_stem,
    parse_timestamp_column,
    rename_legacy_columns,
    timestamp_offset,
)

# Create module-level logger
logger = logging.getLogger(__name__)

# Columns a metadata file must have, else the file is refused (a missing 'device_id' falls back to the folder name)
_REQUIRED_FILE_COLUMNS = ("timestamp", "track_id", "label")

# Columns a metadata row must have values in, else the row is dropped (for image processing and classification)
_REQUIRED_ROW_COLUMNS = (
    "device_id", "track_id", "label", "x_min", "y_min", "x_max", "y_max", "filename"
)

# Columns identifying a detection (finding files that share detections, sorting the merged metadata)
_DETECTION_KEY_COLUMNS = ("device_id", "session_id", "timestamp", "track_id")

# Unparseable timestamps a metadata file may have, else it is refused (a device crash leaves one incomplete line)
_MAX_INVALID_TIMESTAMPS = 3


def _sanitize_metadata_rows(df: pl.DataFrame, source_name: str) -> pl.DataFrame:
    """Drop rows with an unparseable timestamp or nulls in required columns.

    Guards against partially-written CSV lines from a device crash mid-write.
    """
    if "timestamp" not in df.columns:
        return df

    original_count = len(df)

    if df["timestamp"].dtype == pl.Datetime:
        valid_mask = pl.col("timestamp").is_not_null()
    else:
        parsed_ts, _ = parse_timestamp_column("timestamp")
        valid_mask = parsed_ts.is_not_null()
    for col in _REQUIRED_ROW_COLUMNS:
        if col in df.columns:
            valid_mask = valid_mask & pl.col(col).is_not_null()

    df = df.filter(valid_mask)

    dropped = original_count - len(df)
    if dropped > 0:
        logger.warning(
            "Dropped %d of %d rows with missing/invalid values in %s "
            "(likely a partially-written line from a device crash)",
            dropped, original_count, source_name
        )

    return df


def _screen_metadata_file(meta_path: Path) -> tuple[str | None, pl.DataFrame | None]:
    """Check that a metadata file is unmodified camera output and read its detection keys.

    Refuses a file that cannot be parsed, lacks a required column (e.g. saved with ';' as separator)
    or has more unparseable timestamps than a device crash leaves.

    Returns:
        '(reason, None)' for a refused file, '(None, keys)' with the unique detection keys
        (_DETECTION_KEY_COLUMNS, timestamp canonical, without 'device_id' if the file has none)
        otherwise, '(None, None)' for an empty file.
    """
    try:
        # Check the header first: a ';' file with decimal commas would fail the full read below
        header = pl.read_csv(meta_path, n_rows=0).columns
        col_map = {col: LEGACY_COLUMNS.get(col.lower(), col.lower()) for col in header}
        missing = [col for col in _REQUIRED_FILE_COLUMNS if col not in col_map.values()]
        if missing:
            if len(header) == 1 and ";" in header[0]:
                return "uses ';' as separator, it was probably saved by another program (e.g. Excel)", None
            return f"lacks the required column(s) {', '.join(missing)}", None

        df = pl.read_csv(meta_path, infer_schema=False)
    except pl.exceptions.NoDataError:
        return None, None
    except (pl.exceptions.PolarsError, OSError) as e:
        return f"cannot be read as CSV ({str(e).splitlines()[0]})", None

    df = rename_legacy_columns(df.select(col for col in header if col_map[col] in _DETECTION_KEY_COLUMNS))
    parsed_ts, _ = parse_timestamp_column("timestamp")
    df = df.with_columns(parsed_ts.dt.strftime(CANONICAL_TIMESTAMP_FORMAT).alias("timestamp"))
    n_invalid = df["timestamp"].null_count()
    if n_invalid > _MAX_INVALID_TIMESTAMPS:
        return (f"{n_invalid} of {len(df)} timestamps cannot be read, "
                "the file was probably edited (e.g. in a spreadsheet program)"), None

    if "session_id" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("session_id"))
    keys = df.select(c for c in _DETECTION_KEY_COLUMNS if c in df.columns).drop_nulls(["timestamp", "track_id"])
    return None, keys.unique()


def _find_duplicate_files(keys_by_file: dict[Path, pl.DataFrame], root_path: Path) -> dict[Path, str]:
    """Find metadata files sharing detections with another file.

    Of byte-identical files the first (in path order) is kept and the others are refused as copies.
    A file sharing detections with two files that share none with each other is merged from other
    files and refused alone. Files that still share detections are all refused: which one holds the
    camera's data is unknown.

    Args:
        keys_by_file: Unique detection keys per metadata file, in path order.
        root_path: Dataset root, the refusal reasons name files relative to it.

    Returns:
        Refusal reason per refused file.
    """
    if not keys_by_file:
        return {}
    paths = list(keys_by_file)
    shared = (
        pl.concat(keys.with_columns(pl.lit(i).alias("_file")) for i, keys in enumerate(keys_by_file.values()))
        .group_by(_DETECTION_KEY_COLUMNS)
        .agg(pl.col("_file"))
        .filter(pl.col("_file").list.len() > 1)
    )
    # Number of shared detections per pair of files (both directions)
    involved: dict[Path, dict[Path, int]] = {}
    for files in shared["_file"].to_list():
        for a, b in permutations(files, 2):
            counts = involved.setdefault(paths[a], {})
            counts[paths[b]] = counts.get(paths[b], 0) + 1

    refused: dict[Path, str] = {}

    def name(path: Path) -> str:
        """Return the POSIX-style path relative to the dataset root."""
        return path.relative_to(root_path).as_posix()

    def partners(path: Path) -> dict[Path, int]:
        """Return the files not yet refused that share detections with 'path'."""
        return {other: n for other, n in involved[path].items() if other not in refused}

    first_by_digest: dict[str, Path] = {}
    for path in sorted(involved):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in first_by_digest:
            refused[path] = f"identical copy of {name(first_by_digest[digest])}"
        else:
            first_by_digest[digest] = path

    remaining = [p for p in sorted(involved) if p not in refused]
    merged = {p: len(partners(p)) for p in remaining
              if any(b not in involved[a] for a, b in combinations(partners(p), 2))}
    for path, n_others in merged.items():
        refused[path] = f"combines detections of {n_others} other metadata files, it was probably merged from them"

    overlapping = {p: others for p in remaining if p not in refused and (others := partners(p))}
    for path, others in overlapping.items():
        shared_with = "; ".join(f"{n} with {name(other)}" for other, n in sorted(others.items()))
        refused[path] = (f"shares detections with other metadata files ({shared_with}), "
                         "remove the copy that is not the camera's original file")
    return refused


def _find_session_config_json(meta_path: Path) -> Path | None:
    """Find a device config snapshot JSON next to a session's metadata CSV."""
    matches = sorted(meta_path.parent.glob("*config*.json"))
    return matches[0] if matches else None


def _load_deployment_setting(config_path: Path) -> str | None:
    """Read the deployment.setting field from a device config snapshot JSON."""
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read config snapshot %s", config_path.name)
        return None
    setting = data.get("deployment", {}).get("setting")
    return setting if isinstance(setting, str) and setting.strip() else None


@dataclass(frozen=True)
class CropOnlyEntry:
    """A crop-only detection: an existing crop file with no matching detection frame."""
    path: Path
    label: str


@dataclass
class CameraInfo:
    """Aggregated metadata stats for a single camera/device."""
    detections: int = 0
    metadata_files: set[Path] = field(default_factory=set)
    date_range: tuple[datetime, datetime] | None = None

    @property
    def num_metadata_files(self) -> int:
        return len(self.metadata_files)


@dataclass
class DatasetContext:
    """Cached dataset information for efficient pipeline processing."""
    root_path: Path
    metadata_files: list[Path] = field(default_factory=list)
    refused_metadata_files: dict[Path, str] = field(default_factory=dict)
    detection_frames: list[Path] = field(default_factory=list)
    timelapse_frames: list[Path] = field(default_factory=list)
    crop_frames: list[Path] = field(default_factory=list)
    overlay_frames: list[Path] = field(default_factory=list)
    detection_frame_map: dict[tuple[str, str], Path] = field(default_factory=dict)
    crop_only_map: dict[tuple[str, int], CropOnlyEntry] = field(default_factory=dict)
    _metadata_df: pl.DataFrame | None = field(default=None, init=False, repr=False, compare=False)
    _device_id_fallbacks: dict[Path, str] | None = field(default=None, init=False, repr=False, compare=False)
    camera_info: dict[str, CameraInfo] = field(default_factory=dict)

    total_detections: int = 0
    detection_frames_missing: int = 0
    detections_with_frames: int = 0
    detections_with_crops_only: int = 0
    detections_missing_both: int = 0
    legacy_unlinkable_files: int = 0
    legacy_unlinkable_detections: int = 0

    @property
    def has_metadata(self) -> bool:
        return len(self.metadata_files) > 0

    @property
    def num_metadata_files(self) -> int:
        return len(self.metadata_files)

    @property
    def total_detection_frames(self) -> int:
        return len(self.detection_frames)

    @property
    def total_timelapse_frames(self) -> int:
        return len(self.timelapse_frames)

    @property
    def num_crop_frames(self) -> int:
        return len(self.crop_frames)

    @property
    def date_range(self) -> tuple[datetime, datetime] | None:
        """Calculate overall temporal range from camera info."""
        if not self.has_metadata:
            return None

        all_ranges = [info.date_range for info in self.camera_info.values()
                      if info.date_range]
        if not all_ranges:
            return None

        return (
            min(start for start, _ in all_ranges),
            max(end for _, end in all_ranges)
        )

    @classmethod
    def from_inspection(
        cls,
        root_path: Path,
        progress_callback: Callable[[int, str], None] | None = None
    ) -> DatasetContext:
        """Create DatasetContext from read-only filesystem inspection.

        Args:
            root_path: Root directory to scan.
            progress_callback: Optional callback receiving (percent, message) updates.

        Returns:
            A fully populated DatasetContext.
        """
        logger.debug("Inspecting dataset: %s", root_path)
        ctx = cls(root_path=root_path)
        all_full_frames = []
        file_count = 0

        if progress_callback:
            progress_callback(0, "Scanning files...")

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            file_count += 1
            if progress_callback and file_count % 1000 == 0:
                progress_callback(20, f"Scanned {file_count} files...")

            filename = file_path.name
            if filename.endswith(".jpg"):
                if "crop" in filename:
                    ctx.crop_frames.append(file_path)
                elif "overlay" in filename:
                    ctx.overlay_frames.append(file_path)
                elif "timelapse" in filename:
                    ctx.timelapse_frames.append(file_path)
                else:
                    all_full_frames.append(file_path)
            elif "metadata" in filename and filename.endswith(".csv"):
                ctx.metadata_files.append(file_path)

        ctx.metadata_files.sort()
        ctx.crop_frames.sort()
        ctx._screen_metadata_files()

        # Pre-index full frames by name for fast lookup
        full_frame_index: dict[str, Path] = {f.stem: f for f in all_full_frames}

        logger.info(
            "Found %d metadata files and %d full, %d crop, %d overlay, %d timelapse frames",
            len(ctx.metadata_files),
            len(all_full_frames),
            len(ctx.crop_frames),
            len(ctx.overlay_frames),
            len(ctx.timelapse_frames)
        )

        if progress_callback:
            progress_callback(30, "Analyzing metadata...")

        # Analyze metadata and categorize frames
        ctx._analyze_metadata_and_categorize_frames(all_full_frames, full_frame_index)

        if progress_callback:
            progress_callback(100, "Complete")

        return ctx

    def __repr__(self) -> str:
        return (
            f"DatasetContext({self.root_path}: "
            f"{len(self.detection_frames)} detection frames, "
            f"{len(self.timelapse_frames)} timelapse frames, "
            f"{len(self.crop_frames)} crop frames, "
            f"{self.total_detections} detections)"
        )

    def _screen_metadata_files(self) -> None:
        """Move metadata files that are not unmodified camera output to refused_metadata_files.

        Refused are files that cannot be parsed, lack a required column, have more unparseable
        timestamps than a device crash leaves, or share detections with another file.
        """
        keys_by_file: dict[Path, pl.DataFrame] = {}
        for meta_path in self.metadata_files:
            reason, keys = _screen_metadata_file(meta_path)
            if reason is not None:
                self.refused_metadata_files[meta_path] = reason
            elif keys is not None:
                if "device_id" not in keys.columns:
                    device_id = self._get_device_id_fallbacks()[meta_path]
                    keys = keys.with_columns(pl.lit(device_id).alias("device_id"))
                keys_by_file[meta_path] = keys.select(_DETECTION_KEY_COLUMNS)
        self.refused_metadata_files.update(_find_duplicate_files(keys_by_file, self.root_path))

        for meta_path, reason in sorted(self.refused_metadata_files.items()):
            logger.warning(
                "Refused metadata file %s: %s", meta_path.relative_to(self.root_path).as_posix(), reason
            )
        self.metadata_files = [p for p in self.metadata_files if p not in self.refused_metadata_files]

    def _get_device_id_fallbacks(self) -> dict[Path, str]:
        """Map each metadata.csv path to a device_id, for files with none of their own.

        If metadata files sit under multiple distinct top-level subdirectories of root_path
        (e.g. root/camera01/..., root/camera02/...), each file's subdirectory name is used.
        Otherwise every file falls back to the dataset root folder's own name.
        """
        if self._device_id_fallbacks is None:
            top_level: dict[Path, str | None] = {}
            for meta_path in self.metadata_files:
                rel_parts = meta_path.relative_to(self.root_path).parts
                top_level[meta_path] = rel_parts[0] if len(rel_parts) > 1 else None

            distinct = {v for v in top_level.values() if v is not None}
            if len(distinct) > 1:
                self._device_id_fallbacks = {
                    p: (v if v is not None else self.root_path.name) for p, v in top_level.items()
                }
                logger.info(
                    "Metadata file(s) without a device_id column will use subdirectory names as "
                    "fallback (%d distinct device ID(s) inferred)",
                    len(distinct)
                )
            else:
                self._device_id_fallbacks = dict.fromkeys(self.metadata_files, self.root_path.name)
                logger.info(
                    "Metadata file(s) without a device_id column will use dataset folder name %r "
                    "as fallback",
                    self.root_path.name
                )

        return self._device_id_fallbacks

    def _analyze_metadata_and_categorize_frames(
        self,
        all_full_frames: list[Path],
        full_frame_index: dict[str, Path]
    ) -> None:
        """Analyze metadata files and categorize frames.

        Populates detection_frame_map, crop_only_map, camera_info, and the detection/frame
        count fields (total_detections, detections_with_frames, etc.). Falls back to
        categorizing every frame as timelapse if no metadata files exist.
        """
        if not self.metadata_files:
            self.timelapse_frames.extend(all_full_frames)
            if self.refused_metadata_files:
                logger.warning("No usable metadata files (%d refused)", len(self.refused_metadata_files))
            else:
                logger.warning("No metadata files found")
            logger.info("Categorized all %d frames as timelapse frames", len(all_full_frames))
            return

        required_cols = {"device_id", "timestamp", "track_id", "label"}
        total_detections = 0
        camera_info: dict[str, CameraInfo] = {}
        detection_frame_timestamps: set[str] = set()
        timestamp_offsets: dict[Path, list[str]] = {}
        stats = {"with_frame": 0, "crop_only": 0, "missing_both": 0}
        unlinkable_files = 0
        unlinkable_detections = 0

        # Build crop index: (ts_filename, track_id) -> crop_path
        crop_index: dict[tuple[str, int], Path] = {}
        for crop_path in self.crop_frames:
            try:
                parsed = parse_crop_stem(crop_path.stem)
            except ValueError:
                logger.warning("Could not parse crop filename stem %r", crop_path.stem)
                continue
            if parsed:
                crop_index[parsed] = crop_path

        for meta_path in self.metadata_files:
            try:
                rows, with_frame, crop_only, missing_both, has_file_path_col = self._process_metadata_file(
                    meta_path, required_cols, full_frame_index, crop_index,
                    camera_info, detection_frame_timestamps, timestamp_offsets
                )
                total_detections += rows
                stats["with_frame"] += with_frame
                stats["crop_only"] += crop_only
                stats["missing_both"] += missing_both
                if has_file_path_col:
                    unlinkable_files += 1
                    unlinkable_detections += rows

            except pl.exceptions.ColumnNotFoundError as e:
                logger.warning("Metadata %s missing required columns: %s", meta_path.name, e)
            except pl.exceptions.NoDataError:
                logger.warning("Skipping empty metadata file %s", meta_path.name)
            except Exception:
                logger.exception("Could not process %s", meta_path.name)

        if unlinkable_files > 0:
            logger.info(
                "%d detections use crops only (legacy format, full frames not linkable) "
                "- expected, not an error",
                unlinkable_detections
            )

        if timestamp_offsets:
            offsets = sorted({offset for file_offsets in timestamp_offsets.values() for offset in file_offsets})
            logger.warning(
                "Timestamps in %d of %d metadata files carry a UTC offset (%s); "
                "kept as written, not converted to local time",
                len(timestamp_offsets), len(self.metadata_files), ", ".join(offsets)
            )

        for frame_path in all_full_frames:
            if frame_path.stem in detection_frame_timestamps:
                self.detection_frames.append(frame_path)
            else:
                self.timelapse_frames.append(frame_path)

        logger.info(
            "Categorized: %d detection frames, %d timelapse frames",
            len(self.detection_frames),
            len(self.timelapse_frames)
        )

        # Store results
        self.camera_info = camera_info
        self.total_detections = total_detections
        self.detections_with_frames = stats["with_frame"]
        self.detections_with_crops_only = stats["crop_only"]
        self.detections_missing_both = stats["missing_both"]
        self.detection_frames_missing = stats["crop_only"] + stats["missing_both"]
        self.legacy_unlinkable_files = unlinkable_files
        self.legacy_unlinkable_detections = unlinkable_detections

        logger.info(
            "Detection sources: %d with frames, %d crop-only, %d missing both",
            stats["with_frame"],
            stats["crop_only"],
            stats["missing_both"]
        )
        if stats["missing_both"] > 0:
            logger.warning(
                "%d detections have neither a detection frame nor a crop file available",
                stats["missing_both"]
            )

    def _process_metadata_file(
        self,
        meta_path: Path,
        required_cols: set[str],
        full_frame_index: dict[str, Path],
        crop_index: dict[tuple[str, int], Path],
        camera_info: dict[str, CameraInfo],
        detection_frame_timestamps: set[str],
        timestamp_offsets: dict[Path, list[str]]
    ) -> tuple[int, int, int, int, bool]:
        """Process a single metadata file, updating detection_frame_map/crop_only_map (self)
        and the caller-owned camera_info/detection_frame_timestamps/timestamp_offsets accumulators in place.

        Returns (rows, with_frame, crop_only, missing_both, has_file_path_col) for this file.
        """
        header_df = pl.read_csv(meta_path, n_rows=0)
        col_map = {col: LEGACY_COLUMNS.get(col.lower(), col.lower()) for col in header_df.columns}

        # Current format has a dedicated filename column; legacy format derives stem from timestamp
        has_filename_col = "filename" in col_map.values()

        # Legacy format: full-frame and metadata timestamps were captured independently,
        # so full-frame linking is impossible for these files and isn't attempted
        has_file_path_col = "file_path" in col_map.values()

        cols_to_load = required_cols | ({"filename"} if has_filename_col else set())
        required_original = [col for col in header_df.columns if col_map[col] in cols_to_load]
        df = pl.read_csv(meta_path, columns=required_original)
        df = rename_legacy_columns(df)

        if "device_id" not in df.columns:
            device_id = self._get_device_id_fallbacks()[meta_path]
            df = df.with_columns(pl.lit(device_id).alias("device_id")).select(
                ["device_id"] + [c for c in df.columns if c != "device_id"]
            )

        if len(df) == 0:
            return 0, 0, 0, 0, has_file_path_col

        # Drop invalid rows
        df = _sanitize_metadata_rows(df, meta_path.name)

        if len(df) == 0:
            return 0, 0, 0, 0, has_file_path_col

        offsets = df.select(timestamp_offset("timestamp").drop_nulls().unique().sort()).to_series().to_list()
        if offsets:
            timestamp_offsets[meta_path] = offsets

        if has_filename_col:
            # Current format: filename column provides the frame stem directly
            parsed_ts, _ = parse_timestamp_column("timestamp")
            df_parsed = df.with_columns([
                parsed_ts.alias("timestamp_parsed"),
                parsed_ts.dt.strftime(CANONICAL_TIMESTAMP_FORMAT).alias("ts_canonical"),
                pl.col("filename").str.replace(r"\.jpg$", "").alias("ts_filename")
            ])
        else:
            # Legacy format: derive frame stem from timestamp
            parsed_ts, is_legacy_ts = parse_timestamp_column("timestamp")
            df_parsed = (
                df
                .with_columns([
                    parsed_ts.alias("timestamp_parsed"),
                    parsed_ts.dt.strftime(CANONICAL_TIMESTAMP_FORMAT).alias("ts_canonical"),
                    is_legacy_ts.alias("_is_legacy_ts")
                ])
                .with_columns(
                    pl.when(pl.col("_is_legacy_ts"))
                    .then(pl.col("timestamp"))
                    .otherwise(pl.col("timestamp_parsed").dt.strftime(FILENAME_TIMESTAMP_FORMAT))
                    .alias("ts_filename")
                )
                .drop("_is_legacy_ts")
            )

        # Camera/device stats
        camera_stats = (
            df_parsed
            .group_by("device_id")
            .agg([
                pl.len().alias("count"),
                pl.min("timestamp_parsed").alias("ts_min"),
                pl.max("timestamp_parsed").alias("ts_max")
            ])
        )

        for row in camera_stats.iter_rows(named=True):
            cam_id = row["device_id"]
            if cam_id not in camera_info:
                camera_info[cam_id] = CameraInfo()

            info = camera_info[cam_id]
            info.metadata_files.add(meta_path)
            info.detections += row["count"]

            current_min, current_max = row["ts_min"], row["ts_max"]
            if info.date_range is None:
                info.date_range = (current_min, current_max)
            else:
                old_start, old_end = info.date_range
                info.date_range = (min(old_start, current_min), max(old_end, current_max))

        # Extract columns once to avoid per-row overhead in the loop below
        cam_ids = df_parsed["device_id"].to_list()
        ts_filenames = df_parsed["ts_filename"].to_list()
        ts_canonicals = df_parsed["ts_canonical"].to_list()
        track_ids = df_parsed["track_id"].to_numpy()
        labels = df_parsed["label"].to_list()

        counters = {"with_frame": 0, "crop_only": 0, "missing_both": 0}

        for i in range(len(df_parsed)):
            cam_id = cam_ids[i]
            ts_filename = ts_filenames[i]
            ts_canonical = ts_canonicals[i]
            track_id = track_ids[i]

            detection_frame_timestamps.add(ts_filename)

            # Build detection_frame_map for image processing (skipped for legacy format)
            frame_key = (cam_id, ts_canonical)
            if not has_file_path_col and frame_key not in self.detection_frame_map:
                frame_path = full_frame_index.get(ts_filename)
                if frame_path:
                    self.detection_frame_map[frame_key] = frame_path

            # Check availability
            has_frame = frame_key in self.detection_frame_map
            crop_key = (ts_filename, track_id)
            crop_path = crop_index.get(crop_key)

            if has_frame:
                counters["with_frame"] += 1
            elif crop_path:
                # Store crop-only detection for later copying
                self.crop_only_map[crop_key] = CropOnlyEntry(path=crop_path, label=str(labels[i]))
                counters["crop_only"] += 1
            else:
                counters["missing_both"] += 1

        return (
            len(df_parsed), counters["with_frame"], counters["crop_only"], counters["missing_both"],
            has_file_path_col
        )

    def get_metadata_df(self, reload: bool = False) -> pl.DataFrame | None:
        """Load, harmonize, and cache metadata merged from all metadata files.

        Args:
            reload: Force a fresh reload instead of returning the cached DataFrame.

        Returns:
            The merged DataFrame, or None if no metadata files exist or all failed to load.
        """
        if self._metadata_df is None or reload:
            if not self.metadata_files:
                return None

            try:
                df_list = []
                for meta_path in self.metadata_files:
                    try:
                        df = pl.read_csv(meta_path)
                        df = rename_legacy_columns(df)
                        if "device_id" not in df.columns:
                            device_id = self._get_device_id_fallbacks()[meta_path]
                            df = df.with_columns(pl.lit(device_id).alias("device_id")).select(
                                ["device_id"] + [c for c in df.columns if c != "device_id"]
                            )
                        if "file_path" in df.columns:
                            # Drop file_path column from legacy format
                            df = df.drop("file_path")
                        if "timestamp" in df.columns:
                            parsed_ts, _ = parse_timestamp_column("timestamp")
                            df = df.with_columns(parsed_ts.alias("timestamp"))
                        df = _sanitize_metadata_rows(df, meta_path.name)
                        if "timestamp" in df.columns:
                            df = df.with_columns(
                                pl.col("timestamp").dt.strftime(CANONICAL_TIMESTAMP_FORMAT)
                            )

                        config_path = _find_session_config_json(meta_path)
                        if config_path is not None:
                            setting = _load_deployment_setting(config_path)
                            if setting is not None:
                                df = df.with_columns(pl.lit(setting).alias(SETTING_COLUMN))

                        if len(df) > 0:
                            df_list.append(df)
                    except pl.exceptions.NoDataError:
                        logger.warning("Skipping empty metadata file %s", meta_path.name)
                        continue
                    except Exception:
                        logger.exception("Skipping invalid metadata file %s", meta_path.name)
                        continue

                if df_list:
                    # diagonal_relaxed: files may have different columns or mismatched dtypes
                    df_concat = pl.concat(df_list, how="diagonal_relaxed", rechunk=True)
                    if SETTING_COLUMN in df_concat.columns:
                        # Move the setting column to immediately after "session_id" if present
                        cols = [c for c in df_concat.columns if c != SETTING_COLUMN]
                        idx = cols.index("session_id") + 1 if "session_id" in cols else 0
                        df_concat = df_concat.select([*cols[:idx], SETTING_COLUMN, *cols[idx:]])
                    sort_cols = [c for c in _DETECTION_KEY_COLUMNS if c in df_concat.columns]
                    self._metadata_df = df_concat.sort(sort_cols, maintain_order=True) if sort_cols else df_concat
                    logger.info(
                        "Loaded merged metadata: %d rows from %d of %d files",
                        len(df_concat), len(df_list), len(self.metadata_files)
                    )
                else:
                    self._metadata_df = None
                    logger.warning("No usable metadata found across %d file(s)", len(self.metadata_files))

            except Exception:
                logger.exception("Failed to load metadata")
                return None

        return self._metadata_df

    def get_summary_str(self, include_details: bool = True) -> str:
        """Generate comprehensive summary string for logging.

        Args:
            include_details: Include per-camera and post-processing details
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Dataset: {self.root_path.name}")
        lines.append("=" * 60)

        if self.has_metadata:
            num_cameras = len(self.camera_info)

            # Basic stats
            if num_cameras > 1:
                lines.append(f"  Camera traps: {num_cameras}")
            refused = f" ({len(self.refused_metadata_files)} refused)" if self.refused_metadata_files else ""
            lines.append(f"  Total number of metadata files: {self.num_metadata_files}{refused}")
            lines.append(f"  Total number of detection frames: {self.total_detection_frames}")
            if self.total_timelapse_frames > 0:
                lines.append(f"  Total number of timelapse frames: {self.total_timelapse_frames}")
            lines.append(f"  Total number of detections: {self.total_detections}")

            # Date range
            if self.date_range:
                start, end = self.date_range
                lines.append(f"  Start date: {start.strftime('%Y-%m-%d')}")
                lines.append(f"    End date: {end.strftime('%Y-%m-%d')}")

            # Per-camera details (optional)
            if include_details and num_cameras > 1:
                lines.append("")
                for cam_id, cam_info in sorted(self.camera_info.items()):
                    lines.append(f"  Camera: {cam_id}")
                    lines.append(f"    Total number of metadata files: {cam_info.num_metadata_files}")
                    lines.append(f"    Total number of detections: {cam_info.detections}")
                    if cam_info.date_range:
                        start, end = cam_info.date_range
                        lines.append(f"    Start date: {start.strftime('%Y-%m-%d')}")
                        lines.append(f"      End date: {end.strftime('%Y-%m-%d')}")
                    lines.append("")

            # Data availability
            if self.detection_frames_missing > 0:
                lines.append("-" * 60)
                lines.append("Data Availability:")
                lines.append(f"  {self.detections_with_frames} detections with detection frames available")
                if self.legacy_unlinkable_files > 0:
                    lines.append(
                        f"  {self.legacy_unlinkable_detections} detections from "
                        f"{self.legacy_unlinkable_files} legacy metadata file(s) "
                        f"will use crops copied from source data"
                    )
                other_crops_only = self.detections_with_crops_only - self.legacy_unlinkable_detections
                if other_crops_only > 0:
                    lines.append(f"  {other_crops_only} detections with crop frames only (no detection frame)")
                if self.detections_missing_both > 0:
                    lines.append(f"  {self.detections_missing_both} detections missing both detection and crop frames")

        else:
            # No metadata case
            if self.total_timelapse_frames > 0:
                lines.append(f"  Total number of timelapse frames: {self.total_timelapse_frames}")
            else:
                lines.append("  No image files found!")
            if self.refused_metadata_files:
                lines.append(f"  No usable metadata files ({len(self.refused_metadata_files)} refused)!")
            else:
                lines.append("  No metadata files found!")

        if self.refused_metadata_files:
            lines.append("-" * 60)
            lines.append("Refused metadata files (not processed):")
            for meta_path, reason in sorted(self.refused_metadata_files.items()):
                lines.append(f"  {meta_path.relative_to(self.root_path).as_posix()}: {reason}")

        lines.append("=" * 60)
        return "\n".join(lines)

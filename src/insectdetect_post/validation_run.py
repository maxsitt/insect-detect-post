"""Read-only access to a processed run for validation.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Reads the track predictions ('*_classified_final.csv'), the candidates ('*_classified_candidates.csv')
and the per-image predictions with crop paths ('*_classified.csv'). Key columns are read as strings.

Classes:
    ValidationRun: One run's output for validation.

Functions:
    track_id_order(): Sort key of track IDs (numeric order).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from insectdetect_post.constants import (
    SETTING_COLUMN,
    TRACK_KEY_COLUMNS,
    VALIDATION_SAME_INDIVIDUAL_OVERLAP_S,
    VALIDATION_SAME_INDIVIDUAL_WINDOW_S,
    OutputLayout,
)
from insectdetect_post.exceptions import RunSearchCancelled, ValidationRunError
from insectdetect_post.validation_taxonomy import VALIDATION_RANKS, file_provenance

# Create module-level logger
logger = logging.getLogger(__name__)

# Suffixes of the metadata CSVs written by the pipeline
_FINAL_SUFFIX = "_classified_final.csv"
_CANDIDATES_SUFFIX = "_classified_candidates.csv"
_CLASSIFIED_SUFFIX = "_classified.csv"

# Directories skipped when searching for a run
_SKIP_WALK_DIRS = {"images", "validation", "__pycache__", ".git"}

# Per-image columns read (if present)
_IMAGE_BASE_COLUMNS = [*TRACK_KEY_COLUMNS, "timestamp", "confidence", "x_min", "y_min", "x_max", "y_max",
                      "filename", "crop_path", "bioclip_top1_prob", "bioclip_top2_prob"]

# Key columns as strings in every frame, so joins match
_KEY_SCHEMA = dict.fromkeys(TRACK_KEY_COLUMNS, pl.String) | {SETTING_COLUMN: pl.String}

# Type alias of a track key: the values of TRACK_KEY_COLUMNS (device_id, session_id, track_id) as strings
TrackKey = tuple[str, str, str]


@dataclass
class ValidationRun:
    """One run's output for validation.

    Attributes:
        root: Run output root directory.
        config: Config snapshot of the run ('<timestamp>_<config>.json'), empty if missing.
        final_path: Path to the per-track '*_classified_final.csv'.
        tracks: One row per track (final predictions), key columns as strings.
        candidates: Candidate predictions per track (may be empty).
        images: One row per image with crop path and per-image predictions.
        active_ranks: Ranks of VALIDATION_RANKS present as 'bioclip_<rank>' in 'tracks'.
        model_rank: Deepest rank the run was classified to.
        frame_size_mm: (width, height) of the camera frame in mm if size estimation was enabled.
        final_provenance: 'file, bytes, sha256' of the final CSV when the run was opened.
        device_numbers: Number of each device by its alphabetical position (1-based), shown as 'D<n>'.
        track_start_times: Start time of each track as read from the final CSV (None if missing).
    """
    root: Path
    config: dict[str, Any]
    final_path: Path
    tracks: pl.DataFrame
    candidates: pl.DataFrame
    images: pl.DataFrame
    active_ranks: list[str]
    model_rank: str
    frame_size_mm: tuple[float, float] | None = None
    final_provenance: dict[str, Any] = field(default_factory=dict)
    device_numbers: dict[str, int] = field(default_factory=dict)
    track_start_times: dict[TrackKey, Any] = field(default_factory=dict)
    _row_index: dict[TrackKey, int] = field(default_factory=dict, repr=False)
    _image_spans: dict[TrackKey, tuple[int, int]] = field(default_factory=dict, repr=False)
    _candidate_spans: dict[TrackKey, tuple[int, int]] = field(default_factory=dict, repr=False)
    _timeline: dict[tuple[str, str], list[tuple[datetime, datetime, TrackKey]]] = field(
        default_factory=dict, repr=False
    )
    _by_taxon: dict[tuple[str, str], list[tuple[TrackKey, float | None]]] = field(default_factory=dict, repr=False)
    _best_crops: dict[tuple[TrackKey, str | None], Path | None] = field(default_factory=dict, repr=False)

    @staticmethod
    def find_final_csvs(root: Path, limit: int = 100, should_stop: Callable[[], bool] | None = None) -> list[Path]:
        """Return every '*_classified_final.csv' at or below a directory, sorted by path.

        Checks '<root>/metadata' and '<root>' first, then walks the tree below 'root'.

        Args:
            root: Directory to search.
            limit: Stop after this many matches.
            should_stop: Checked before every directory, 'True' ends the walk.

        Returns:
            Sorted final CSV paths (empty if none).
        """
        root = Path(root)
        for directory in (OutputLayout(root).metadata_dir, root):
            if directory.is_dir():
                matches = sorted(directory.glob(f"*{_FINAL_SUFFIX}"))
                if matches:
                    return matches[:limit]
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            if should_stop is not None and should_stop():
                break
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP_WALK_DIRS)
            found.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(_FINAL_SUFFIX))
            if len(found) >= limit:
                break
        return sorted(found)[:limit]

    @staticmethod
    def find_final_csv(root: Path) -> Path | None:
        """Return the single '*_classified_final.csv' at or below a directory, None if none or several."""
        matches = ValidationRun.find_final_csvs(root, limit=2)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def run_root_from_final_csv(final_path: Path) -> Path:
        """Return the run root of a final CSV: the parent of its 'metadata' directory, else its directory."""
        parent = Path(final_path).parent
        root = parent.parent
        return root if parent == OutputLayout(root).metadata_dir else parent

    @classmethod
    def open(cls, root: Path, should_stop: Callable[[], bool] | None = None) -> ValidationRun:
        """Open a run directory for validation.

        Args:
            root: Run output root, or a directory that contains exactly one run.
            should_stop: Cancels the search for the run.

        Raises:
            ValidationRunError: If required files are missing or the run has no BioCLIP results.
            RunSearchCancelled: If 'should_stop' ended the search.

        Returns:
            Loaded ValidationRun.
        """
        root = Path(root)
        matches = cls.find_final_csvs(root, limit=2, should_stop=should_stop)
        if should_stop is not None and should_stop():
            raise RunSearchCancelled(str(root))
        if not matches:
            raise ValidationRunError(
                f"No '*{_FINAL_SUFFIX}' found in '{root}'. Select a run directory created by the "
                "post-processing pipeline with BioCLIP classification enabled."
            )
        if len(matches) > 1:
            names = ", ".join(sorted({cls.run_root_from_final_csv(m).name for m in matches}))
            raise ValidationRunError(
                f"Several processed runs found in '{root.name}' ({names}, ...). Select an individual "
                "run directory, not a directory that contains several runs."
            )
        final_path = matches[0]
        root = cls.run_root_from_final_csv(final_path)  # a parent directory may have been selected
        stem = final_path.name.removesuffix(_FINAL_SUFFIX)
        classified_path = final_path.with_name(f"{stem}{_CLASSIFIED_SUFFIX}")
        candidates_path = final_path.with_name(f"{stem}{_CANDIDATES_SUFFIX}")
        if not classified_path.exists():
            raise ValidationRunError(f"Per-image classification CSV not found: '{classified_path}'.")

        config = cls._read_config_snapshot(root)

        tracks = _drop_keyless_rows(pl.read_csv(final_path, schema_overrides=_KEY_SCHEMA), final_path.name)
        active_ranks = [r for r in VALIDATION_RANKS if f"bioclip_{r}" in tracks.columns]
        if not active_ranks:
            raise ValidationRunError(
                "This run has no BioCLIP taxonomy columns (bioclip_<rank>). Validation currently supports "
                "runs classified with BioCLIP only."
            )
        model_rank = active_ranks[-1]
        tracks = tracks.with_columns(pl.col("pred_prob_weighted").cast(pl.Float64, strict=False))
        candidates = (
            _drop_keyless_rows(pl.read_csv(candidates_path, schema_overrides=_KEY_SCHEMA), candidates_path.name)
            if candidates_path.exists() else pl.DataFrame()
        )
        if not candidates.is_empty():
            sort_cols = [*TRACK_KEY_COLUMNS]
            if "candidate_prob_weighted" in candidates.columns:
                sort_cols.append("candidate_prob_weighted")
            descending = [False] * len(TRACK_KEY_COLUMNS) + [True] * (len(sort_cols) - len(TRACK_KEY_COLUMNS))
            candidates = candidates.sort(sort_cols, descending=descending, nulls_last=True, maintain_order=True)

        header = pl.read_csv(classified_path, n_rows=0).columns
        rank_cols = [c for c in header if c.startswith(("bioclip_top1_", "bioclip_top2_"))]
        wanted = [*(c for c in _IMAGE_BASE_COLUMNS if c in header),
                  *(c for c in rank_cols if c not in _IMAGE_BASE_COLUMNS)]
        images = _drop_keyless_rows(pl.read_csv(classified_path, columns=wanted, schema_overrides=_KEY_SCHEMA),
                                    classified_path.name)
        if "crop_path" not in images.columns:
            raise ValidationRunError(f"'{classified_path.name}' has no 'crop_path' column - crops were not saved.")
        images = images.sort([*TRACK_KEY_COLUMNS, "timestamp"])

        frame_size_mm = cls._frame_size_mm(config)
        if frame_size_mm and {"x_min", "x_max", "y_min", "y_max"} <= set(images.columns):
            width_mm, height_mm = frame_size_mm
            images = images.with_columns(
                ((pl.col("x_max") - pl.col("x_min")) * width_mm).alias("_size_x"),
                ((pl.col("y_max") - pl.col("y_min")) * height_mm).alias("_size_y"),
            ).with_columns(
                pl.max_horizontal("_size_x", "_size_y").round(2).alias("bbox_length_mm"),
                pl.min_horizontal("_size_x", "_size_y").round(2).alias("bbox_width_mm"),
            ).drop(["_size_x", "_size_y"])

        run = cls(
            root=root, config=config, final_path=final_path, tracks=tracks, candidates=candidates,
            images=images, active_ranks=active_ranks, model_rank=model_rank, frame_size_mm=frame_size_mm,
            final_provenance=file_provenance(final_path),
            device_numbers={device: i + 1 for i, device in enumerate(sorted(set(tracks["device_id"].to_list())))},
        )
        run._row_index = {key: i for i, key in enumerate(tracks.select(TRACK_KEY_COLUMNS).iter_rows())}
        run._image_spans = _spans(images)
        run._candidate_spans = _spans(candidates) if not candidates.is_empty() else {}
        run._timeline = _timeline(tracks)
        if "start_time" in tracks.columns:
            run.track_start_times = {tuple(row[:3]): row[3] for row in  # type: ignore[misc]
                                     tracks.select([*TRACK_KEY_COLUMNS, "start_time"]).iter_rows()}
        logger.info("Opened run '%s': %d tracks, %d images, model rank '%s'",
                    root.name, tracks.height, images.height, model_rank)
        return run

    @staticmethod
    def _read_config_snapshot(root: Path) -> dict[str, Any]:
        """Read the post-processing run's config snapshot JSON, if present."""
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and "classification" in data:
                return data
        logger.warning("No config snapshot found in '%s'", root)
        return {}

    @staticmethod
    def _frame_size_mm(config: dict[str, Any]) -> tuple[float, float] | None:
        """Return the frame size in mm from the config snapshot, None without size estimation."""
        size_cfg = config.get("metadata", {}).get("estimate_size", {})
        if not size_cfg.get("enabled"):
            return None
        try:
            return float(size_cfg["frame_width_mm"]), float(size_cfg["frame_height_mm"])
        except (KeyError, TypeError, ValueError):
            return None

    @property
    def has_size_estimate(self) -> bool:
        """Whether track-level size estimates (mm) are available."""
        return "bbox_length_mm_mean" in self.tracks.columns

    @property
    def n_tracks_classified(self) -> int:
        """Number of classified tracks before the run's track filters."""
        return self.images.select(TRACK_KEY_COLUMNS).n_unique()

    def track_row(self, key: TrackKey) -> dict[str, Any] | None:
        """Return the final-CSV row of a track as a dict, or None if unknown."""
        i = self._row_index.get(key)
        return self.tracks.row(i, named=True) if i is not None else None

    def images_for(self, key: TrackKey) -> pl.DataFrame:
        """Return the per-image rows of a track, sorted by timestamp."""
        span = self._image_spans.get(key)
        return self.images.slice(*span) if span is not None else self.images.clear()

    def candidates_for(self, key: TrackKey) -> pl.DataFrame:
        """Return the candidate predictions of a track by weighted probability (may be empty)."""
        span = self._candidate_spans.get(key)
        return self.candidates.slice(*span) if span is not None else self.candidates.clear()

    def crop_abs_path(self, crop_path: str | None) -> Path | None:
        """Return the absolute path of a 'crop_path' value (relative to the run root)."""
        if not crop_path:
            return None
        path = Path(crop_path)
        return path if path.is_absolute() else self.root / path

    def best_crop(self, key: TrackKey, taxon: str | None = None) -> Path | None:
        """Return the crop of a track that shows 'taxon' with the highest classification probability.

        Falls back to the most confident crop of any taxon, then to the first crop.

        Args:
            key: Track.
            taxon: Taxon at the model rank, defaults to the track's prediction.

        Returns:
            Absolute crop path, or None if the track has no crops.
        """
        images = self.images_for(key)
        if images.is_empty():
            return None
        if taxon is None:
            taxon = (self.track_row(key) or {}).get(f"bioclip_{self.model_rank}")
        if (key, taxon) in self._best_crops:
            return self._best_crops[(key, taxon)]
        path = None
        for slot in (1, 2):
            name, prob = f"bioclip_top{slot}_{self.model_rank}", f"bioclip_top{slot}_prob"
            if not taxon or name not in images.columns or prob not in images.columns:
                continue
            matching = images.filter(pl.col(name) == taxon).drop_nulls(prob)
            if not matching.is_empty():
                path = self.crop_abs_path(_first_max(matching, prob))
                break
        if path is None and "bioclip_top1_prob" in images.columns:
            ranked = images.drop_nulls("bioclip_top1_prob")
            if not ranked.is_empty():
                path = self.crop_abs_path(_first_max(ranked, "bioclip_top1_prob"))
        if path is None:
            path = self.crop_abs_path(images["crop_path"][0])
        self._best_crops[(key, taxon)] = path
        return path

    def previous_tracks(
        self,
        key: TrackKey,
        window_s: float = VALIDATION_SAME_INDIVIDUAL_WINDOW_S,
        overlap_s: float = VALIDATION_SAME_INDIVIDUAL_OVERLAP_S,
    ) -> list[dict[str, Any]]:
        """Return the earlier tracks of the same device/session that ended shortly before 'key' started.

        Candidates for a "same individual" link. Earlier means an earlier start, or the same start and a lower
        track ID (the tracker assigns IDs in ascending order, see sort_tracks_chronologically()). One individual
        can show up as two tracks in the same frame (a box lagging behind a fast insect), so overlaps up to
        'overlap_s' are allowed.

        Args:
            key: Track.
            window_s: Maximum gap (s) between the earlier track's end and this track's start.
            overlap_s: Maximum overlap (s) of an earlier track.

        Returns:
            Dicts with 'key', 'track_id', 'gap_s' (negative = overlap), 'track_imgs', 'pred',
            'pred_prob_weighted', 'start_time', 'end_time', sorted by the gap (nearest first).
        """
        row = self.track_row(key)
        if row is None:
            return []
        start, end = _parse_ts(row.get("start_time")), _parse_ts(row.get("end_time"))
        if start is None or end is None:
            return []
        result: list[dict[str, Any]] = []
        for _o_start, o_end, other_key in self._timeline.get((key[0], key[1]), []):
            if other_key == key:
                break  # sorted by start, then track ID: the rest is later
            gap = _candidate_gap(start, end, o_end, window_s, overlap_s)
            if gap is None:
                continue
            other = self.tracks.row(self._row_index[other_key], named=True)
            result.append({
                "key": other_key, "track_id": other_key[2],
                "gap_s": round(gap, 1), "track_imgs": other.get("track_imgs"),
                "pred": other.get("pred"), "pred_prob_weighted": other.get("pred_prob_weighted"),
                "start_time": other.get("start_time"), "end_time": other.get("end_time"),
            })
        result.sort(key=lambda r: abs(r["gap_s"]))
        return result

    def is_link_candidate(
        self,
        earlier: TrackKey,
        later: TrackKey,
        window_s: float = VALIDATION_SAME_INDIVIDUAL_WINDOW_S,
        overlap_s: float = VALIDATION_SAME_INDIVIDUAL_OVERLAP_S,
    ) -> bool:
        """Return whether 'earlier' is among the previous_tracks() of 'later' (same rule, one pair)."""
        if earlier[:2] != later[:2] or earlier == later:
            return False
        rows = self.track_row(earlier), self.track_row(later)
        if rows[0] is None or rows[1] is None:
            return False
        o_start, o_end = _parse_ts(rows[0].get("start_time")), _parse_ts(rows[0].get("end_time"))
        start, end = _parse_ts(rows[1].get("start_time")), _parse_ts(rows[1].get("end_time"))
        if o_start is None or o_end is None or start is None or end is None:
            return False
        if (o_start, track_id_order(earlier[2])) >= (start, track_id_order(later[2])):
            return False  # not earlier, see previous_tracks()
        return _candidate_gap(start, end, o_end, window_s, overlap_s) is not None

    def link_partners(
        self,
        keys: Collection[TrackKey],
        window_s: float = VALIDATION_SAME_INDIVIDUAL_WINDOW_S,
        overlap_s: float = VALIDATION_SAME_INDIVIDUAL_OVERLAP_S,
    ) -> dict[TrackKey, dict[TrackKey, float]]:
        """Return the link candidates of each track in 'keys' among 'keys', earlier and later (rule of previous_tracks()).

        Args:
            keys: Tracks to pair, e.g. the validation queue.
            window_s: Maximum gap (s) between the earlier track's end and the later track's start.
            overlap_s: Maximum overlap (s) of the earlier track.

        Returns:
            Partner -> gap (s, negative = overlap, as 'gap_s' of previous_tracks()) per track,
            only tracks with at least one partner.
        """
        partners: dict[TrackKey, dict[TrackKey, float]] = {}
        for entries in self._timeline.values():
            active: list[tuple[datetime, datetime, TrackKey]] = []
            for start, end, key in entries:  # sorted by start, then track ID: all active tracks are earlier
                if key not in keys:
                    continue
                active = [a for a in active if (start - a[1]).total_seconds() <= window_s]
                for _o_start, o_end, other in active:
                    gap = _candidate_gap(start, end, o_end, window_s, overlap_s)
                    if gap is not None:
                        partners.setdefault(key, {})[other] = partners.setdefault(other, {})[key] = round(gap, 1)
                active.append((start, end, key))
        return partners

    def model_lineage(self, row: dict[str, Any]) -> dict[str, str | None]:
        """Return the model taxon per validation rank of a track row."""
        return {rank: row.get(f"bioclip_{rank}") for rank in VALIDATION_RANKS}

    def tracks_predicted_as(self, rank: str, taxon: str) -> list[tuple[TrackKey, float | None]]:
        """Return the tracks predicted as 'taxon' at 'rank' with their weighted probability, highest first."""
        cached = self._by_taxon.get((rank, taxon))
        if cached is None:
            col = f"bioclip_{rank}"
            cached = []
            if col in self.tracks.columns:
                same = self.tracks.filter(pl.col(col) == taxon).sort(
                    ["pred_prob_weighted", *TRACK_KEY_COLUMNS],
                    descending=[True, *[False] * len(TRACK_KEY_COLUMNS)], nulls_last=True,
                )
                cached = [((d, s, tid), prob) for d, s, tid, prob in
                          same.select([*TRACK_KEY_COLUMNS, "pred_prob_weighted"]).iter_rows()]
            self._by_taxon[(rank, taxon)] = cached
        return cached

    def taxon_counts(self, rank: str, tracks: pl.DataFrame | None = None) -> pl.DataFrame:
        """Return the predicted taxa at a rank with their track counts, largest first.

        Args:
            rank: Rank of the 'bioclip_<rank>' column.
            tracks: Tracks to count, all tracks of the run if None.

        Returns:
            Frame with 'taxon, n_tracks'; empty if the rank is not in 'tracks'.
        """
        tracks = self.tracks if tracks is None else tracks
        col = f"bioclip_{rank}"
        if col not in tracks.columns:
            return pl.DataFrame(schema={"taxon": pl.String, "n_tracks": pl.UInt32})
        return (
            tracks.group_by(col).len().rename({col: "taxon", "len": "n_tracks"})
            .drop_nulls("taxon").sort(["n_tracks", "taxon"], descending=[True, False])
        )

    def track_filters(self) -> dict[str, float] | None:
        """Return the track filters the run applied, as '{setting: threshold}'.

        Returns:
            The active filters (empty if none), or None without a config snapshot.
        """
        if not self.config:
            return None
        meta = self.config.get("metadata", {})
        tracks_cfg = meta.get("filter_tracks", {})
        pred_cfg = meta.get("filter_predictions", {})
        active: dict[str, float] = {}
        if tracks_cfg.get("enabled"):
            for name in ("min_det_conf", "min_dur_s"):
                value = tracks_cfg.get(name)
                if isinstance(value, (int, float)) and value > 0:
                    active[name] = float(value)
            if isinstance(tracks_cfg.get("max_dur_s"), (int, float)):
                active["max_dur_s"] = float(tracks_cfg["max_dur_s"])
        if pred_cfg.get("enabled"):
            value = pred_cfg.get("min_prob_weighted")
            if isinstance(value, (int, float)) and value > 0:
                active["min_prob_weighted"] = float(value)
        return active

    def summary(self) -> dict[str, Any]:
        """Return a summary of the post-processing run for the validation setup page."""
        levels = (0.1, 0.5, 0.9)
        values = self.tracks.select(pl.col("pred_prob_weighted").quantile(q).alias(str(q)) for q in levels).row(0)
        quantiles = {q: round(v, 2) for q, v in zip(levels, values, strict=True) if v is not None}
        filter_cfg = self.config.get("classification", {}).get("bioclip", {}).get("filter_arthropods", {})
        return {
            "run_name": self.root.name,
            "n_tracks": self.tracks.height,
            # Crops of the kept tracks (the per-image CSV also holds removed ones)
            "n_crops": int(self.tracks["track_imgs"].sum()) if "track_imgs" in self.tracks.columns
            else self.images.height,
            "n_tracks_classified": self.n_tracks_classified,
            "filters": self.track_filters(),
            "model_rank": self.model_rank,
            "taxa_per_rank": {r: self.tracks[f"bioclip_{r}"].drop_nulls().n_unique() for r in self.active_ranks},
            "prob_quantiles": quantiles,
            "has_size_estimate": self.has_size_estimate,
            "countries": list(filter_cfg.get("countries", [])),
            "continents": list(filter_cfg.get("continents", [])),
            "taxa_filter": list(filter_cfg.get("taxa", [])),
        }


def _drop_keyless_rows(frame: pl.DataFrame, source_name: str) -> pl.DataFrame:
    """Drop rows without a complete track key: blank or incomplete lines, which polars reads as nulls."""
    if not set(TRACK_KEY_COLUMNS) <= set(frame.columns):
        return frame
    kept = frame.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in TRACK_KEY_COLUMNS))
    if dropped := frame.height - kept.height:
        logger.warning("Dropped %d of %d rows without a track key in '%s' (blank or incomplete lines)",
                       dropped, frame.height, source_name)
    return kept


def _spans(frame: pl.DataFrame) -> dict[TrackKey, tuple[int, int]]:
    """Return '(offset, length)' of every track's rows in a frame sorted by track key."""
    groups = (
        frame.select(TRACK_KEY_COLUMNS).with_row_index("_i")
        .group_by(TRACK_KEY_COLUMNS, maintain_order=True).agg(pl.col("_i").first(), pl.len().alias("_n"))
    )
    return {(d, s, tid): (int(i), int(n)) for d, s, tid, i, n in groups.iter_rows()}


def _timeline(tracks: pl.DataFrame) -> dict[tuple[str, str], list[tuple[datetime, datetime, TrackKey]]]:
    """Return the tracks per device/session in start order, then track ID (tracks without timestamps skipped)."""
    if "start_time" not in tracks.columns or "end_time" not in tracks.columns:
        return {}
    timeline: dict[tuple[str, str], list[tuple[datetime, datetime, TrackKey]]] = {}
    for device, session_id, track_id, start, end in tracks.select(
        [*TRACK_KEY_COLUMNS, "start_time", "end_time"]
    ).iter_rows():
        o_start, o_end = _parse_ts(start), _parse_ts(end)
        if o_start is None or o_end is None:
            continue
        timeline.setdefault((device, session_id), []).append((o_start, o_end, (device, session_id, track_id)))
    for entries in timeline.values():
        entries.sort(key=lambda e: (e[0], track_id_order(e[2][2])))
    return timeline


def track_id_order(track_id: str) -> tuple[int, int | str]:
    """Return a sort key putting numeric track IDs first in numeric order."""
    return (0, int(track_id)) if track_id.isdigit() else (1, track_id)


def _candidate_gap(start: datetime, end: datetime, o_end: datetime, window_s: float, overlap_s: float) -> float | None:
    """Return the gap (s, negative = overlap) of an earlier-starting track ending at 'o_end' if it is a link candidate.

    None if it overlaps the track starting at 'start' by more than 'overlap_s' or ended more than 'window_s' before.
    """
    if o_end <= start:
        gap = (start - o_end).total_seconds()
    else:
        overlap = (min(end, o_end) - start).total_seconds()
        if overlap > overlap_s:
            return None
        gap = -overlap
    return gap if gap <= window_s else None


def _parse_ts(value: Any) -> datetime | None:
    """Parse a timestamp (ISO string or datetime), None if missing or invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _first_max(images: pl.DataFrame, prob: str) -> str | None:
    """Return the 'crop_path' of the first image with the highest 'prob'."""
    return images.select(pl.col("crop_path").get(pl.col(prob).arg_max())).item()

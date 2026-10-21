"""Validation session: verdicts, persistence, resume and validated export.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

A session is stored in the run's 'validation/' directory as '<validator>_<started>_validation.csv'
(one row per sampled track) and '<validator>_<started>_validation.json' (sampling, provenance, UI state),
both rewritten atomically on every save. The validation session ID '<started>' is the start time.

Fields added later to the session files get a default when an older session is read (sampling fields
from SamplingConfig, top-level JSON keys from _META_DEFAULTS, verdict columns stay empty), so the
default must reproduce how sessions behaved before the field existed. Fields unknown to this version
(a session of a newer version) are refused, as the next save would drop them.

Classes:
    SamplingConfig: Parameters that define the validated subsample.
    Verdict: Human verdict of one track.
    ValidationSession: Rows, verdicts and persistence of one session.

Functions:
    safe_validator_name(): Filename-safe validator name.
    new_validation_session_id(): ID of a new session.
    parse_links(), join_links(), with_link(): Read and edit 'same_individual_as' values.
    has_qualifier(), with_qualifier(), allowed_qualifier(), qualifier_mark(): Identification qualifiers.
    derive_verdict(): Verdict type, agreement rank and lowest determined rank.
    verdict_columns(), verdict_frame(): Verdict columns of the session CSV.
    validated_frame(): Final CSV joined with the verdicts of a session table.
    session_files(): Files and directories of a session.
    delete_session(): Remove a session with all its files.
    source_csv_changed(): Whether the run's final CSV changed after a session started.
    open_session_log(), close_session_log(): Session log file.
    sort_tracks_chronologically(): Tracks in time order (start time, then track ID).
    anchor_of(): Track ID naming an individual.
    individual_groups(): Individuals from 'same_individual_as' links.
    now_iso(), app_version(): Time and package version recorded in the session files.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import polars as pl

from insectdetect_post.constants import (
    TRACK_KEY_COLUMNS,
    VALIDATION_EFFORT_IDLE_CAP_S,
    VALIDATION_SEED_DEFAULT,
    OutputLayout,
)
from insectdetect_post.validation_run import TrackKey, ValidationRun, track_id_order
from insectdetect_post.validation_sampling import (
    STRATUM_COLUMNS,
    assign_strata,
    filter_duration,
    filter_periods,
    filter_settings,
    filter_taxa,
    small_strata,
    stratum_summary,
)
from insectdetect_post.validation_taxonomy import (
    INDET_LABEL,
    VALIDATION_RANKS,
    Lineage,
    TaxonomyIndex,
    ranks_up_to,
)

# Create module-level logger
logger = logging.getLogger(__name__)

# File name suffixes
_SESSION_CSV_SUFFIX = "_validation.csv"
_SESSION_JSON_SUFFIX = "_validation.json"
_SESSION_LOG_SUFFIX = "_validation.log"
_HISTORY_SUFFIX = "_history.jsonl"
EXPORT_SUFFIX = "_validated"  # validated export: '<final_stem>_validated_<stem>.csv'

# Validation session ID and export time: the start time (a counter is appended within the same second)
STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# IDs issued in this process (a session reaches the disk only on its first save)
_ISSUED_SESSION_IDS: set[tuple[str, str]] = set()

# Characters replaced in validator names used as filenames
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Derived verdicts (see derive_verdict)
VERDICT_TYPES = ("agree", "partial", "corrected", "false_positive", "indeterminate", "not_assessable")

# Reasons of the low image quality flag
IMAGE_QUALITY_REASONS = ("blurry", "too_small", "partially_visible", "occluded", "other")

# Several individuals in a track: the same taxon or different taxa inside the crops, or different taxa along
# the track (one individual per crop: a tracking error)
DIFFERENT_TAXA_IN_CROPS = "different_taxa_in_crops"
DIFFERENT_TAXA_IN_TRACK = "different_taxa_in_track"
MULTIPLE_INDIVIDUALS_TYPES = ("same_taxon_in_crops", DIFFERENT_TAXA_IN_CROPS, DIFFERENT_TAXA_IN_TRACK)
# Types that leave the model no single target (several taxa in the crops or along the track): not assessable
MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE = (DIFFERENT_TAXA_IN_CROPS, DIFFERENT_TAXA_IN_TRACK)

# Identification qualifiers (open nomenclature): 'inc.' = uncertain determination, 'agg.' = species aggregate
ID_QUALIFIER_INC = "inc."
ID_QUALIFIER_AGG = "agg."
_ID_QUALIFIERS = (ID_QUALIFIER_INC, ID_QUALIFIER_AGG)

# Rank abbreviations written before a qualifier (e.g. 'Syrphidae fam. inc.')
_RANK_ABBREVIATIONS = {"class": "cl.", "order": "ord.", "family": "fam.", "genus": "gen.", "species": "sp."}

# Rank column order in the session CSV (deepest first)
_RANKS_CSV = VALIDATION_RANKS[::-1]

# Distinct recent human lineages kept per session
_RECENT_LINEAGES_KEPT = 32

# Verdict columns of the session CSV
_VERDICT_COLUMNS = [
    "validated", "validated_at", "validator", "verdict",
    "image_quality", "image_quality_reason", "multiple_individuals", "same_individual_as",
    *[f"val_{r}" for r in _RANKS_CSV],
    "val_lowest_rank", "val_qualifier", "val_sex", "val_source", "agree_rank", "notes",
    "seconds_spent",
]

# Model columns copied into the session CSV
_MODEL_COLUMNS = ["model_rank", "pred", "pred_prob_weighted", *[f"pred_{r}" for r in _RANKS_CSV]]

# Required keys of the session JSON (written by ValidationSession.create())
_META_KEYS = frozenset({
    "validator", "validation_session_id", "started", "updated", "app_version", "blind",
    "run_name", "run_root", "final_csv", "taxonomy", "model_rank", "n_tracks_classified", "run_filters",
    "n_population_total", "sampling", "n_population_selected", "strata_excluded", "strata_population", "n_sampled",
    "n_validated",
    "custom_taxa", "ui",
})

# Keys added to the session JSON later, with the value that reproduces older sessions (not in _META_KEYS)
_META_DEFAULTS: dict[str, Any] = {}

# Separator of the track IDs in a 'same_individual_as' value
_LINK_SEPARATOR = ";"


def safe_validator_name(name: str) -> str:
    """Return the filename-safe version of a validator name (letters, digits, '._-')."""
    cleaned = _SAFE_NAME_RE.sub("_", name.strip()).strip("._-")
    return cleaned or "validator"


def new_validation_session_id(run_root: Path, validator: str, now: datetime | None = None) -> str:
    """Return the ID of a new session of 'validator' in 'run_root': the start time, never reused.

    Args:
        run_root: Run output root.
        validator: Validator name.
        now: Start time, defaults to the current local time.

    Returns:
        Validation session ID, e.g. '2026-09-16_14-33-05'.
    """
    stamp = (now or datetime.now()).strftime(STAMP_FORMAT)  # noqa: DTZ005
    directory = OutputLayout(Path(run_root)).validation_dir
    name = safe_validator_name(validator)
    validation_session_id, n = stamp, 1
    while ((str(directory), f"{name}_{validation_session_id}") in _ISSUED_SESSION_IDS
           or (directory / f"{name}_{validation_session_id}{_SESSION_CSV_SUFFIX}").exists()):
        n += 1
        validation_session_id = f"{stamp}-{n}"
    _ISSUED_SESSION_IDS.add((str(directory), f"{name}_{validation_session_id}"))
    return validation_session_id


@dataclass
class SamplingConfig:
    """Parameters of the validated subsample (stored in the session JSON).

    A field added later needs a default that reproduces the sampling of older sessions.
    """
    taxon_filter_rank: str | None = None
    taxon_filter_values: list[str] = field(default_factory=list)
    period_filter_granularity: str | None = None
    period_filter_values: list[str] = field(default_factory=list)
    setting_filter_values: list[str] = field(default_factory=list)
    min_duration_s: int | None = None
    max_duration_s: int | None = None
    stratum_rank: str | None = None
    prob_edges: list[float] | None = None
    stratum_setting: bool = False
    min_per_stratum: int | None = None
    max_per_stratum: int | None = None
    seed: int = VALIDATION_SEED_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters as a dict (session JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingConfig:
        """Create the config from its dict (see to_dict())."""
        return cls(**data)

    def select_tracks(self, tracks: pl.DataFrame) -> pl.DataFrame:
        """Return the tracks selected by the taxa, period, setting and duration filters."""
        tracks = filter_taxa(tracks, self.taxon_filter_rank, self.taxon_filter_values or None)
        tracks = filter_periods(tracks, self.period_filter_granularity, self.period_filter_values or None)
        tracks = filter_settings(tracks, self.setting_filter_values or None)
        return filter_duration(tracks, self.min_duration_s, self.max_duration_s)


@dataclass
class Verdict:
    """Human verdict of one track."""
    validated: bool = False
    validated_at: str | None = None
    false_positive: bool = False
    image_quality: str = "ok"
    image_quality_reason: str | None = None
    multiple_individuals: str | None = None  # one of MULTIPLE_INDIVIDUALS_TYPES
    same_individual_as: str | None = None  # earlier track IDs, separated by _LINK_SEPARATOR
    val: dict[str, str | None] = field(default_factory=lambda: dict.fromkeys(VALIDATION_RANKS))
    val_source: str | None = None
    qualifier: str | None = None  # space-separated _ID_QUALIFIERS, e.g. 'inc. agg.'
    sex: str | None = None  # 'female' or 'male'
    notes: str = ""
    seconds: float = 0.0  # validation effort (managed by the session)

    def copy(self) -> Verdict:
        """Return an independent copy (including the lineage dict)."""
        return Verdict(**{**asdict(self), "val": dict(self.val)})

    @property
    def not_assessable(self) -> bool:
        """Whether the track has low image quality or shows different taxa (in the crops or along the track)."""
        return self.image_quality == "low" or self.multiple_individuals in MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE

    def determinate_ranks(self) -> list[str]:
        """Return the ranks with a taxon name (not empty, not indet.)."""
        return [r for r in VALIDATION_RANKS if self.val.get(r) not in (None, "", INDET_LABEL)]


def parse_links(value: str | None) -> list[str]:
    """Return the track IDs stored in a 'same_individual_as' value ('1;3' -> ['1', '3'])."""
    if not value:
        return []
    seen: list[str] = []
    for part in str(value).split(_LINK_SEPARATOR):
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return seen


def join_links(track_ids: list[str]) -> str | None:
    """Join track IDs into a 'same_individual_as' value (inverse of parse_links(), None if empty)."""
    ids = parse_links(_LINK_SEPARATOR.join(track_ids))
    ids.sort(key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    return _LINK_SEPARATOR.join(ids) or None


def with_link(value: str | None, track_id: str, enabled: bool) -> str | None:
    """Add or remove one track ID in a 'same_individual_as' value."""
    ids = parse_links(value)
    if enabled and track_id not in ids:
        ids.append(track_id)
    elif not enabled:
        ids = [i for i in ids if i != track_id]
    return join_links(ids)


def has_qualifier(qualifier: str | None, which: str) -> bool:
    """Return whether a qualifier string (e.g. 'inc. agg.') contains 'which'."""
    return bool(qualifier) and which in qualifier.split()


def with_qualifier(qualifier: str | None, which: str, enabled: bool) -> str | None:
    """Add or remove one qualifier, in _ID_QUALIFIERS order (None if empty)."""
    present = {q for q in (qualifier or "").split() if q in _ID_QUALIFIERS}
    if enabled:
        present.add(which)
    else:
        present.discard(which)
    return " ".join(q for q in _ID_QUALIFIERS if q in present) or None


def allowed_qualifier(qualifier: str | None, lowest_rank: str | None) -> str | None:
    """Return the qualifiers a determination down to 'lowest_rank' can carry ('agg.' only at species).

    Args:
        qualifier: Stored qualifier string.
        lowest_rank: Deepest determined rank, or None.

    Returns:
        The allowed qualifiers, or None.
    """
    if not qualifier or not lowest_rank:
        return None
    if lowest_rank != "species":
        return with_qualifier(qualifier, ID_QUALIFIER_AGG, False)
    return qualifier


def qualifier_mark(qualifier: str | None, lowest_rank: str | None) -> str | None:
    """Return the open-nomenclature mark after the taxon name, e.g. 'sp. inc.' or 'fam. inc.'.

    Args:
        qualifier: Stored qualifier string.
        lowest_rank: Deepest determined rank, or None.

    Returns:
        The mark, or None.
    """
    qualifier = allowed_qualifier(qualifier, lowest_rank)
    if not qualifier or not lowest_rank:
        return None
    marks = []
    if has_qualifier(qualifier, ID_QUALIFIER_INC):
        marks.append(f"{_RANK_ABBREVIATIONS[lowest_rank]} {ID_QUALIFIER_INC}")
    if has_qualifier(qualifier, ID_QUALIFIER_AGG):
        marks.append(ID_QUALIFIER_AGG)
    return " ".join(marks) or None


def derive_verdict(
    model: Lineage,
    verdict: Verdict,
    model_rank: str,
) -> tuple[str | None, str | None, str | None]:
    """Derive the verdict type, agreement rank and lowest determined rank of a track.

    Verdict types: 'false_positive' if flagged, 'not_assessable' / 'indeterminate' if nothing
    is determined (with / without a not-assessable flag), 'corrected' if a rank down to
    'model_rank' contradicts the model, 'partial' if some are indet., else 'agree'. The species of an
    'agg.' determination counts as indet.: the aggregate may include the model's species.

    Args:
        model: Model lineage (canonical names).
        verdict: Human verdict.
        model_rank: Deepest rank of the run.

    Returns:
        '(verdict_type, agree_rank, val_lowest_rank)', all None if not validated.
    """
    if not verdict.validated:
        return None, None, None
    if verdict.false_positive:
        return "false_positive", None, None

    determinate = verdict.determinate_ranks()
    lowest = determinate[-1] if determinate else None
    if lowest is None:
        return ("not_assessable" if verdict.not_assessable else "indeterminate"), None, None

    agree_rank: str | None = None
    contradicted = False
    incomplete = False
    aggregate = has_qualifier(allowed_qualifier(verdict.qualifier, lowest), ID_QUALIFIER_AGG)
    for rank in ranks_up_to(model_rank):
        model_name = model.get(rank)
        human_name = verdict.val.get(rank)
        if not model_name:
            continue
        if human_name in (None, "", INDET_LABEL) or (aggregate and rank == lowest):
            incomplete = True
            continue
        if human_name == model_name:
            if not incomplete and not contradicted:
                agree_rank = rank
        else:
            contradicted = True
    if contradicted:
        return "corrected", agree_rank, lowest
    if incomplete:
        return "partial", agree_rank, lowest
    return "agree", agree_rank, lowest


class ValidationSession:
    """Rows, verdicts and persistence of one validator's session."""

    def __init__(self, run_root: Path, validator: str, base: pl.DataFrame, meta: dict[str, Any]) -> None:
        """Initialize a session.

        Args:
            run_root: Run output root.
            validator: Validator name.
            base: Per-track key, stratum, sample_order and model columns.
            meta: Session metadata.
        """
        self.run_root = Path(run_root)
        self.validator = validator
        self.base = base.sort("sample_order")
        self.meta = meta
        # Change verdicts only through _store() / _remove(), which keep the indexes in sync
        self.verdicts: dict[TrackKey, Verdict] = {}
        self._pending_seconds: dict[TrackKey, float] = {}
        self._history: list[dict[str, Any]] = []
        self._base_rows: dict[TrackKey, dict[str, Any]] = {
            (row["device_id"], row["session_id"], row["track_id"]): row for row in self.base.iter_rows(named=True)
        }
        self._keys: list[TrackKey] = list(self._base_rows)
        self._model: dict[TrackKey, Lineage] = {
            key: {r: row.get(f"pred_{r}") for r in VALIDATION_RANKS} for key, row in self._base_rows.items()
        }
        # Indexes: derived verdicts, tracks linking to a key, recent lineages
        self._derived: dict[TrackKey, tuple[str | None, str | None, str | None]] = {}
        self._successors: dict[TrackKey, set[str]] = {}
        self._recent: list[Lineage] | None = []
        # Set by every change of verdicts, notes, effort, free-text taxa or UI state, cleared by save()
        self.unsaved = False

    @staticmethod
    def session_dir(run_root: Path) -> Path:
        """Return the directory of the session files of a run."""
        return OutputLayout(Path(run_root)).validation_dir

    @property
    def validation_session_id(self) -> str:
        """Start time identifying the session."""
        return str(self.meta["validation_session_id"])

    @property
    def stem(self) -> str:
        """Filename stem of the session files: '<validator>_<started>'."""
        return f"{safe_validator_name(self.validator)}_{self.validation_session_id}"

    @property
    def csv_path(self) -> Path:
        """Session CSV (one row per sampled track)."""
        return self.session_dir(self.run_root) / f"{self.stem}{_SESSION_CSV_SUFFIX}"

    @property
    def json_path(self) -> Path:
        """Session JSON (sampling, provenance, UI state)."""
        return self.session_dir(self.run_root) / f"{self.stem}{_SESSION_JSON_SUFFIX}"

    @property
    def log_path(self) -> Path:
        """Log file written while the session is open."""
        return self.session_dir(self.run_root) / f"{self.stem}{_SESSION_LOG_SUFFIX}"

    @property
    def history_path(self) -> Path:
        """Append-only log of verdict changes (JSON lines)."""
        return self.session_dir(self.run_root) / f"{self.stem}{_HISTORY_SUFFIX}"

    @property
    def stats_dir(self) -> Path:
        """Directory of the exported statistics."""
        return self.session_dir(self.run_root) / f"stats_{self.stem}"

    @property
    def blind(self) -> bool:
        """Whether the model prediction is hidden until a determination is entered."""
        return bool(self.meta["blind"])

    @property
    def model_rank(self) -> str:
        """Deepest rank the run was classified to."""
        return str(self.meta["model_rank"])

    @staticmethod
    def list_sessions(run_root: Path) -> list[dict[str, Any]]:
        """Return summaries of the stored sessions of a run, last updated first."""
        result = []
        for json_path in sorted(ValidationSession.session_dir(run_root).glob(f"*{_SESSION_JSON_SUFFIX}")):
            try:
                meta = _read_meta(json_path)
            except ValueError as e:
                logger.warning("%s The session is not listed.", e)
                continue
            stem = json_path.name.removesuffix(_SESSION_JSON_SUFFIX)
            result.append({
                **{key: meta[key] for key in ("validator", "validation_session_id", "started", "updated", "blind",
                                              "final_csv", "sampling", "n_sampled", "n_validated")},
                "csv_path": json_path.with_name(f"{stem}{_SESSION_CSV_SUFFIX}"),
            })
        result.sort(key=lambda s: s["updated"], reverse=True)
        return result

    @classmethod
    def create(
        cls,
        run: ValidationRun,
        validator: str,
        sampled: pl.DataFrame,
        sampling: SamplingConfig,
        taxonomy: TaxonomyIndex,
        blind: bool = False,
    ) -> ValidationSession:
        """Create a new session from sampled tracks.

        Args:
            run: Opened run.
            validator: Validator name.
            sampled: Output of sample_strata().
            sampling: Sampling parameters.
            taxonomy: Maps model species to canonical names.
            blind: Hide the model prediction until a determination is entered.

        Returns:
            New unsaved session.
        """
        pred_cols = []
        for rank in _RANKS_CSV:
            col = f"bioclip_{rank}"
            if col not in sampled.columns:
                pred_cols.append(pl.lit(None, dtype=pl.String).alias(f"pred_{rank}"))
            elif rank == "species":
                pred_cols.append(taxonomy.canonical_species_expr(sampled.columns).alias("pred_species"))
            else:
                pred_cols.append(pl.col(col).alias(f"pred_{rank}"))
        base = sampled.select(
            *TRACK_KEY_COLUMNS,
            *[pl.col(c) if c in sampled.columns else pl.lit(None, dtype=pl.String).alias(c) for c in STRATUM_COLUMNS],
            pl.col("stratum_n_population").cast(pl.Int64),
            pl.col("sample_order").cast(pl.Int64),
            pl.lit(run.model_rank).alias("model_rank"),
            pl.col("pred"),
            pl.col("pred_prob_weighted").cast(pl.Float64),
            *pred_cols,
        )
        stratum_pop = (
            sampled.group_by("stratum_id").agg(pl.col("stratum_n_population").first())
            .sort("stratum_id").to_dicts()
        )
        # Selected tracks and the strata the minimum excluded from the sample (see sample_strata())
        selected = sampling.select_tracks(run.tracks)
        summary = stratum_summary(assign_strata(selected, sampling.stratum_rank, sampling.prob_edges,
                                                by_setting=sampling.stratum_setting))
        excluded = summary.filter(pl.col("stratum_id").is_in(small_strata(summary, sampling.min_per_stratum)))
        meta = {
            # Session
            "validator": validator,
            "validation_session_id": new_validation_session_id(run.root, validator),
            "started": now_iso(),
            "updated": now_iso(),
            "app_version": app_version(),
            "blind": bool(blind),
            # Source: the run, its final CSV at session start (see source_csv_changed()) and the taxonomy list
            "run_name": run.root.name,
            "run_root": str(run.root),
            "final_csv": dict(run.final_provenance),
            "taxonomy": dict(taxonomy.provenance),
            "model_rank": run.model_rank,
            # Track counts from the classified tracks down to the sample, each after the step before it
            "n_tracks_classified": run.n_tracks_classified,
            "run_filters": run.run_filters(),  # None = no config snapshot
            "n_population_total": run.tracks.height,
            "sampling": sampling.to_dict(),
            "n_population_selected": selected.height,
            # Population size per stratum: below 'min_per_stratum' (not sampled), sampled (the weighted population)
            "strata_excluded": dict(sorted(excluded.select("stratum_id", "n_population").iter_rows())),
            "strata_population": {d["stratum_id"]: d["stratum_n_population"] for d in stratum_pop},
            "n_sampled": base.height,
            "n_validated": 0,
            # State
            "custom_taxa": [],
            "ui": {},
            **copy.deepcopy(_META_DEFAULTS),
        }
        session = cls(run.root, validator, base, meta)
        session.unsaved = True
        return session

    @classmethod
    def load(cls, csv_path: Path) -> ValidationSession:
        """Load a saved session from its CSV and JSON.

        Raises:
            FileNotFoundError: If the CSV does not exist.
            ValueError: If the JSON is missing, unreadable or incomplete, or the CSV lacks columns.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        stem = csv_path.name.removesuffix(_SESSION_CSV_SUFFIX)
        json_path = csv_path.with_name(f"{stem}{_SESSION_JSON_SUFFIX}")
        meta = _read_meta(json_path)

        df = pl.read_csv(csv_path, infer_schema_length=0)
        base_cols = [*TRACK_KEY_COLUMNS, *STRATUM_COLUMNS, "stratum_n_population", "sample_order", *_MODEL_COLUMNS]
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Session CSV '{csv_path.name}' lacks columns: {missing}")
        if unknown := [c for c in df.columns if c not in base_cols and c not in _VERDICT_COLUMNS]:
            raise ValueError(_newer_version_message(csv_path.name, "columns", unknown, meta))
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias(c) for c in _VERDICT_COLUMNS if c not in df.columns)
        df = df.with_columns(
            pl.col("stratum_n_population").cast(pl.Int64, strict=False),
            pl.col("sample_order").cast(pl.Int64, strict=False),
            pl.col("pred_prob_weighted").cast(pl.Float64, strict=False),
            pl.col("seconds_spent").cast(pl.Float64, strict=False),
        )
        base = df.select(base_cols)
        validator = str(meta["validator"])
        run_root = csv_path.parent.parent
        session = cls(run_root, validator, base, meta)
        for row in df.select([*TRACK_KEY_COLUMNS, *_VERDICT_COLUMNS]).iter_rows(named=True):
            validated = str(row.get("validated") or "").lower() == "true"
            pending_content = ((row.get("notes") or "").strip() or row.get("image_quality") == "low"
                               or row.get("multiple_individuals") or row.get("same_individual_as")
                               or row.get("val_qualifier") or row.get("val_sex"))
            key: TrackKey = tuple(row[c] for c in TRACK_KEY_COLUMNS)  # type: ignore[assignment]
            if not validated and not pending_content:
                if row.get("seconds_spent"):  # effort on a track without a verdict yet
                    session._pending_seconds[key] = float(row["seconds_spent"])
                continue
            session._store(key, Verdict(
                validated=validated,
                validated_at=row.get("validated_at") or None,
                false_positive=row.get("verdict") == "false_positive",
                image_quality=row.get("image_quality") or "ok",
                image_quality_reason=row.get("image_quality_reason") or None,
                multiple_individuals=row.get("multiple_individuals") or None,
                same_individual_as=row.get("same_individual_as") or None,
                val={r: (row.get(f"val_{r}") or None) for r in VALIDATION_RANKS},
                val_source=row.get("val_source") or None,
                qualifier=row.get("val_qualifier") or None,
                sex=row.get("val_sex") or None,
                notes=row.get("notes") or "",
                seconds=float(row.get("seconds_spent") or 0.0),
            ))
        session._recent = None  # rebuilt on first use
        logger.info("Loaded session '%s': %d/%d tracks validated", csv_path.name, session.progress()[0], base.height)
        return session

    def keys(self) -> list[TrackKey]:
        """Return the track keys in validation queue order."""
        return list(self._keys)

    def model_lineage(self, key: TrackKey) -> Lineage:
        """Return the model lineage (canonical names) of a sampled track."""
        return dict(self._model.get(key, dict.fromkeys(VALIDATION_RANKS)))

    def base_row(self, key: TrackKey) -> dict[str, Any] | None:
        """Return the sampling and model columns of a track (shared dict, do not mutate)."""
        return self._base_rows.get(key)

    def verdict(self, key: TrackKey) -> Verdict | None:
        """Return the stored verdict of a track (also a pending one), or None."""
        return self.verdicts.get(key)

    def derived(self, key: TrackKey) -> tuple[str | None, str | None, str | None]:
        """Return the cached '(verdict_type, agree_rank, val_lowest_rank)' of a track."""
        result = self._derived.get(key)
        if result is None:
            verdict = self.verdicts.get(key)
            if verdict is None:
                return None, None, None
            result = derive_verdict(self.model_lineage(key), verdict, self.model_rank)
            self._derived[key] = result
        return result

    def progress(self) -> tuple[int, int]:
        """Return the '(validated, total)' track counts."""
        return sum(1 for v in self.verdicts.values() if v.validated), len(self._keys)

    def validated_keys(self) -> list[TrackKey]:
        """Return the keys of the validated tracks in queue order."""
        return [k for k in self._keys if (v := self.verdicts.get(k)) is not None and v.validated]

    def predecessors_of(self, key: TrackKey) -> list[str]:
        """Return the earlier track IDs this track links to as the same individual."""
        entry = self.verdicts.get(key)
        return parse_links(entry.same_individual_as) if entry is not None and not entry.false_positive else []

    def successors_of(self, key: TrackKey) -> list[str]:
        """Return the track IDs that link to this track as the same individual."""
        return sorted(self._successors.get(key, ()), key=track_id_order)

    def relink_around(self, key: TrackKey) -> list[tuple[TrackKey, list[TrackKey]]]:
        """Move the links to the false positive 'key' onto the tracks it linked to.

        A false positive leaves its individual (see predecessors_of()) and no track may link to one. The tracks
        linking to 'key' link to its earlier tracks instead, whatever the time gap: the user linked them as one
        individual. If 'key' linked to no track, their link to it is dropped.

        Returns:
            '(later track, earlier tracks it now links to)' per changed track, an empty list if its link was dropped.
        """
        entry = self.verdicts.get(key)
        if entry is None or not entry.false_positive:
            return []
        device, session_id = key[0], key[1]
        earlier = [(device, session_id, tid) for tid in parse_links(entry.same_individual_as)]
        earlier = [e for e in earlier if not ((other := self.verdicts.get(e)) is not None and other.false_positive)]
        relinked = []
        for tid in self.successors_of(key):
            later = (device, session_id, tid)
            existing = self.verdicts.get(later)
            if existing is None:
                continue
            value = with_link(existing.same_individual_as, key[2], False)
            for e in earlier:
                value = with_link(value, e[2], True)
            verdict = existing.copy()
            verdict.same_individual_as = value
            if verdict.validated:
                self.set_verdict(later, verdict)
            else:
                self.set_pending(later, verdict)
            relinked.append((later, list(earlier)))
        return relinked

    def group_of(self, key: TrackKey) -> list[TrackKey]:
        """Return all tracks of the individual 'key' belongs to, including 'key' and unsampled tracks."""
        device, session_id = key[0], key[1]
        members, todo = {key[2]}, [key[2]]
        while todo:
            current = todo.pop()
            neighbors = set(self._successors.get((device, session_id, current), ()))
            neighbors.update(self.predecessors_of((device, session_id, current)))
            for other in neighbors:
                if other not in members:
                    members.add(other)
                    todo.append(other)
        return [(device, session_id, tid) for tid in sorted(members, key=track_id_order)]

    def recent_lineages(self, limit: int = 8) -> list[Lineage]:
        """Return the most recent distinct human lineages, newest first."""
        if self._recent is None:
            recent: list[Lineage] = []
            ordered = sorted(self.verdicts.values(), key=lambda v: v.validated_at or "", reverse=True)
            for verdict in ordered:
                if not verdict.validated or verdict.false_positive or not verdict.determinate_ranks():
                    continue
                lineage = dict(verdict.val)
                if lineage not in recent:
                    recent.append(lineage)
                if len(recent) >= _RECENT_LINEAGES_KEPT:
                    break
            self._recent = recent
        return [dict(lineage) for lineage in self._recent[:limit]]

    def set_verdict(self, key: TrackKey, verdict: Verdict) -> None:
        """Store a verdict and mark the track validated. 'seconds' is kept by the session."""
        previous = self.verdicts.get(key)
        verdict.validated = True
        verdict.validated_at = verdict.validated_at or now_iso()
        verdict.seconds = (previous.seconds if previous else 0.0) + self._pending_seconds.pop(key, 0.0)
        self._store(key, verdict)
        self._log("verdict", key, verdict)
        self.unsaved = True

    def set_note(self, key: TrackKey, notes: str) -> None:
        """Store the note of a track (not a history event, a note alone is no verdict)."""
        entry = self.verdicts.get(key)
        if entry is not None and entry.validated:
            if entry.notes != notes:
                entry.notes = notes
                self.unsaved = True
            return
        pending = entry.copy() if entry is not None else Verdict(validated=False)
        pending.notes = notes
        self.set_pending(key, pending)

    def set_pending(self, key: TrackKey, verdict: Verdict) -> bool:
        """Store notes, flags, qualifier and sex of a track that is not validated yet.

        The track stays to do, an entry without content is dropped. Ignored for validated tracks.

        Returns:
            Whether the stored entry changed.
        """
        entry = self.verdicts.get(key)
        if entry is not None and entry.validated:
            return False
        has_content = ((verdict.notes or "").strip() or verdict.image_quality == "low" or verdict.multiple_individuals
                       or verdict.same_individual_as or verdict.qualifier or verdict.sex)
        if not has_content:
            if entry is not None:
                self._remove(key)
                self.unsaved = True
                return True
            return False
        pending = verdict.copy()
        pending.validated = False
        pending.validated_at = None
        pending.seconds = entry.seconds if entry is not None else 0.0
        if entry is not None and entry == pending:
            return False
        self._store(key, pending)
        self.unsaved = True
        return True

    def reset(self, key: TrackKey) -> None:
        """Remove the human verdict of a track."""
        entry = self._remove(key)
        if entry is not None:
            if entry.validated:
                self._log("reset", key, None)
            # Keep the effort already spent
            if entry.seconds:
                self._pending_seconds[key] = self._pending_seconds.get(key, 0.0) + entry.seconds
            self.unsaved = True

    def add_seconds(self, key: TrackKey, seconds: float) -> None:
        """Add validation effort to a track (capped per interval to limit idle time)."""
        seconds = max(0.0, min(float(seconds), float(VALIDATION_EFFORT_IDLE_CAP_S)))
        if seconds <= 0:
            return
        entry = self.verdicts.get(key)
        if entry is not None:
            entry.seconds += seconds
        else:
            self._pending_seconds[key] = self._pending_seconds.get(key, 0.0) + seconds
        self.unsaved = True

    def set_ui(self, name: str, value: Any) -> None:
        """Store a UI state value in the session JSON (e.g. the last shown track)."""
        if self.meta["ui"].get(name) != value:
            self.meta["ui"][name] = value
            self.unsaved = True

    def add_custom_taxon(self, rank: str, lineage: Lineage) -> None:
        """Remember a free-text lineage."""
        entry = {"rank": rank, **{r: lineage.get(r) for r in ranks_up_to(rank)}}
        if entry not in self.meta["custom_taxa"]:
            self.meta["custom_taxa"].append(entry)
            self.unsaved = True

    def seconds_spent(self, key: TrackKey) -> float | None:
        """Return the validation effort of a track (also without a verdict), None if none was recorded."""
        entry = self.verdicts.get(key)
        if entry is not None:
            return round(entry.seconds + self._pending_seconds.get(key, 0.0), 1)
        return round(self._pending_seconds[key], 1) if key in self._pending_seconds else None

    def to_frame(self) -> pl.DataFrame:
        """Return the full session table (base columns + verdict columns) in queue order."""
        rows = [
            verdict_columns(entry := self.verdicts.get(key), self.derived(key),
                            self.validator if entry else None, self.seconds_spent(key))
            for key in self._keys
        ]
        return self.base.hstack(verdict_frame(rows))

    def save(self) -> None:
        """Write CSV + JSON atomically (temp file, then replace)."""
        self.session_dir(self.run_root).mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        self.meta["updated"] = now_iso()
        self.meta["n_sampled"] = frame.height
        self.meta["n_validated"] = self.progress()[0]
        self.meta["validator"] = self.validator

        tmp_csv = self.csv_path.with_suffix(".csv.tmp")
        frame.write_csv(tmp_csv)
        tmp_csv.replace(self.csv_path)
        tmp_json = self.json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_json.replace(self.json_path)
        if self._history:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(event, ensure_ascii=False) + "\n" for event in self._history)
            self._history = []
        self.unsaved = False
        logger.debug("Session saved: %s (%d/%d validated)", self.csv_path.name, self.progress()[0], frame.height)

    def export_validated(self, run: ValidationRun, taxonomy: TaxonomyIndex | None = None) -> Path:
        """Write the final CSV joined with the verdicts ('<final_stem>_validated_<stem>.csv').

        See validated_frame() for the added columns.

        Returns:
            Path of the written CSV.
        """
        exported = validated_frame(run, self.to_frame(), taxonomy)
        out_path = run.final_path.with_name(f"{run.final_path.stem}{EXPORT_SUFFIX}_{self.stem}.csv")
        exported.write_csv(out_path)
        logger.info("Validated export written: %s", out_path)
        return out_path

    def _store(self, key: TrackKey, verdict: Verdict) -> None:
        """Store a verdict and update the indexes."""
        self._remove(key)
        self.verdicts[key] = verdict
        if not verdict.false_positive:
            for other in parse_links(verdict.same_individual_as):
                self._successors.setdefault((key[0], key[1], other), set()).add(key[2])
        if (verdict.validated and self._recent is not None and not verdict.false_positive
                and verdict.determinate_ranks()):
            lineage = dict(verdict.val)
            self._recent = [lineage, *[lin for lin in self._recent if lin != lineage]][:_RECENT_LINEAGES_KEPT]

    def _remove(self, key: TrackKey) -> Verdict | None:
        """Remove a verdict from the store and the indexes, then return it."""
        entry = self.verdicts.pop(key, None)
        self._derived.pop(key, None)
        if entry is not None:
            for other in parse_links(entry.same_individual_as):
                successors = self._successors.get((key[0], key[1], other))
                if successors is not None:
                    successors.discard(key[2])
                    if not successors:
                        del self._successors[(key[0], key[1], other)]
            if entry.validated:
                self._recent = None  # rebuilt on next use
        return entry

    def _log(self, action: str, key: TrackKey, verdict: Verdict | None) -> None:
        """Queue a history event (written to the history file on the next save)."""
        event: dict[str, Any] = {"time": now_iso(), "validator": self.validator, "action": action,
                                 "device_id": key[0], "session_id": key[1], "track_id": key[2]}
        if verdict is not None:
            verdict_type, agree_rank, lowest = derive_verdict(self.model_lineage(key), verdict, self.model_rank)
            event.update({
                "verdict": verdict_type, "agree_rank": agree_rank, "val_lowest_rank": lowest,
                "val": {r: verdict.val.get(r) for r in VALIDATION_RANKS},
                "image_quality": verdict.image_quality, "image_quality_reason": verdict.image_quality_reason,
                "multiple_individuals": verdict.multiple_individuals,
                "same_individual_as": verdict.same_individual_as,
                "qualifier": verdict.qualifier, "sex": verdict.sex,
                "notes": verdict.notes or None,
            })
        self._history.append(event)


def verdict_columns(
    entry: Verdict | None,
    derived: tuple[str | None, str | None, str | None],
    validator: str | None,
    seconds: float | None,
) -> dict[str, Any]:
    """Return the verdict columns of one track, as written to the session CSV.

    Args:
        entry: Stored verdict (also a pending one), or None.
        derived: '(verdict_type, agree_rank, val_lowest_rank)' of the verdict (see derive_verdict()).
        validator: Validator named in the row (None without an entry).
        seconds: Validation effort of the track, or None.

    Returns:
        One value per verdict column.
    """
    verdict = entry if entry is not None and entry.validated else None
    verdict_type, agree_rank, lowest = derived
    row: dict[str, Any] = {
        "validated": verdict is not None,
        "validated_at": verdict.validated_at if verdict else None,
        "validator": validator,
        "verdict": verdict_type,
        # Flags are kept for pending entries as well
        "image_quality": entry.image_quality if entry else None,
        "image_quality_reason": entry.image_quality_reason if entry and entry.image_quality == "low" else None,
        "multiple_individuals": (
            entry.multiple_individuals if entry and not (verdict and verdict.false_positive) else None
        ),
        "same_individual_as": entry.same_individual_as if entry and not (verdict and verdict.false_positive) else None,
        "val_lowest_rank": lowest,
        # Kept for pending entries, the rank rule needs a determination
        "val_qualifier": (
            (allowed_qualifier(entry.qualifier, lowest) if verdict else entry.qualifier) if entry else None
        ),
        "val_sex": entry.sex if entry and not (verdict and verdict.false_positive) else None,
        "val_source": verdict.val_source if verdict and not verdict.false_positive else None,
        "agree_rank": agree_rank,
        "notes": (entry.notes or None) if entry else None,
        "seconds_spent": seconds,
    }
    for rank in VALIDATION_RANKS:
        row[f"val_{rank}"] = (verdict.val.get(rank) or None) if verdict and not verdict.false_positive else None
    return row


def verdict_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Return verdict_columns() rows as a dataframe with the session CSV's verdict columns and types."""
    schema = dict.fromkeys(_VERDICT_COLUMNS, pl.String) | {"validated": pl.Boolean, "seconds_spent": pl.Float64}
    return pl.DataFrame(rows, schema=schema).select(_VERDICT_COLUMNS)


def validated_frame(
    run: ValidationRun,
    frame: pl.DataFrame,
    taxonomy: TaxonomyIndex | None = None,
    extra_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Return every track of the run's final CSV joined with the verdicts of a session table.

    Adds 'combined_<rank>' (validator's names for validated tracks, indet. empty and an aggregate
    species with ' agg.', else the model's, its species as GBIF canonical name like the comparison),
    'combined_source' and 'individual_id' / 'individual_n_tracks'. The final CSV's 'gbif_species'
    and 'gbif_species_key' come along; a validator's species gets its key by joining
    'combined_species' with 'gbif_species' of the species mapping.

    Args:
        run: Opened run.
        frame: Rows with the track key, 'stratum_id', 'stratum_n_population' and the verdict columns.
        taxonomy: Maps model species to canonical names.
        extra_columns: Further columns of 'frame' to append, in this order.

    Returns:
        One row per track, in the order of the final CSV.
    """
    join_cols = [*TRACK_KEY_COLUMNS, "stratum_id", "stratum_n_population", *_VERDICT_COLUMNS, *extra_columns]
    exported = run.tracks.join(frame.select(join_cols), on=TRACK_KEY_COLUMNS, how="left", maintain_order="left")
    exported = exported.with_columns(pl.col("validated").fill_null(False))

    # The model's species in the form the validation compares on
    model_species = (taxonomy or TaxonomyIndex()).canonical_species_expr(exported.columns)
    validated = pl.col("validated")
    false_positive = pl.col("verdict") == "false_positive"
    combined_cols = []
    for rank in _RANKS_CSV:
        model_col = f"bioclip_{rank}"
        if model_col not in exported.columns:
            model_expr = pl.lit(None, dtype=pl.String)
        elif rank == "species":
            model_expr = model_species
        else:
            model_expr = pl.col(model_col)
        val_col = pl.col(f"val_{rank}")
        human = pl.when(val_col != INDET_LABEL).then(val_col)
        if rank == "species":
            # An aggregate is a group of species, not the species it is named after
            aggregate = pl.col("val_qualifier").fill_null("").str.split(" ").list.contains(ID_QUALIFIER_AGG)
            human = (pl.when(aggregate & (val_col != INDET_LABEL)).then(val_col + f" {ID_QUALIFIER_AGG}")
                     .otherwise(human))
        # One source per track: an indet. rank stays empty, not the model's name
        combined_cols.append(
            pl.when(false_positive).then(pl.lit(None, dtype=pl.String))
            .when(validated).then(human)
            .otherwise(model_expr)
            .alias(f"combined_{rank}")
        )
    exported = exported.with_columns(combined_cols).with_columns(
        pl.when(false_positive).then(pl.lit("false_positive"))
        .when(validated).then(pl.lit("human"))
        .otherwise(pl.lit("model"))
        .alias("combined_source")
    )
    exported = exported.join(individual_groups(exported), on=TRACK_KEY_COLUMNS, how="left", maintain_order="left")
    # The extra columns describe the verdict: they stay at the end, after the individual
    return exported.select([*(c for c in exported.columns if c not in extra_columns), *extra_columns])


def session_files(csv_path: Path) -> list[Path]:
    """Return the existing files and directories of a session (CSV, JSON, log, history, statistics).

    Args:
        csv_path: Session CSV.

    Raises:
        ValueError: If 'csv_path' is not a session CSV.

    Returns:
        The paths that exist, in deletion order.
    """
    csv_path = Path(csv_path)
    if not csv_path.name.endswith(_SESSION_CSV_SUFFIX):
        raise ValueError(f"'{csv_path.name}' is not a session CSV ('*{_SESSION_CSV_SUFFIX}')")
    stem = csv_path.name.removesuffix(_SESSION_CSV_SUFFIX)
    directory = csv_path.parent
    candidates = [csv_path, directory / f"{stem}{_SESSION_JSON_SUFFIX}", directory / f"{stem}{_SESSION_LOG_SUFFIX}",
                  directory / f"{stem}{_HISTORY_SUFFIX}", directory / f"stats_{stem}"]
    return [path for path in candidates if path.exists()]


def delete_session(csv_path: Path) -> list[Path]:
    """Delete a session with its files (see session_files()), the validated export is kept.

    Raises:
        OSError: If a file could not be removed.
        ValueError: If 'csv_path' is not a session CSV.

    Returns:
        The paths that were removed.
    """
    removed = []
    for path in session_files(csv_path):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    logger.info("Session deleted: %s (%d paths)", Path(csv_path).name, len(removed))
    return removed


def source_csv_changed(session: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return whether the run's final CSV has other content (sha256) than the one at session start.

    A copied or renamed run is the same run: only the content counts.

    Args:
        session: Session metadata or a list_sessions() entry.
        current: 'final_provenance' of the opened run.

    Returns:
        True if it differs. False if nothing was recorded.
    """
    recorded = (session.get("final_csv") or {}).get("sha256")
    return recorded is not None and recorded != current.get("sha256")


def open_session_log(session: ValidationSession) -> logging.Handler:
    """Append the package log to the session's log file.

    Args:
        session: Open session.

    Returns:
        The added handler (close it with close_session_log()).
    """
    session.log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(session.log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger("insectdetect_post").addHandler(handler)
    return handler


def close_session_log(handler: logging.Handler | None) -> None:
    """Detach and close a handler from open_session_log()."""
    if handler is None:
        return
    logging.getLogger("insectdetect_post").removeHandler(handler)
    handler.close()


def sort_tracks_chronologically(keys: Sequence[TrackKey],
                                starts: Mapping[TrackKey, Any] | None = None) -> list[TrackKey]:
    """Return tracks in time order: by start time, then numeric track ID (missing start times first).

    The one order of an individual's tracks: its name (anchor_of()), the panel's individual line and
    the chain of a batch link.

    Args:
        keys: Track keys.
        starts: Start times per track key (ISO strings, as read from the final CSV), if known.

    Returns:
        The keys in time order.
    """
    starts = starts or {}
    return sorted(keys, key=lambda k: (str(starts.get(k) or ""), track_id_order(k[2])))


def anchor_of(members: Sequence[TrackKey], starts: Mapping[TrackKey, Any] | None = None) -> str:
    """Return the track ID naming an individual: its earliest track, else the lowest numeric ID.

    Args:
        members: Track keys of one individual.
        starts: Start times per track key, if known.

    Returns:
        The 'individual_id'.
    """
    return sort_tracks_chronologically(members, starts)[0][2]


def individual_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Group tracks into individuals via their 'same_individual_as' links.

    Args:
        frame: Rows with the track key, 'same_individual_as' and optionally 'start_time'.

    Returns:
        'device_id, session_id, track_id, individual_id, individual_n_tracks' per row
        (see anchor_of()).
    """
    keys: list[TrackKey] = [tuple(row) for row in frame.select(TRACK_KEY_COLUMNS).iter_rows()]  # type: ignore[misc]
    parent: dict[TrackKey, TrackKey] = {k: k for k in keys}

    def find(k: TrackKey) -> TrackKey:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    if "same_individual_as" in frame.columns:
        for key, value in zip(keys, frame["same_individual_as"].to_list(), strict=True):
            for other_id in parse_links(value):
                other: TrackKey = (key[0], key[1], other_id)
                if other in parent:
                    parent[find(key)] = find(other)
    starts = dict(zip(keys, frame["start_time"].to_list(), strict=True)) if "start_time" in frame.columns else {}
    groups: dict[TrackKey, list[TrackKey]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    individual: dict[TrackKey, tuple[str, int]] = {}
    for members in groups.values():
        first = anchor_of(members, starts)
        for k in members:
            individual[k] = (first, len(members))
    return pl.DataFrame(
        {
            "device_id": [k[0] for k in keys], "session_id": [k[1] for k in keys], "track_id": [k[2] for k in keys],
            "individual_id": [individual[k][0] for k in keys],
            "individual_n_tracks": [individual[k][1] for k in keys],
        },
        schema={"device_id": pl.String, "session_id": pl.String, "track_id": pl.String, "individual_id": pl.String,
                "individual_n_tracks": pl.Int64},
    )


def now_iso() -> str:
    """Return the current local time as an ISO string (seconds)."""
    return datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005


def app_version() -> str:
    """Return the installed package version, 'unknown' if not installed."""
    try:
        return version("insect-detect-post")
    except PackageNotFoundError:
        return "unknown"


def _read_meta(json_path: Path) -> dict[str, Any]:
    """Read a validation session JSON.

    Keys and sampling fields added after the session was started get their defaults.

    Raises:
        ValueError: If it is missing or unreadable, lacks required keys (see _META_KEYS) or has sampling
            fields unknown to this version.
    """
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Session metadata '{json_path.name}' is missing or unreadable.") from e
    missing = sorted(_META_KEYS - meta.keys()) if isinstance(meta, dict) else sorted(_META_KEYS)
    if not missing and not isinstance(meta["sampling"], dict):
        missing.append("sampling fields")
    if missing:
        raise ValueError(f"Session metadata '{json_path.name}' is incomplete (missing: {', '.join(missing)}).")
    if unknown := sorted(set(meta["sampling"]) - set(SamplingConfig.__dataclass_fields__)):
        raise ValueError(_newer_version_message(json_path.name, "sampling fields", unknown, meta))
    for key, default in _META_DEFAULTS.items():
        meta.setdefault(key, copy.deepcopy(default))
    meta["sampling"] = SamplingConfig.from_dict(meta["sampling"]).to_dict()
    return meta


def _newer_version_message(file_name: str, kind: str, unknown: list[str], meta: dict[str, Any]) -> str:
    """Return the error message for a session file with fields of a newer version."""
    return (f"Session file '{file_name}' has {kind} unknown to this version ({', '.join(unknown)}): it was "
            f"probably saved by a newer version (created with {meta.get('app_version', 'unknown')}).")

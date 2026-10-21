"""Several validation sessions of one run: one verdict per track and the multi-session validated export.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

The sessions are given in priority order, the first one wins. A track validated in several sessions gets
the verdict of the first of them ('priority'), or the deepest taxon all of them agree on
('lowest_common_ancestor'). Everything but the determination (flags, notes, links, effort) comes from the
first session either way. The export copies the session files into its folder first and reads them from
there, so the folder holds the exact input of its CSV.

Classes:
    SelectedVerdict: The verdict chosen for one track, with the sessions it comes from.
    VerdictSelection: One verdict per track of several sessions.
    MultiExport: Paths and counts of a multi-session export.

Functions:
    select_verdicts(): One verdict per track from sessions in priority order.
    export_validated_sessions(): Validated CSV of several sessions, with conflicts, provenance and session copies.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from insectdetect_post.constants import TRACK_KEY_COLUMNS
from insectdetect_post.validation_run import TrackKey, ValidationRun
from insectdetect_post.validation_session import (
    EXPORT_SUFFIX,
    ID_QUALIFIER_AGG,
    ID_QUALIFIER_INC,
    STAMP_FORMAT,
    ValidationSession,
    Verdict,
    allowed_qualifier,
    app_version,
    derive_verdict,
    has_qualifier,
    join_links,
    now_iso,
    parse_links,
    session_files,
    source_csv_changed,
    validated_frame,
    verdict_columns,
    verdict_frame,
    with_qualifier,
)
from insectdetect_post.validation_taxonomy import INDET_LABEL, VALIDATION_RANKS, TaxonomyIndex, file_provenance

# Create module-level logger
logger = logging.getLogger(__name__)

# Rules for a track validated in several sessions
CONFLICT_RULE_PRIORITY = "priority"
CONFLICT_RULE_COMMON_ANCESTOR = "lowest_common_ancestor"
CONFLICT_RULES = (CONFLICT_RULE_PRIORITY, CONFLICT_RULE_COMMON_ANCESTOR)

# How the verdicts of a track validated in several sessions differ, least severe first
DIFFERENT_DEPTH = "different_depth"  # same lineage, determined to different ranks
DIFFERENT_TAXA = "different_taxa"  # different names at a rank all of them determined
FALSE_POSITIVE_DISPUTED = "false_positive_disputed"  # 'not an arthropod' against an arthropod
VALIDATION_CONFLICTS = (DIFFERENT_DEPTH, DIFFERENT_TAXA, FALSE_POSITIVE_DISPUTED)

# Columns added to the validated CSV of several sessions
MULTI_COLUMNS = ["validation_session", "validation_n_sessions", "validation_conflict"]

# Separator of several validators or sessions in one value (as in 'same_individual_as')
_JOIN_SEPARATOR = ";"

# Files of the export folder
_MULTI_SUFFIX = "_multi"
_SESSIONS_DIR = "validation_sessions"
_CONFLICTS_FILE = "verdict_conflicts.csv"
_PROVENANCE_FILE = "provenance.json"


@dataclass
class SelectedVerdict:
    """The verdict chosen for one track.

    Attributes:
        verdict: Exported verdict (a copy; a consensus under 'lowest_common_ancestor').
        session: Session of everything but a consensus determination (the first with a verdict).
        contributors: Sessions the determination comes from, in priority order.
        candidates: '(session, verdict)' of every session that validated the track, in priority order.
        conflict: How the candidates differ (see VALIDATION_CONFLICTS), None if they agree.
    """
    verdict: Verdict
    session: ValidationSession
    contributors: list[ValidationSession]
    candidates: list[tuple[ValidationSession, Verdict]] = field(default_factory=list)
    conflict: str | None = None


@dataclass
class VerdictSelection:
    """One verdict per track of several sessions.

    Attributes:
        verdicts: Chosen verdict per track with a stored entry in any session (validated or pending).
        dropped_links: '(track, linked track ID, session of the link)' of the links removed because they
            pointed to a track chosen as a false positive.
    """
    verdicts: dict[TrackKey, SelectedVerdict]
    dropped_links: list[tuple[TrackKey, str, ValidationSession]] = field(default_factory=list)

    @property
    def n_links_dropped(self) -> int:
        """Return the number of removed links."""
        return len(self.dropped_links)


@dataclass
class MultiExport:
    """Result of export_validated_sessions().

    Attributes:
        folder: Export folder.
        csv_path: Validated CSV.
        counts: Track, conflict and link counts (as in the provenance file).
        taxonomy_differs: Whether the sessions were started with different taxonomy lists.
    """
    folder: Path
    csv_path: Path
    counts: dict[str, int]
    taxonomy_differs: bool


def select_verdicts(sessions: Sequence[ValidationSession], rule: str = CONFLICT_RULE_PRIORITY) -> VerdictSelection:
    """Return one verdict per track from sessions in priority order (the first session wins).

    A validated verdict wins over a pending entry (notes, flags or links without a verdict), which is taken
    from the first session that has one. Links to a track whose chosen verdict is a false positive are removed.

    Args:
        sessions: Sessions of one run, highest priority first.
        rule: CONFLICT_RULE_PRIORITY or CONFLICT_RULE_COMMON_ANCESTOR.

    Raises:
        ValueError: If 'rule' is unknown.

    Returns:
        The chosen verdicts.
    """
    if rule not in CONFLICT_RULES:
        raise ValueError(f"Unknown conflict rule '{rule}' (one of {', '.join(CONFLICT_RULES)})")
    chosen: dict[TrackKey, SelectedVerdict] = {}
    for key, entries in _entries_by_key(sessions).items():
        candidates = [(session, entry) for session, entry in entries if entry.validated]
        if not candidates:
            session, entry = entries[0]
            chosen[key] = SelectedVerdict(entry.copy(), session, [session])
            continue
        session, first = candidates[0]
        conflict = _conflict([entry for _, entry in candidates])
        if rule == CONFLICT_RULE_COMMON_ANCESTOR and len(candidates) > 1:
            verdict, contributors = _lowest_common_ancestor(candidates), [s for s, _ in candidates]
        else:
            verdict, contributors = first.copy(), [session]
        chosen[key] = SelectedVerdict(verdict, session, contributors, candidates, conflict)

    false_positives = {key for key, sel in chosen.items() if sel.verdict.validated and sel.verdict.false_positive}
    dropped = []
    for key, sel in chosen.items():
        links = parse_links(sel.verdict.same_individual_as)
        kept = [tid for tid in links if (key[0], key[1], tid) not in false_positives]
        if len(kept) < len(links):
            dropped.extend((key, tid, sel.session) for tid in links if tid not in kept)
            sel.verdict.same_individual_as = join_links(kept)
    return VerdictSelection(chosen, dropped)


def export_validated_sessions(
    run: ValidationRun,
    csv_paths: Sequence[Path],
    taxonomy: TaxonomyIndex | None = None,
    rule: str = CONFLICT_RULE_PRIORITY,
    now: datetime | None = None,
) -> MultiExport:
    """Write the validated CSV of several sessions into a new folder next to the run's final CSV.

    The folder '<final_stem>_validated_multi_<time>/' holds the CSV of the same name (the columns of the
    single-session export plus MULTI_COLUMNS), 'verdict_conflicts.csv' (every track validated in several
    sessions, one row per session), 'provenance.json' (rule, session order, checksums, counts and the
    removed links, see select_verdicts()) and 'validation_sessions/' with copies of the session CSVs,
    JSONs and histories, which are the files the CSV is computed from.

    Args:
        run: Opened run.
        csv_paths: Session CSVs, highest priority first.
        taxonomy: Maps model species to canonical names.
        rule: CONFLICT_RULE_PRIORITY or CONFLICT_RULE_COMMON_ANCESTOR.
        now: Export time, defaults to the current local time.

    Raises:
        ValueError: If no session or one twice is given, a session cannot be read, the run's final CSV
            changed after a session started, or 'rule' is unknown.
        OSError: If a file cannot be copied or written.

    Returns:
        Paths and counts of the export.
    """
    paths = [Path(p) for p in csv_paths]
    if not paths:
        raise ValueError("No session to export")
    if len({p.name for p in paths}) < len(paths):
        raise ValueError("A session is given twice")
    if rule not in CONFLICT_RULES:
        raise ValueError(f"Unknown conflict rule '{rule}' (one of {', '.join(CONFLICT_RULES)})")
    folder = _new_folder(run, now)
    try:
        sessions = _copy_and_load(paths, folder / _SESSIONS_DIR)
        if stale := [s.stem for s in sessions if source_csv_changed(s.meta, run.final_provenance)]:
            raise ValueError(f"'{run.final_path.name}' has changed since these sessions were started: "
                             f"{', '.join(stale)}")
        selection = select_verdicts(sessions, rule)
        frame, counts = _export_frame(sessions, selection)
        exported = validated_frame(run, frame, taxonomy, MULTI_COLUMNS).with_columns(
            pl.col("validation_n_sessions").fill_null(0))
        csv_path = folder / f"{folder.name}.csv"
        exported.write_csv(csv_path)
        _conflicts_frame(run, selection).write_csv(folder / _CONFLICTS_FILE)
        counts = {"n_tracks": run.tracks.height, **counts, "n_links_dropped": selection.n_links_dropped}
        taxonomy_differs = len({(s.meta["taxonomy"] or {}).get("sha256") for s in sessions}) > 1
        provenance = {
            "exported": now_iso(),
            "app_version": app_version(),
            "conflict_rule": rule,
            "run_name": run.root.name,
            "final_csv": dict(run.final_provenance),
            "taxonomy": dict(taxonomy.provenance) if taxonomy is not None else None,
            "files": {"validated": csv_path.name, "verdict_conflicts": _CONFLICTS_FILE, "sessions": _SESSIONS_DIR},
            "sessions": [_session_record(s, folder / _SESSIONS_DIR) for s in sessions],
            "taxonomy_differs": taxonomy_differs,
            "custom_taxa": _custom_taxa(sessions),
            "counts": counts,
            "links_dropped": [{**dict(zip(TRACK_KEY_COLUMNS, key, strict=True)), "linked_track_id": tid,
                               "validation_session": session.stem} for key, tid, session in selection.dropped_links],
        }
        (folder / _PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    except (OSError, ValueError):
        shutil.rmtree(folder, ignore_errors=True)  # no half-written export
        raise
    logger.info("Validated export of %d sessions written: %s (%s)", len(sessions), folder, counts)
    return MultiExport(folder, csv_path, counts, taxonomy_differs)


def _entries_by_key(sessions: Sequence[ValidationSession]) -> dict[TrackKey, list[tuple[ValidationSession, Verdict]]]:
    """Return the stored entries (validated or pending) of every track, in session priority order."""
    entries: dict[TrackKey, list[tuple[ValidationSession, Verdict]]] = {}
    for session in sessions:
        queue = session.keys()
        for key in queue:
            entry = session.verdict(key)
            if entry is not None:
                entries.setdefault(key, []).append((session, entry))
    return entries


def _conflict(verdicts: Sequence[Verdict]) -> str | None:
    """Return how validated verdicts of one track differ in their determination, None if they agree."""
    false_positive = {v.false_positive for v in verdicts}
    if len(false_positive) > 1:
        return FALSE_POSITIVE_DISPUTED
    if True in false_positive:
        return None
    names = [_determined_names(v) for v in verdicts]
    for rank in VALIDATION_RANKS:
        if len({n[rank] for n in names if n[rank]}) > 1:
            return DIFFERENT_TAXA
    return DIFFERENT_DEPTH if len({tuple(n.values()) for n in names}) > 1 else None


def _lowest_common_ancestor(candidates: Sequence[tuple[ValidationSession, Verdict]]) -> Verdict:
    """Return the deepest taxon all candidates agree on, with everything else of the first one.

    From class downwards a rank is kept while every verdict has the same entry. From the first rank where
    they differ, every rank is 'indet.'. A false positive against an arthropod gives 'indet.' at every rank.
    'agg.' is kept if one verdict names the kept species as an aggregate, 'inc.' if one marks the deepest
    kept rank as uncertain, the sex if all recorded values agree.
    """
    verdicts = [verdict for _, verdict in candidates]
    result = verdicts[0].copy()
    if all(v.false_positive for v in verdicts):
        return result
    arthropods = [v for v in verdicts if not v.false_positive]
    result.false_positive = False
    result.val = {}
    cut = len(arthropods) < len(verdicts)  # disputed: nothing is determined
    for rank in VALIDATION_RANKS:
        entries = {v.val.get(rank) or None for v in arthropods}
        cut = cut or len(entries) > 1
        result.val[rank] = INDET_LABEL if cut else entries.pop()
    lowest = _lowest(result)
    qualifier = None
    if lowest == "species" and any(has_qualifier(v.qualifier, ID_QUALIFIER_AGG) for v in arthropods):
        qualifier = with_qualifier(qualifier, ID_QUALIFIER_AGG, True)
    if lowest is not None and any(has_qualifier(v.qualifier, ID_QUALIFIER_INC) and _lowest(v) == lowest
                                  for v in arthropods):
        qualifier = with_qualifier(qualifier, ID_QUALIFIER_INC, True)
    result.qualifier = qualifier
    sexes = {v.sex for v in arthropods if v.sex}
    result.sex = sexes.pop() if len(sexes) == 1 else None
    return result


def _determined_names(verdict: Verdict) -> dict[str, str | None]:
    """Return the determined name per rank; the species of an aggregate counts as undetermined, as in the verdict."""
    lowest = _lowest(verdict)
    aggregate = has_qualifier(allowed_qualifier(verdict.qualifier, lowest), ID_QUALIFIER_AGG)
    return {rank: name if name not in (None, "", INDET_LABEL) and not (aggregate and rank == "species") else None
            for rank, name in ((r, verdict.val.get(r)) for r in VALIDATION_RANKS)}


def _lowest(verdict: Verdict) -> str | None:
    """Return the deepest determined rank of a verdict."""
    ranks = verdict.determinate_ranks()
    return ranks[-1] if ranks else None


def _new_folder(run: ValidationRun, now: datetime | None) -> Path:
    """Create and return a new export folder named by the export time (a counter within the same second)."""
    stamp = (now or datetime.now()).strftime(STAMP_FORMAT)  # noqa: DTZ005
    base = f"{run.final_path.stem}{EXPORT_SUFFIX}{_MULTI_SUFFIX}_{stamp}"
    folder, n = run.final_path.with_name(base), 1
    while folder.exists():
        n += 1
        folder = run.final_path.with_name(f"{base}-{n}")
    folder.mkdir(parents=True)
    return folder


def _copy_and_load(paths: Sequence[Path], directory: Path) -> list[ValidationSession]:
    """Copy each session's CSV, JSON and history into 'directory' and load the copies, in the given order."""
    directory.mkdir()
    sessions = []
    for path in paths:
        for source in session_files(path):
            if source.is_file() and source.suffix in (".csv", ".json", ".jsonl"):
                shutil.copy2(source, directory / source.name)
        sessions.append(ValidationSession.load(directory / path.name))
    return sessions


def _export_frame(sessions: Sequence[ValidationSession],
                  selection: VerdictSelection) -> tuple[pl.DataFrame, dict[str, int]]:
    """Return the rows of every sampled track for validated_frame() (with MULTI_COLUMNS), and the track counts.

    A track without a verdict in any session takes its stratum from the first session that sampled it.
    """
    sampled_by: dict[TrackKey, ValidationSession] = {}
    for session in sessions:
        queue = session.keys()
        for key in queue:
            sampled_by.setdefault(key, session)
    rows = []
    for key, first_sampled in sampled_by.items():
        sel = selection.verdicts.get(key)
        session = sel.session if sel is not None else first_sampled
        base = session.base_row(key) or {}
        if sel is None:
            row = verdict_columns(None, (None, None, None), None, session.seconds_spent(key))
            stems, n_sessions, conflict = [session.stem], 0, None
        else:
            derived = derive_verdict(session.model_lineage(key), sel.verdict, session.model_rank)
            validators = _JOIN_SEPARATOR.join(dict.fromkeys(s.validator for s in sel.contributors))
            row = verdict_columns(sel.verdict, derived, validators, session.seconds_spent(key))
            stems, n_sessions, conflict = [s.stem for s in sel.contributors], len(sel.candidates), sel.conflict
        rows.append({
            **dict(zip(TRACK_KEY_COLUMNS, key, strict=True)),
            "stratum_id": base.get("stratum_id"),
            "stratum_n_population": base.get("stratum_n_population"),
            **row,
            "validation_session": _JOIN_SEPARATOR.join(stems),
            "validation_n_sessions": n_sessions,
            "validation_conflict": conflict,
        })
    schema = {**dict.fromkeys(TRACK_KEY_COLUMNS, pl.String), "stratum_id": pl.String,
              "stratum_n_population": pl.Int64, **verdict_frame([]).schema, "validation_session": pl.String,
              "validation_n_sessions": pl.Int64, "validation_conflict": pl.String}
    several = [sel for sel in selection.verdicts.values() if len(sel.candidates) > 1]
    counts = {
        "n_sampled": len(sampled_by),
        "n_validated": sum(1 for sel in selection.verdicts.values() if sel.verdict.validated),
        "n_validated_in_several_sessions": len(several),
        **{conflict: sum(1 for sel in several if sel.conflict == conflict) for conflict in VALIDATION_CONFLICTS},
    }
    return pl.DataFrame(rows, schema=schema), counts


def _conflicts_frame(run: ValidationRun, selection: VerdictSelection) -> pl.DataFrame:
    """Return every verdict of the tracks validated in several sessions, in the order of the final CSV.

    'used' tells the verdicts the exported determination comes from.
    """
    order = {key: i for i, key in enumerate(run.tracks.select(TRACK_KEY_COLUMNS).iter_rows())}
    rows = []
    for key, sel in sorted(selection.verdicts.items(), key=lambda item: order.get(item[0], len(order))):
        if len(sel.candidates) < 2:
            continue
        for session, verdict in sel.candidates:
            derived = derive_verdict(session.model_lineage(key), verdict, session.model_rank)
            rows.append({
                **dict(zip(TRACK_KEY_COLUMNS, key, strict=True)),
                "validation_conflict": sel.conflict,
                "validation_session": session.stem,
                "used": session in sel.contributors,
                "blind": session.blind,
                **verdict_columns(verdict, derived, session.validator, session.seconds_spent(key)),
            })
    schema = {**dict.fromkeys(TRACK_KEY_COLUMNS, pl.String), "validation_conflict": pl.String,
              "validation_session": pl.String, "used": pl.Boolean, "blind": pl.Boolean, **verdict_frame([]).schema}
    return pl.DataFrame(rows, schema=schema)


def _session_record(session: ValidationSession, directory: Path) -> dict[str, Any]:
    """Return the provenance entry of one session: its settings and its copied files with their checksums."""
    meta = session.meta
    files = [path for path in sorted(directory.glob(f"{session.stem}_*")) if path.is_file()]
    return {
        "validation_session": session.stem,
        **{key: meta[key] for key in ("validator", "validation_session_id", "started", "updated", "app_version",
                                      "blind", "taxonomy", "sampling", "n_sampled", "n_validated")},
        "files": [file_provenance(path) for path in files],
    }


def _custom_taxa(sessions: Sequence[ValidationSession]) -> list[dict[str, Any]]:
    """Return the free-text lineages of all sessions, each once, in session order."""
    taxa: list[dict[str, Any]] = []
    for session in sessions:
        taxa.extend(entry for entry in session.meta["custom_taxa"] if entry not in taxa)
    return taxa

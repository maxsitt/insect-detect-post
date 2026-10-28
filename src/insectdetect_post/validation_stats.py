"""Statistics of validation results (model vs. validator).

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

All functions take the session table ('ValidationSession.to_frame()') and use validated tracks
only. Not-assessable tracks (low image quality, different taxa in the crops or along the track) are
excluded from the accuracies unless
'include_not_assessable' is set. 'include_uncertain=False' drops 'inc.' determinations from
their deepest determined rank. Accuracy and precision come with Wilson 95 % intervals, kappa with
its large-sample interval. A stratified sample (sampling fractions differ between strata) adds
stratum-weighted estimates for the tracks of the sampled strata (the run's tracks after the session's
track selection, without strata below the minimum size): every validated track stands for
N_h / n_h tracks of its stratum, and only strata with a validated track count.

Functions:
    column_description(): Meaning of a column of a statistics table.
    wilson_interval(): Wilson score interval of a proportion.
    cohens_kappa(): Cohen's kappa of paired categorical labels.
    cohens_kappa_interval(): Large-sample confidence interval of Cohen's kappa.
    is_stratified(): Whether the sampling fractions differ between strata.
    validated_in_queue_order(): Whether the validated tracks are the first ones of the queue.
    n_low_quality(), n_different_taxa_in_crops(), n_different_taxa_in_track(): Counts of the not-assessable
        tracks.
    n_uncertain(): Count of the uncertain (inc.) determinations.
    n_individuals(): Distinct individuals among validated tracks.
    progress_by_stratum(): Validated vs. sampled tracks per stratum.
    unvalidated_strata(): Sampled strata without a validated track.
    verdict_distribution(): Counts and shares of the verdict types and flags.
    verdict_table(): The verdict distribution with one row per verdict type or flag.
    agreement_by_rank(): Accuracy per rank (sample and stratum-weighted) with CIs.
    reliability_by_prob_bin(): Accuracy per weighted-probability bin.
    per_taxon_table(): Outcome, precision, recall and F1 per predicted taxon.
    confusion_matrix(): Model vs. validator confusion matrix at a rank.
    confusion_pairs(): Every disagreement at a rank, most frequent first.
    inter_validator_agreement(): Agreement and kappa between two validators on shared tracks.
    compare_sessions(): Agreement with every other session of the run.
    linked_track_conflicts(): Contradicting determinations of linked tracks.
    effort_summary(): Validation effort (time) overall.
    effort_by_taxon(): Validation effort per predicted taxon.
    session_summary(): Key figures of a session.
    export_stats(): Write all tables as CSV files and the export record 'summary.json'.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from insectdetect_post.constants import TRACK_KEY_COLUMNS
from insectdetect_post.validation_sampling import STRATUM_GROUP_COLUMNS, prob_bin_labels
from insectdetect_post.validation_session import (
    DIFFERENT_TAXA_IN_CROPS,
    DIFFERENT_TAXA_IN_TRACK,
    ID_QUALIFIER_AGG,
    ID_QUALIFIER_INC,
    MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE,
    MULTIPLE_INDIVIDUALS_TYPES,
    VERDICT_TYPES,
    has_qualifier,
    individual_groups,
    parse_links,
)
from insectdetect_post.validation_taxonomy import INDET_LABEL, VALIDATION_RANKS, ranks_up_to

# Verdicts whose lineage is compared with the model
_COMPARABLE_VERDICTS = ("agree", "partial", "corrected")

# Decimals of the exported fractions, kappa values and intervals (times are rounded to 0.1 s already)
_EXPORT_DECIMALS = 4

# Meaning of the cells of 'confusion_matrix_<rank>.csv' (one column per taxon)
_MATRIX_CELL_DESCRIPTION = "Tracks predicted as the taxon of the row and determined as the taxon of the column"

# Meaning of every column of the statistics tables (used in summary.json, header tooltips in the dialog)
# STATS_TABLE_DESCRIPTIONS overrides it where a table uses the column differently
STATS_COLUMN_DESCRIPTIONS = {
    "rank": "Taxonomic rank",
    "prob_bin": "Range of the model's weighted probability (lower edge included, the last bin also includes 1)",
    "level": "What the sessions are compared on: the false-positive verdict, the verdict type or the taxon at a rank",
    "category": f"Verdict type ({', '.join(VERDICT_TYPES)}) or flag (uncertain, aggregate, linked, "
                f"{', '.join(MULTIPLE_INDIVIDUALS_TYPES)}, low_quality); "
                "a track has one verdict type and any number of flags",
    "taxon": "Taxon the model predicted, at the rank in the file name",
    "model_taxon": "Taxon the model predicted",
    "human_taxon": "Taxon the validator determined instead",
    "device_id": "Camera the tracks were recorded with",
    "session_id": "Recording session of the camera",
    "track_id": "Tracking ID of the track",
    "linked_track_id": "Earlier track it was linked to as the same individual",
    "val_track": "Validator's taxon of the track at this rank",
    "val_linked": "Validator's taxon of the linked track at this rank",
    "stratum_id": "Stratum the tracks were sampled from",
    "stratum_setting": "Camera setting of the stratum",
    "stratum_taxon": "Predicted taxon of the stratum",
    "stratum_prob_bin": "Range of the model's weighted probability of the stratum (lower edge included)",
    "other_session": "Validator and start time of the other session, as in its file names",
    "comparison": "Whether the other session is by another validator or the same one (test-retest)",
    "n_population": "Tracks of the stratum in the selection (its population)",
    "n_sampled": "Tracks drawn from the stratum for validation",
    "n_validated": "Tracks with a verdict",
    "n_tracks": "Validated tracks",
    "n_comparable": "Validated tracks that the model and the validator both named at this rank (no false positives, "
                    "no agg. at this rank; not-assessable and uncertain tracks as set in the export settings)",
    "n_agree": "Comparable tracks where the validator confirmed the model's taxon",
    "n_human_determinate": "Validated tracks the validator named at this rank (an agg. names no species); below the "
                           "rank of the run the only count, showing how often the validator could go deeper than "
                           "the model",
    "n_correct": "Tracks where the validator confirmed the model's taxon",
    "n_wrong": "Tracks the validator determined as another taxon",
    "n_indet_at_rank": "Tracks the validator determined only down to a higher rank",
    "n_false_positive": "Tracks the validator marked as not an arthropod (false positive)",
    "n_no_taxon": "Tracks without any determination (verdict indeterminate or not_assessable)",
    "n_not_assessable": "Tracks with low image quality or different taxa (in the crops or along the track); in the "
                        "other counts only if the export settings include them",
    "n_human": "Tracks the validator determined as this taxon (recall denominator)",
    "n_shared": "Tracks both sessions validated independently, i.e. not pre-filled from the other session "
                "(at a rank: both named a taxon there)",
    "share": "Share of the validated tracks: n_tracks / n_validated",
    "share_weighted": "Share estimated for the tracks of the sampled strata: "
                      "every stratum with validated tracks counts with its population size",
    "share_validated": "n_validated / n_sampled",
    "share_agree": "n_agree / n_shared",
    "share_corrected": "Share of the tracks with the verdict corrected",
    "accuracy": "Share of the comparable tracks the model got right: n_agree / n_comparable",
    "accuracy_ci_low": "Lower end of the Wilson 95 % interval of the accuracy",
    "accuracy_ci_high": "Upper end of the Wilson 95 % interval of the accuracy",
    "accuracy_weighted": "Accuracy estimated for the tracks of the sampled strata: "
                         "every stratum with validated tracks counts with its population size",
    "accuracy_weighted_ci_low": "Lower end of the 95 % interval of the weighted accuracy (empty at zero variance)",
    "accuracy_weighted_ci_high": "Upper end of the 95 % interval of the weighted accuracy (empty at zero variance)",
    "precision": "Share of the tracks predicted as this taxon that the validator confirmed: "
                 "n_correct / (n_correct + n_wrong); false positives are detection errors, counted separately",
    "precision_ci_low": "Lower end of the Wilson 95 % interval of the precision",
    "precision_ci_high": "Upper end of the Wilson 95 % interval of the precision",
    "precision_weighted": "Precision estimated for the tracks of the sampled strata: "
                          "every stratum with validated tracks counts with its population size",
    "precision_weighted_ci_low": "Lower end of the 95 % interval of the weighted precision (empty at zero variance)",
    "precision_weighted_ci_high": "Upper end of the 95 % interval of the weighted precision (empty at zero variance)",
    "recall": "Share of the tracks the validator determined as this taxon that the model predicted as it: "
              "n_correct / n_human; unstratified samples only",
    "f1": "Harmonic mean of precision and recall",
    "kappa": "Cohen's kappa: the agreement corrected for the label frequencies",
    "kappa_ci_low": "Lower end of the 95 % interval of kappa",
    "kappa_ci_high": "Upper end of the 95 % interval of kappa",
    "mean_seconds": "Mean validation time per track in seconds",
    "median_seconds": "Median validation time per track in seconds",
}

# Meaning of the columns a table uses differently, by table (file name without the rank suffix)
STATS_TABLE_DESCRIPTIONS = {
    "progress_by_stratum": {"n_validated": "Tracks of the stratum that already have a verdict"},
    "verdict_distribution": {"n_tracks": "Validated tracks with this verdict type or flag",
                             "n_validated": "Validated tracks of the session"},
    "verdict_distribution_by_stratum": {"n_tracks": "Validated tracks of the stratum with this verdict type or flag",
                                        "n_validated": "Validated tracks of the stratum"},
    "linked_track_conflicts": {"rank": "Highest rank at which the two tracks were determined differently",
                               "track_id": "Track that was linked to an earlier one as the same individual"},
    "effort_by_taxon": {"n_validated": "Validated tracks with this model prediction and a recorded validation time"},
    "per_taxon": {"n_validated": "Validated tracks with this model prediction (not-assessable ones only if the "
                                 "export settings include them)"},
    "confusion_pairs": {"n_tracks": "Comparable tracks with this pair of predicted and determined taxon"},
    "other_sessions": {"n_agree": "Shared tracks both sessions determined the same way"},
}

def column_description(table: str, column: str) -> str:
    """Return the meaning of a column of a statistics table ('table': its file name without '.csv')."""
    kind, _, suffix = table.rpartition("_")
    overrides = STATS_TABLE_DESCRIPTIONS.get(kind if suffix in VALIDATION_RANKS else table, {})
    return overrides.get(column, STATS_COLUMN_DESCRIPTIONS.get(column, ""))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Return the Wilson score interval of a proportion 'successes / total' (None, None if total is 0)."""
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Return Cohen's kappa of paired labels (None if no pairs or if chance agreement is 1)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[label] * right.get(label, 0) for label in left) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def cohens_kappa_interval(pairs: list[tuple[str, str]], z: float = 1.96) -> tuple[float | None, float | None]:
    """Return the large-sample confidence interval of Cohen's kappa (Fleiss, Cohen & Everitt 1969).

    Args:
        pairs: Paired labels.
        z: Normal quantile (1.96 = 95 %).

    Returns:
        '(low, high)' clipped to [-1, 1]. '(None, None)' if kappa is undefined, with fewer than
        two pairs or zero variance (perfect agreement).
    """
    kappa = cohens_kappa(pairs)
    n = len(pairs)
    if kappa is None or n < 2:
        return None, None
    row = Counter(a for a, _ in pairs)  # marginals: model (rows), validator (columns)
    col = Counter(b for _, b in pairs)
    p_e = sum(row[label] * col.get(label, 0) for label in row) / (n * n)
    diagonal = off_diagonal = 0.0
    for (a, b), count in Counter(pairs).items():
        share = count / n
        if a == b:
            diagonal += share * (1 - (row[a] + col[a]) / n * (1 - kappa)) ** 2
        else:
            off_diagonal += share * ((col[a] + row[b]) / n) ** 2
    variance = (diagonal + (1 - kappa) ** 2 * off_diagonal
                - (kappa - p_e * (1 - kappa)) ** 2) / (n * (1 - p_e) ** 2)
    if variance <= 1e-12:  # zero up to rounding, e.g. when one rater gives a single label
        return None, None
    half = z * math.sqrt(variance)
    return max(-1.0, kappa - half), min(1.0, kappa + half)


def is_stratified(frame: pl.DataFrame) -> bool:
    """Return whether the sampling fractions differ between strata (then only weighted estimates describe the run)."""
    fractions = (
        frame.group_by("stratum_id")
        .agg((pl.len() / pl.col("stratum_n_population").first()).alias("fraction"))["fraction"]
        .drop_nulls()
    )
    return fractions.len() > 1 and bool(fractions.max() - fractions.min() > 1e-9)  # type: ignore[operator]


def validated_in_queue_order(frame: pl.DataFrame) -> bool:
    """Return whether the validated tracks are the first ones of the queue (the session table's order)."""
    validated = frame["validated"].fill_null(False)
    return bool(validated.head(int(validated.sum())).all())


def n_low_quality(frame: pl.DataFrame) -> int:
    """Return the number of validated tracks flagged as low image quality."""
    return frame.filter(pl.col("validated") & (pl.col("image_quality") == "low")).height


def n_different_taxa_in_crops(frame: pl.DataFrame) -> int:
    """Return the number of validated tracks flagged as showing different taxa inside their crops."""
    return frame.filter(pl.col("validated") & (pl.col("multiple_individuals") == DIFFERENT_TAXA_IN_CROPS)).height


def n_different_taxa_in_track(frame: pl.DataFrame) -> int:
    """Return the number of validated tracks flagged as showing different taxa along the track (tracking error)."""
    return frame.filter(pl.col("validated") & (pl.col("multiple_individuals") == DIFFERENT_TAXA_IN_TRACK)).height


def n_uncertain(frame: pl.DataFrame) -> int:
    """Return the number of validated tracks with an uncertain (inc.) determination."""
    return frame.filter(pl.col("validated") & _has_qualifier(ID_QUALIFIER_INC)).height


def n_individuals(frame: pl.DataFrame) -> int:
    """Return the number of distinct individuals among validated tracks (linked tracks once, false positives not)."""
    validated = frame.filter(pl.col("validated") & (pl.col("verdict").fill_null("") != "false_positive"))
    if validated.is_empty():
        return 0
    groups = individual_groups(validated)
    return groups.select(["device_id", "session_id", "individual_id"]).unique().height


def progress_by_stratum(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the validated vs. sampled tracks per stratum.

    Returns:
        'stratum_id, stratum_setting, stratum_taxon, stratum_prob_bin, n_population, n_sampled,
        n_validated, share_validated'.
    """
    return (
        frame.group_by(STRATUM_GROUP_COLUMNS)
        .agg(
            pl.col("stratum_n_population").first().alias("n_population"),
            pl.len().alias("n_sampled"),
            pl.col("validated").cast(pl.Boolean).sum().alias("n_validated"),
        )
        .with_columns((pl.col("n_validated") / pl.col("n_sampled")).alias("share_validated"))
        .sort(STRATUM_GROUP_COLUMNS[1:], nulls_last=True)
    )


def unvalidated_strata(frame: pl.DataFrame) -> pl.DataFrame:
    """Return 'stratum_id, n_population' of the sampled strata without a validated track (no weighted estimate)."""
    return (
        frame.group_by("stratum_id")
        .agg(pl.col("stratum_n_population").first().alias("n_population"), pl.col("validated").any().alias("_any"))
        .filter(~pl.col("_any")).drop("_any").sort("stratum_id")
    )


def verdict_distribution(frame: pl.DataFrame, by_stratum: bool = False) -> pl.DataFrame:
    """Return the counts and shares of verdicts and flags over validated tracks.

    Overall (not 'by_stratum'), a stratified sample adds 'share_<name>_weighted': the shares estimated
    for the tracks of the sampled strata.
    """
    df = frame.filter(pl.col("validated"))
    conditions = {verdict: pl.col("verdict") == verdict for verdict in VERDICT_TYPES}
    conditions["uncertain"] = _has_qualifier(ID_QUALIFIER_INC)
    conditions["aggregate"] = _has_qualifier(ID_QUALIFIER_AGG)
    # Tracks continuing an earlier track (fragmentation)
    conditions["linked"] = pl.col("same_individual_as").is_not_null()
    conditions |= {kind: pl.col("multiple_individuals") == kind for kind in MULTIPLE_INDIVIDUALS_TYPES}
    conditions["low_quality"] = pl.col("image_quality") == "low"
    aggs = [pl.len().alias("n_validated"), *[cond.sum().alias(f"n_{name}") for name, cond in conditions.items()]]
    group_cols = ["stratum_id"] if by_stratum else []
    result = df.group_by(group_cols).agg(aggs) if group_cols else df.select(aggs)
    shares = [(pl.col(f"n_{name}") / pl.col("n_validated")).alias(f"share_{name}") for name in conditions]
    # No validated tracks: empty instead of NaN
    result = result.with_columns(shares).with_columns(pl.col("^share_.*$").fill_nan(None))
    if group_cols:
        return result.sort("stratum_id")
    if is_stratified(frame) and not df.is_empty():
        weight = _design_weight()
        weighted = df.select(
            ((weight * cond.fill_null(False).cast(pl.Float64)).sum() / weight.sum()).alias(f"share_{name}_weighted")
            for name, cond in conditions.items()
        )
        result = result.hstack(weighted)  # one row each
    return result


def verdict_table(frame: pl.DataFrame, by_stratum: bool = False) -> pl.DataFrame:
    """Return the verdict distribution with one row per verdict type or flag.

    Returns:
        '[stratum_id,] category, n_tracks, n_validated, share[, share_weighted]'; the weighted share
        overall for a stratified sample only.
    """
    dist = verdict_distribution(frame, by_stratum)
    names = [c.removeprefix("n_") for c in dist.columns if c.startswith("n_") and c != "n_validated"]
    weighted = f"share_{names[0]}_weighted" in dist.columns
    stratum: list[str] = ["stratum_id"] if by_stratum else []
    rows = [
        {**{c: row[c] for c in stratum}, "category": name, "n_tracks": row[f"n_{name}"],
         "n_validated": row["n_validated"], "share": row[f"share_{name}"],
         **({"share_weighted": row[f"share_{name}_weighted"]} if weighted else {})}
        for row in dist.iter_rows(named=True) for name in names
    ]
    schema = {**{c: pl.String for c in stratum}, "category": pl.String, "n_tracks": pl.Int64,
              "n_validated": pl.Int64, "share": pl.Float64, **({"share_weighted": pl.Float64} if weighted else {})}
    return pl.DataFrame(rows, schema=schema)


def agreement_by_rank(
    frame: pl.DataFrame, model_rank: str, include_not_assessable: bool = False, include_uncertain: bool = True
) -> pl.DataFrame:
    """Return the accuracy per rank down to 'model_rank'. Deeper ranks only count determinations.

    Returns:
        'rank, n_comparable, n_agree, accuracy, accuracy_ci_low, accuracy_ci_high, [accuracy_weighted,
        accuracy_weighted_ci_low, accuracy_weighted_ci_high,] n_human_determinate'; the weighted columns
        for a stratified sample only.
    """
    validated = _validated(frame, include_not_assessable)
    all_validated = frame.filter(pl.col("validated"))
    rows = []
    comparable_ranks = ranks_up_to(model_rank)
    for rank in VALIDATION_RANKS:
        val_col = pl.col(f"val_{rank}")
        # An aggregate ('agg.') does not determine its deepest rank, as in _comparable_mask()
        aggregate = _lowest_rank_qualifier(ID_QUALIFIER_AGG, rank).fill_null(False)
        n_det = validated.filter(val_col.is_not_null() & (val_col != INDET_LABEL) & ~aggregate).height
        if rank in comparable_ranks:
            df = _comparable(frame, rank, include_not_assessable, include_uncertain)
            n = df.height
            k = int(df["_correct"].sum()) if n else 0
            lo, hi = wilson_interval(k, n)
            mask = _comparable_mask(rank, include_not_assessable, include_uncertain)
            weighted, w_lo, w_hi = _weighted_ratio(all_validated, mask & _correct_expr(rank), mask)
            rows.append({
                "rank": rank, "n_comparable": n, "n_agree": k,
                "accuracy": k / n if n else None, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
                "accuracy_weighted": weighted, "accuracy_weighted_ci_low": w_lo, "accuracy_weighted_ci_high": w_hi,
                "n_human_determinate": n_det,
            })
        else:
            rows.append({
                "rank": rank, "n_comparable": None, "n_agree": None, "accuracy": None, "accuracy_ci_low": None,
                "accuracy_ci_high": None, "accuracy_weighted": None, "accuracy_weighted_ci_low": None,
                "accuracy_weighted_ci_high": None,
                "n_human_determinate": n_det,
            })
    schema = {
        "rank": pl.String, "n_comparable": pl.Int64, "n_agree": pl.Int64, "accuracy": pl.Float64,
        "accuracy_ci_low": pl.Float64, "accuracy_ci_high": pl.Float64, "accuracy_weighted": pl.Float64,
        "accuracy_weighted_ci_low": pl.Float64, "accuracy_weighted_ci_high": pl.Float64,
        "n_human_determinate": pl.Int64,
    }
    return _weighted_columns_if_stratified(pl.DataFrame(rows, schema=schema), frame)


def reliability_by_prob_bin(
    frame: pl.DataFrame,
    rank: str,
    edges: list[float],
    include_not_assessable: bool = False,
    include_uncertain: bool = True,
) -> pl.DataFrame:
    """Return the accuracy at 'rank' per bin of the weighted prediction probability.

    Returns:
        'prob_bin, n_comparable, n_agree, accuracy, accuracy_ci_low, accuracy_ci_high[, accuracy_weighted,
        accuracy_weighted_ci_low, accuracy_weighted_ci_high]' for every bin; the weighted columns for a
        stratified sample only.
    """
    labels = prob_bin_labels(edges)
    prob_bin = (pl.col("pred_prob_weighted").cast(pl.Float64, strict=False).fill_nan(None)
                .bin_intervals(edges[1:-1], labels=labels).cast(pl.String))
    binned = frame.filter(pl.col("validated")).with_columns(prob_bin.alias("prob_bin"))
    mask = _comparable_mask(rank, include_not_assessable, include_uncertain)
    rows = []
    for label in labels:
        in_bin = mask & (pl.col("prob_bin") == label).fill_null(False)
        n = int(binned.select(in_bin.sum()).item())
        k = int(binned.select((in_bin & _correct_expr(rank)).sum()).item())
        lo, hi = wilson_interval(k, n)
        weighted, w_lo, w_hi = _weighted_ratio(binned, in_bin & _correct_expr(rank), in_bin)
        rows.append({
            "prob_bin": label, "n_comparable": n, "n_agree": k,
            "accuracy": k / n if n else None, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
            "accuracy_weighted": weighted, "accuracy_weighted_ci_low": w_lo, "accuracy_weighted_ci_high": w_hi,
        })
    schema = {"prob_bin": pl.String, "n_comparable": pl.Int64, "n_agree": pl.Int64, "accuracy": pl.Float64,
              "accuracy_ci_low": pl.Float64, "accuracy_ci_high": pl.Float64, "accuracy_weighted": pl.Float64,
              "accuracy_weighted_ci_low": pl.Float64, "accuracy_weighted_ci_high": pl.Float64}
    return _weighted_columns_if_stratified(pl.DataFrame(rows, schema=schema), frame)


def per_taxon_table(
    frame: pl.DataFrame, rank: str, include_not_assessable: bool = False, include_uncertain: bool = True
) -> pl.DataFrame:
    """Return the outcome of the validated tracks per predicted taxon at 'rank'.

    Precision rates the classification: false positives are errors of the detection and are counted
    separately. A stratified sample gets the weighted precision instead of recall and F1, as its design
    sets the recall denominator.

    Returns:
        'taxon, n_validated, n_correct, n_wrong, n_indet_at_rank, n_false_positive, n_no_taxon,
        n_not_assessable, n_human, precision, precision_ci_low, precision_ci_high' and 'recall, f1' or
        'precision_weighted, precision_weighted_ci_low, precision_weighted_ci_high', largest first.
    """
    validated = _validated(frame, include_not_assessable)
    pred_col, val_col = pl.col(f"pred_{rank}"), pl.col(f"val_{rank}")
    not_assessable = frame.filter(pl.col("validated") & _not_assessable_expr())
    comparable_mask = pl.col("verdict").is_in(_COMPARABLE_VERDICTS)
    if not include_uncertain:
        comparable_mask = comparable_mask & ~_lowest_rank_qualifier(ID_QUALIFIER_INC, rank)
    # The species of an aggregate stays open, like an indet. rank
    determined = (val_col.is_not_null() & (val_col != INDET_LABEL)
                  & ~_lowest_rank_qualifier(ID_QUALIFIER_AGG, rank)).fill_null(False)
    stats = (
        validated.filter(pred_col.is_not_null())
        .group_by(pred_col.alias("taxon"))
        .agg(
            pl.len().alias("n_validated"),
            (comparable_mask & determined & (val_col == pred_col)).sum().alias("n_correct"),
            (comparable_mask & determined & (val_col != pred_col)).sum().alias("n_wrong"),
            (comparable_mask & ~determined).sum().alias("n_indet_at_rank"),
            (pl.col("verdict") == "false_positive").sum().alias("n_false_positive"),
            (pl.col("verdict").is_in(["indeterminate", "not_assessable"])).sum().alias("n_no_taxon"),
        )
    )
    excluded = not_assessable.filter(pred_col.is_not_null()).group_by(pred_col.alias("taxon")).len().rename(
        {"len": "n_not_assessable"}
    )
    human_totals = (
        validated.filter(comparable_mask & determined)
        .group_by(val_col.alias("taxon")).len().rename({"len": "n_human"})
    )
    count_cols = ["n_validated", "n_correct", "n_wrong", "n_indet_at_rank", "n_false_positive", "n_no_taxon"]
    table = (
        # Full join keeps taxa whose tracks are all not assessable
        stats.join(excluded, on="taxon", how="full", coalesce=True).join(human_totals, on="taxon", how="left")
        .with_columns(pl.col(*count_cols, "n_not_assessable", "n_human").fill_null(0))
        .with_columns((pl.col("n_correct") / (pl.col("n_correct") + pl.col("n_wrong"))).alias("precision"))
    )
    intervals = [wilson_interval(c, c + w) for c, w in zip(table["n_correct"], table["n_wrong"], strict=True)]
    table = table.with_columns(
        pl.Series("precision_ci_low", [lo for lo, _ in intervals], dtype=pl.Float64),
        pl.Series("precision_ci_high", [hi for _, hi in intervals], dtype=pl.Float64),
    )
    if is_stratified(frame):
        # Recall is not estimable from a stratified sample. Precision: the ratio estimator of the
        # accuracy, per predicted taxon (every validated track of a stratum counts for N_h / n_h)
        judged = _comparable_mask(rank, include_not_assessable, include_uncertain)
        weighted = _weighted_ratios(frame.filter(pl.col("validated")), pred_col, judged & _correct_expr(rank), judged)
        table = table.join(weighted.rename({
            "_group": "taxon", "estimate": "precision_weighted", "ci_low": "precision_weighted_ci_low",
            "ci_high": "precision_weighted_ci_high",
        }), on="taxon", how="left")
    else:
        table = table.with_columns((pl.col("n_correct") / pl.col("n_human")).alias("recall")).with_columns(
            # Harmonic mean, 0 if both are 0 (no correct track)
            pl.when(pl.col("precision") + pl.col("recall") == 0).then(0.0)
            .otherwise(2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
            .alias("f1")
        )
    return (
        table.with_columns(pl.col(pl.Float64).fill_nan(None))
        .sort(["n_validated", "taxon"], descending=[True, False])
    )


def confusion_matrix(
    frame: pl.DataFrame,
    rank: str,
    include_not_assessable: bool = False,
    include_uncertain: bool = True,
) -> tuple[list[str], list[list[int]]]:
    """Return the model (rows) vs. validator (columns) confusion matrix at 'rank'.

    Args:
        frame: Session table.
        rank: Rank.
        include_not_assessable: Include not-assessable tracks (low image quality, different taxa).
        include_uncertain: Include 'inc.' determinations.

    Returns:
        '(labels, matrix)'; 'matrix[i][j]' = tracks predicted as 'labels[i]', validated as 'labels[j]',
        labels ordered by frequency.
    """
    df = _comparable(frame, rank, include_not_assessable, include_uncertain)
    pairs = list(zip(df[f"pred_{rank}"].to_list(), df[f"val_{rank}"].to_list(), strict=True))
    counts = Counter(pairs)
    freq: Counter[str] = Counter()
    for (pred, val), n in counts.items():
        freq[pred] += n
        freq[val] += n
    labels = [label for label, _ in freq.most_common()]
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for (pred, val), n in counts.items():
        matrix[index[pred]][index[val]] += n
    return labels, matrix


def confusion_pairs(
    frame: pl.DataFrame, rank: str, include_not_assessable: bool = False, include_uncertain: bool = True
) -> pl.DataFrame:
    """Return every (model taxon, validator taxon) disagreement at 'rank' with its count, most frequent first."""
    df = _comparable(frame, rank, include_not_assessable, include_uncertain).filter(~pl.col("_correct"))
    return (
        df.group_by(pl.col(f"pred_{rank}").alias("model_taxon"), pl.col(f"val_{rank}").alias("human_taxon"))
        .len().rename({"len": "n_tracks"})
        .sort(["n_tracks", "model_taxon", "human_taxon"], descending=[True, False, False])
    )


def inter_validator_agreement(frame_a: pl.DataFrame, frame_b: pl.DataFrame, model_rank: str) -> pl.DataFrame:
    """Return the agreement between two validators on the tracks both validated.

    Returns:
        Rows for 'false_positive', 'verdict' and each rank down to 'model_rank': 'n_shared,
        n_agree, share_agree, kappa' with interval.
    """
    shared = (
        frame_a.filter(pl.col("validated"))
        .join(frame_b.filter(pl.col("validated")), on=TRACK_KEY_COLUMNS, suffix="_b")
    )
    rows = []
    fp_pairs = list(zip(
        (shared["verdict"] == "false_positive").to_list(), (shared["verdict_b"] == "false_positive").to_list(),
        strict=True,
    ))
    rows.append(_agreement_row("false_positive", [(str(a), str(b)) for a, b in fp_pairs]))
    rows.append(_agreement_row("verdict", list(zip(
        shared["verdict"].fill_null("").to_list(), shared["verdict_b"].fill_null("").to_list(), strict=True
    ))))
    for rank in ranks_up_to(model_rank):
        a, b = pl.col(f"val_{rank}"), pl.col(f"val_{rank}_b")
        det = shared.filter(a.is_not_null() & (a != INDET_LABEL) & b.is_not_null() & (b != INDET_LABEL))
        labels = det.select(_determination(rank).alias("a"), _determination(rank, "_b").alias("b"))
        rows.append(_agreement_row(rank, list(zip(labels["a"].to_list(), labels["b"].to_list(), strict=True))))
    return pl.DataFrame(rows, schema={"level": pl.String, "n_shared": pl.Int64, "n_agree": pl.Int64,
                                      "share_agree": pl.Float64, "kappa": pl.Float64,
                                      "kappa_ci_low": pl.Float64, "kappa_ci_high": pl.Float64})


def compare_sessions(
    frame: pl.DataFrame,
    validator: str,
    others: Sequence[tuple[str, str, pl.DataFrame]],
    model_rank: str,
    stem: str | None = None,
) -> pl.DataFrame:
    """Return the agreement of a session with every other session of the run.

    Tracks one session pre-filled from the other are left out of their comparison, also once edited: the
    validator saw the other session's entry, so the two are no independent assessments.

    Args:
        frame: Session table.
        validator: Validator of the session.
        others: '(validator, file stem, session table)' of the other sessions (see ValidationSession.stem).
        model_rank: Deepest rank of the run.
        stem: File stem of the session.

    Returns:
        'other_session, comparison, level, n_shared, n_agree, share_agree, kappa, kappa_ci_low,
        kappa_ci_high'; 'comparison' tells another validator from the same one (test-retest).
    """
    def independent(table: pl.DataFrame, source: str | None) -> pl.DataFrame:
        if source is None or "prefilled_from" not in table.columns:
            return table
        return table.filter(pl.col("prefilled_from").fill_null("") != source)

    columns = ["other_session", "comparison", "level", "n_shared", "n_agree", "share_agree", "kappa",
               "kappa_ci_low", "kappa_ci_high"]
    rows = [
        inter_validator_agreement(independent(frame, other_stem), independent(other_frame, stem),
                                  model_rank).with_columns(
            pl.lit(other_stem).alias("other_session"),
            pl.lit("same validator (test-retest)" if other_validator == validator else "other validator")
            .alias("comparison"),
        ).select(columns)
        for other_validator, other_stem, other_frame in others
    ]
    if not rows:
        return pl.DataFrame(schema={"other_session": pl.String, "comparison": pl.String, "level": pl.String,
                                    "n_shared": pl.Int64, "n_agree": pl.Int64, "share_agree": pl.Float64,
                                    "kappa": pl.Float64, "kappa_ci_low": pl.Float64, "kappa_ci_high": pl.Float64})
    return pl.concat(rows)


def linked_track_conflicts(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the linked tracks whose determinations contradict (different names at the same rank).

    Returns:
        'device_id, session_id, track_id, linked_track_id, rank, val_track, val_linked'.
    """
    schema = {"device_id": pl.String, "session_id": pl.String, "track_id": pl.String, "linked_track_id": pl.String,
              "rank": pl.String, "val_track": pl.String, "val_linked": pl.String}
    validated = frame.filter(pl.col("validated") & pl.col("verdict").is_in(_COMPARABLE_VERDICTS))
    by_key = {tuple(r[c] for c in TRACK_KEY_COLUMNS): r for r in validated.to_dicts()}
    rows = []
    seen: set[tuple] = set()
    for row in validated.filter(pl.col("same_individual_as").is_not_null()).to_dicts():
        key = tuple(row[c] for c in TRACK_KEY_COLUMNS)
        for other_id in parse_links(row["same_individual_as"]):
            other_key = (key[0], key[1], other_id)
            other = by_key.get(other_key)
            pair = tuple(sorted([key[2], other_id]))
            if other is None or (key[0], key[1], pair) in seen:
                continue
            seen.add((key[0], key[1], pair))
            # An aggregate does not contradict a species it may include
            aggregate = (has_qualifier(row.get("val_qualifier"), ID_QUALIFIER_AGG)
                         or has_qualifier(other.get("val_qualifier"), ID_QUALIFIER_AGG))
            for rank in VALIDATION_RANKS:
                a, b = row.get(f"val_{rank}"), other.get(f"val_{rank}")
                if aggregate and rank == "species":
                    continue
                if a and b and a != INDET_LABEL and b != INDET_LABEL and a != b:
                    rows.append({"device_id": key[0], "session_id": key[1], "track_id": key[2],
                                 "linked_track_id": other_id, "rank": rank, "val_track": a, "val_linked": b})
                    break  # highest contradicting rank only
    return pl.DataFrame(rows, schema=schema)


def effort_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the validation effort: tracks, total minutes (and those spent in earlier sessions on pre-filled
    tracks), mean and median seconds per track."""
    seconds = pl.col("seconds_spent").cast(pl.Float64)
    prefilled = pl.col("prefilled_from").is_not_null() if "prefilled_from" in frame.columns else pl.lit(False)
    return frame.filter(pl.col("validated") & pl.col("seconds_spent").is_not_null()).select(
        pl.len().cast(pl.Int64).alias("n_validated"),
        (seconds.sum() / 60).round(1).alias("total_minutes"),
        (seconds.filter(prefilled).sum() / 60).round(1).alias("prefilled_minutes"),
        seconds.mean().round(1).alias("mean_seconds"),
        seconds.median().round(1).alias("median_seconds"),
    )


def effort_by_taxon(frame: pl.DataFrame, rank: str) -> pl.DataFrame:
    """Return the validation effort per predicted taxon at 'rank'."""
    df = frame.filter(pl.col("validated") & pl.col("seconds_spent").is_not_null()
                      & pl.col(f"pred_{rank}").is_not_null())
    return (
        df.group_by(pl.col(f"pred_{rank}").alias("taxon"))
        .agg(
            pl.len().alias("n_validated"),
            pl.col("seconds_spent").cast(pl.Float64).mean().round(1).alias("mean_seconds"),
            pl.col("seconds_spent").cast(pl.Float64).median().round(1).alias("median_seconds"),
            (pl.col("verdict") == "corrected").mean().alias("share_corrected"),
        )
        .sort(["median_seconds", "n_validated", "taxon"], descending=[True, True, False])
    )


def session_summary(frame: pl.DataFrame) -> dict[str, Any]:
    """Return the key figures of a session: sample, verdict counts, flags and validation effort."""
    validated = frame.filter(pl.col("validated"))
    effort = effort_summary(frame).row(0, named=True)
    prefilled = validated.filter(pl.col("prefilled_from").is_not_null()) if "prefilled_from" in frame.columns else None
    return {
        "n_sampled": frame.height,
        "n_validated": validated.height,
        "n_prefilled": prefilled.height if prefilled is not None else 0,
        "n_prefill_edited": prefilled.filter(pl.col("prefill_edited")).height if prefilled is not None else 0,
        "n_false_positive": validated.filter(pl.col("verdict") == "false_positive").height,
        "n_individuals": n_individuals(frame),
        "n_low_quality": n_low_quality(frame),
        "n_different_taxa_in_crops": n_different_taxa_in_crops(frame),
        "n_different_taxa_in_track": n_different_taxa_in_track(frame),
        "n_uncertain": n_uncertain(frame),
        "stratified": is_stratified(frame),
        "n_strata": frame["stratum_id"].n_unique(),
        "n_strata_unvalidated": unvalidated_strata(frame).height,
        "validated_in_queue_order": validated_in_queue_order(frame),
        "effort_total_minutes": effort["total_minutes"],
        "effort_prefilled_minutes": effort["prefilled_minutes"],
        "effort_median_seconds": effort["median_seconds"],
    }


def export_stats(
    frame: pl.DataFrame,
    out_dir: Path,
    model_rank: str,
    prob_edges: list[float],
    include_not_assessable: bool = False,
    taxon_rank: str | None = None,
    include_uncertain: bool = True,
    info: Mapping[str, Any] | None = None,
    other_sessions: pl.DataFrame | None = None,
) -> list[Path]:
    """Write all statistics tables as CSV files and the export record 'summary.json' into 'out_dir'.

    The files of an earlier export are replaced. Fractions, kappa values and intervals are rounded to
    four decimals. 'summary.json' records the session, the export settings, the key figures and the
    meaning of every column of the written tables.

    Args:
        frame: Session table.
        out_dir: Output directory of the session's statistics (created if missing).
        model_rank: Deepest rank of the run.
        prob_edges: Probability bin edges of the reliability tables.
        include_not_assessable: Include not-assessable tracks (low image quality, different taxa).
        taxon_rank: Rank of the effort table and the confusion matrix (default: family, else 'model_rank').
        include_uncertain: Include 'inc.' determinations.
        info: Session details for 'summary.json' (e.g. validator, population, sampling settings).
        other_sessions: Output of compare_sessions(), written if not empty.

    Returns:
        Paths of the written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxon_rank = taxon_rank or ("family" if "family" in ranks_up_to(model_rank) else model_rank)
    tables: dict[str, pl.DataFrame] = {
        "progress_by_stratum": progress_by_stratum(frame),
        "verdict_distribution": verdict_table(frame),
        "verdict_distribution_by_stratum": verdict_table(frame, by_stratum=True),
        "agreement_by_rank": agreement_by_rank(frame, model_rank, include_not_assessable, include_uncertain),
        "linked_track_conflicts": linked_track_conflicts(frame),
        f"effort_by_taxon_{taxon_rank}": effort_by_taxon(frame, taxon_rank),
    }
    for rank in ranks_up_to(model_rank):
        tables[f"reliability_by_prob_bin_{rank}"] = reliability_by_prob_bin(
            frame, rank, prob_edges, include_not_assessable, include_uncertain
        )
        tables[f"per_taxon_{rank}"] = per_taxon_table(frame, rank, include_not_assessable, include_uncertain)
        tables[f"confusion_pairs_{rank}"] = confusion_pairs(
            frame, rank, include_not_assessable=include_not_assessable, include_uncertain=include_uncertain
        )
    labels, matrix = confusion_matrix(frame, taxon_rank, include_not_assessable, include_uncertain)
    matrix_name = f"confusion_matrix_{taxon_rank}"
    tables[matrix_name] = pl.DataFrame(
        {"model_taxon": labels, **{label: [row[j] for row in matrix] for j, label in enumerate(labels)}}
    ) if labels else pl.DataFrame(schema={"model_taxon": pl.String})
    if other_sessions is not None and not other_sessions.is_empty():
        tables["other_sessions"] = other_sessions

    columns = {
        f"{name}.csv": {"model_taxon": STATS_COLUMN_DESCRIPTIONS["model_taxon"], "<taxon>": _MATRIX_CELL_DESCRIPTION}
        if name == matrix_name else {c: column_description(name, c) for c in table.columns}
        for name, table in tables.items()
    }
    summary = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
        "session": dict(info or {}),
        "settings": {"include_not_assessable": include_not_assessable, "include_uncertain": include_uncertain,
                     "taxon_rank": taxon_rank, "prob_edges": list(prob_edges)},
        "figures": session_summary(frame),
        "columns": columns,
    }

    # Remove all files exported earlier at another rank or with other settings
    record = out_dir / "summary.json"
    for old in [*out_dir.glob("*.csv"), record]:
        old.unlink(missing_ok=True)
    written = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.with_columns(pl.col(pl.Float64).round(_EXPORT_DECIMALS)).write_csv(path)
        written.append(path)
    # Escape non-ASCII characters, e.g. validator names
    record.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    written.append(record)
    return written


def _not_assessable_expr() -> pl.Expr:
    """Return the filter for not-assessable rows (low image quality, different taxa in the crops or the track)."""
    return (pl.col("image_quality").fill_null("ok") == "low") | (
        pl.col("multiple_individuals").fill_null("").is_in(MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE)
    )


def _has_qualifier(which: str) -> pl.Expr:
    """Return the filter for rows whose 'val_qualifier' (e.g. 'inc. agg.') contains 'which'."""
    return pl.col("val_qualifier").fill_null("").str.split(" ").list.contains(which)


def _lowest_rank_qualifier(which: str, rank: str) -> pl.Expr:
    """Return the filter for rows with qualifier 'which' on 'rank' (their deepest determined rank)."""
    return _has_qualifier(which) & (pl.col("val_lowest_rank") == rank)


def _determination(rank: str, suffix: str = "") -> pl.Expr:
    """Return the validator's name at 'rank', an aggregate species with its ' agg.' mark (another label)."""
    name = pl.col(f"val_{rank}{suffix}")
    if rank != "species":
        return name
    aggregate = pl.col(f"val_qualifier{suffix}").fill_null("").str.split(" ").list.contains(ID_QUALIFIER_AGG)
    return pl.when(aggregate).then(name + f" {ID_QUALIFIER_AGG}").otherwise(name)


def _validated(frame: pl.DataFrame, include_not_assessable: bool) -> pl.DataFrame:
    """Return the validated rows, optionally without not-assessable tracks."""
    df = frame.filter(pl.col("validated"))
    if not include_not_assessable:
        df = df.filter(~_not_assessable_expr())
    return df


def _comparable_mask(rank: str, include_not_assessable: bool, include_uncertain: bool = True) -> pl.Expr:
    """Return the filter for validated rows where model and human both have a determinate taxon at 'rank'."""
    mask = pl.col("verdict").is_in(_COMPARABLE_VERDICTS)
    if not include_not_assessable:
        mask = mask & ~_not_assessable_expr()
    # Qualifiers apply to the deepest determined rank only, the ranks above stay comparable
    if not include_uncertain:
        mask = mask & ~_lowest_rank_qualifier(ID_QUALIFIER_INC, rank)
    # An aggregate ('agg.') does not determine its deepest rank
    mask = mask & ~_lowest_rank_qualifier(ID_QUALIFIER_AGG, rank)
    val_col, pred_col = pl.col(f"val_{rank}"), pl.col(f"pred_{rank}")
    return (mask & val_col.is_not_null() & (val_col != INDET_LABEL) & pred_col.is_not_null()).fill_null(False)


def _correct_expr(rank: str) -> pl.Expr:
    """Return the filter for rows whose determination at 'rank' equals the model's taxon."""
    return (pl.col(f"val_{rank}") == pl.col(f"pred_{rank}")).fill_null(False)


def _comparable(
    frame: pl.DataFrame, rank: str, include_not_assessable: bool, include_uncertain: bool = True
) -> pl.DataFrame:
    """Return the comparable validated rows at 'rank' (see _comparable_mask()) with '_correct'."""
    return (
        frame.filter(pl.col("validated") & _comparable_mask(rank, include_not_assessable, include_uncertain))
        .with_columns(_correct_expr(rank).alias("_correct"))
    )


def _weighted_ratio(
    rows: pl.DataFrame, numerator: pl.Expr, denominator: pl.Expr, z: float = 1.96
) -> tuple[float | None, float | None, float | None]:
    """Return the stratum-weighted ratio of two counts with a normal interval (combined ratio estimator).

    Every validated track stands for N_h / n_h tracks of its stratum (n_h: validated tracks of the
    stratum, also those outside the ratio), so strata count with their population size. The variance
    of the linearised residuals 'y - R x' uses a finite-population correction; a stratum with a single
    validated track adds none (the interval is then too narrow). Only strata with validated tracks count.

    Args:
        rows: Validated rows with 'stratum_id' and 'stratum_n_population'.
        numerator: Rows counted above the fraction line (e.g. correct).
        denominator: Rows counted below it (e.g. comparable).
        z: Normal quantile (1.96 = 95 %).

    Returns:
        '(estimate, ci_low, ci_high)'; the interval is None at zero variance, all None without denominator rows.
    """
    result = _weighted_ratios(rows, pl.lit("all"), numerator, denominator, z)
    if result.is_empty():
        return None, None, None
    _, estimate, low, high = result.row(0)
    return estimate, low, high


def _weighted_ratios(
    rows: pl.DataFrame, group: pl.Expr, numerator: pl.Expr, denominator: pl.Expr, z: float = 1.96
) -> pl.DataFrame:
    """Return _weighted_ratio() for every value of 'group' (e.g. the predicted taxon) in one pass.

    Rows of other groups count as sampled tracks of their stratum with zero counts, as if the
    ratio of one group were computed over all rows.

    Returns:
        '_group, estimate, ci_low, ci_high' for every group with denominator rows.
    """
    schema = {"_group": pl.String, "estimate": pl.Float64, "ci_low": pl.Float64, "ci_high": pl.Float64}
    df = rows.select(
        pl.col("stratum_id"),
        pl.col("stratum_n_population").cast(pl.Float64).alias("_N"),
        group.cast(pl.String).alias("_group"),
        numerator.fill_null(False).cast(pl.Float64).alias("_y"),
        denominator.fill_null(False).cast(pl.Float64).alias("_x"),
    ).drop_nulls(["stratum_id", "_N"]).with_columns(pl.len().over("stratum_id").alias("_n"))
    df = df.filter(pl.col("_group").is_not_null())
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    # Per group and stratum: sums of y and x; N_h * mean = N_h / n_h * sum
    cells = df.group_by("_group", "stratum_id").agg(
        pl.col("_N").first(), pl.col("_n").first(), pl.col("_y").sum().alias("_sy"), pl.col("_x").sum().alias("_sx"),
    )
    ratios = cells.group_by("_group").agg(
        (pl.col("_N") * pl.col("_sy") / pl.col("_n")).sum().alias("_ty"),
        (pl.col("_N") * pl.col("_sx") / pl.col("_n")).sum().alias("_tx"),
    ).filter(pl.col("_tx") > 0).with_columns((pl.col("_ty") / pl.col("_tx")).alias("estimate"))
    # Stratum variance of the residuals over all n_h rows: the m rows of the group, the others have e = 0
    residuals = df.join(ratios.select("_group", "estimate"), on="_group").with_columns(
        (pl.col("_y") - pl.col("estimate") * pl.col("_x")).alias("_e")
    ).with_columns((pl.col("_e").sum().over("_group", "stratum_id") / pl.col("_n")).alias("_mean"))
    spread = residuals.group_by("_group", "stratum_id").agg(
        pl.col("_N").first(), pl.col("_n").first(), pl.len().alias("_m"), pl.col("_mean").first(),
        ((pl.col("_e") - pl.col("_mean")) ** 2).sum().alias("_dev2"),
    ).filter(pl.col("_n") >= 2).with_columns(
        ((pl.col("_dev2") + (pl.col("_n") - pl.col("_m")) * pl.col("_mean") ** 2) / (pl.col("_n") - 1)).alias("_s2")
    )
    variance = spread.group_by("_group").agg(
        (pl.col("_N") ** 2 * (1 - pl.col("_n") / pl.col("_N")).clip(0) * pl.col("_s2") / pl.col("_n"))
        .sum().alias("_v")
    )
    result = ratios.join(variance, on="_group", how="left").with_columns(
        (pl.col("_v") / pl.col("_tx") ** 2).alias("_v")
    ).with_columns(
        # No interval at zero variance
        pl.when(pl.col("_v") > 0).then(z * pl.col("_v").sqrt()).alias("_half")
    )
    return result.select(
        "_group", "estimate",
        (pl.col("estimate") - pl.col("_half")).clip(lower_bound=0.0).alias("ci_low"),
        (pl.col("estimate") + pl.col("_half")).clip(upper_bound=1.0).alias("ci_high"),
    ).cast(pl.Schema(schema))


def _design_weight() -> pl.Expr:
    """Return the weight of a validated row: N_h / n_h tracks of its stratum (rows: the validated tracks)."""
    return pl.col("stratum_n_population").cast(pl.Float64) / pl.len().over("stratum_id")


def _weighted_columns_if_stratified(table: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Return 'table' without its stratum-weighted columns unless the sample is stratified."""
    if is_stratified(frame):
        return table
    return table.drop("accuracy_weighted", "accuracy_weighted_ci_low", "accuracy_weighted_ci_high")


def _agreement_row(level: str, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Return one row of inter_validator_agreement(): counts, share and kappa of paired labels."""
    n = len(pairs)
    k = sum(1 for a, b in pairs if a == b)
    kappa = cohens_kappa(pairs)
    lo, hi = cohens_kappa_interval(pairs)
    return {"level": level, "n_shared": n, "n_agree": k, "share_agree": k / n if n else None,
            "kappa": kappa, "kappa_ci_low": lo, "kappa_ci_high": hi}

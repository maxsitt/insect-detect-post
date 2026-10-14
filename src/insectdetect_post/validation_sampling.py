"""Track filters and stratified random subsampling for validation.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Strata combine the predicted taxon at one rank, bins of the weighted probability and the camera
setting. Sampling is seeded and keeps the population size of every stratum for weighted estimates,
the draw depends only on the seed and the track keys, not on library versions or row order.

Functions:
    filter_taxa(): Keep tracks of selected predicted taxa.
    filter_settings(): Keep tracks of selected camera settings.
    setting_counts(): Track counts per camera setting.
    setting_strata_available(): Whether a run has at least two setting groups.
    available_strata_ranks(): Ranks offered as taxon strata.
    parse_prob_edges(): Parse probability bin edges.
    prob_bin_labels(): Labels of the probability bins.
    assign_strata(): Add stratum columns to a track frame.
    stratum_summary(): Population size per stratum.
    small_strata(): Strata below a minimum size.
    sample_strata(): Seeded sample per stratum, shuffled into the validation queue.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

import polars as pl

from insectdetect_post.constants import SETTING_COLUMN, TRACK_KEY_COLUMNS
from insectdetect_post.validation_taxonomy import rank_index

# Ranks that make sense as strata (class is too coarse, species too fine)
_STRATUM_RANK_OPTIONS = ["order", "family", "genus"]

# Columns added by assign_strata()
STRATUM_COLUMNS = ["stratum_rank", "stratum_taxon", "stratum_prob_bin", "stratum_setting", "stratum_id"]

# Columns of a stratum in the per-stratum tables: its id, then what it was drawn by (the sort order)
STRATUM_GROUP_COLUMNS = ["stratum_id", "stratum_setting", "stratum_taxon", "stratum_prob_bin"]

# Stratum id used when no stratification is active
STRATUM_ALL = "all"

# Stratum value of tracks without a taxon / probability / setting
NONE_LABEL = "(none)"

# Separator of the hashed fields of a random key (unit separator, not part of any id)
_KEY_SEPARATOR = "\x1f"


def filter_taxa(tracks: pl.DataFrame, rank: str | None, names: list[str] | None) -> pl.DataFrame:
    """Keep tracks whose predicted taxon at 'rank' is one of 'names'."""
    if not rank or not names:
        return tracks
    col = f"bioclip_{rank}"
    if col not in tracks.columns:
        return tracks
    return tracks.filter(pl.col(col).is_in(names))


def filter_settings(tracks: pl.DataFrame, settings: list[str] | None) -> pl.DataFrame:
    """Keep tracks whose setting is one of 'settings' (NONE_LABEL: without one)."""
    if not settings or SETTING_COLUMN not in tracks.columns:
        return tracks
    return tracks.filter(pl.col(SETTING_COLUMN).fill_null(NONE_LABEL).is_in(settings))


def setting_counts(tracks: pl.DataFrame) -> pl.DataFrame:
    """Return the track counts per camera setting, tracks without one grouped as NONE_LABEL.

    Returns:
        Frame with 'setting, n_tracks', largest first. Empty without a 'setting' column.
    """
    if SETTING_COLUMN not in tracks.columns:
        return pl.DataFrame(schema={SETTING_COLUMN: pl.String, "n_tracks": pl.UInt32})
    return (
        tracks.group_by(pl.col(SETTING_COLUMN).fill_null(NONE_LABEL))
        .len().rename({"len": "n_tracks"})
        .sort(["n_tracks", SETTING_COLUMN], descending=[True, False])
    )


def setting_strata_available(tracks: pl.DataFrame) -> bool:
    """Return whether the tracks fall into at least two setting groups (see setting_counts())."""
    return setting_counts(tracks).height >= 2


def available_strata_ranks(active_ranks: list[str], model_rank: str) -> list[str]:
    """Return the strata rank options present in the data and not deeper than the model's rank."""
    return [r for r in _STRATUM_RANK_OPTIONS if r in active_ranks and rank_index(r) <= rank_index(model_rank)]


def parse_prob_edges(text: str) -> list[float]:
    """Parse comma/space separated probability bin edges.

    Args:
        text: e.g. '0, 0.2, 0.4, 0.6, 0.8, 1.0'.

    Raises:
        ValueError: If fewer than two edges, non-numeric, not from 0 to 1 or not strictly increasing.

    Returns:
        Strictly increasing list of edges from 0 to 1.
    """
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    try:
        edges = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Probability bin edges must be numbers: '{text}'") from e
    if len(edges) < 2:
        raise ValueError("At least two probability bin edges are required")
    if not all(0 <= e <= 1 for e in edges):  # also rejects NaN
        raise ValueError("Probability bin edges must be within [0, 1]")
    if any(b <= a for a, b in pairwise(edges)):
        raise ValueError("Probability bin edges must be strictly increasing")
    if edges[0] != 0 or edges[-1] != 1:  # the bins cover every probability
        raise ValueError("Probability bin edges must start at 0 and end at 1")
    return edges


def prob_bin_labels(edges: list[float]) -> list[str]:
    """Return the labels 'lo-hi' of the bins between the edges (lower edge included, the last bin includes 1)."""
    return [f"{lo:g}-{hi:g}" for lo, hi in pairwise(edges)]


def assign_strata(tracks: pl.DataFrame, rank: str | None, prob_edges: list[float] | None, *,
                  by_setting: bool = False) -> pl.DataFrame:
    """Add stratum columns to a track frame.

    Args:
        tracks: Track frame ('*_classified_final.csv' rows).
        rank: Rank of the taxon strata, or None.
        prob_edges: Probability bin edges, or None.
        by_setting: Also stratify by camera setting.

    Returns:
        Frame with STRATUM_COLUMNS added, a missing taxon, probability or setting becomes NONE_LABEL.
    """
    df = tracks.with_columns(pl.lit(rank).cast(pl.String).alias("stratum_rank"))
    if rank and f"bioclip_{rank}" in df.columns:
        df = df.with_columns(pl.col(f"bioclip_{rank}").fill_null(NONE_LABEL).alias("stratum_taxon"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("stratum_taxon"))

    if by_setting and SETTING_COLUMN in df.columns:
        df = df.with_columns(pl.col(SETTING_COLUMN).fill_null(NONE_LABEL).alias("stratum_setting"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("stratum_setting"))

    if prob_edges and "pred_prob_weighted" in df.columns:
        labels = prob_bin_labels(prob_edges)
        inner_breaks = prob_edges[1:-1]
        df = df.with_columns(
            pl.col("pred_prob_weighted").cast(pl.Float64, strict=False)
            .cut(breaks=inner_breaks, labels=labels, left_closed=True)
            .cast(pl.String).fill_null(NONE_LABEL).alias("stratum_prob_bin")
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("stratum_prob_bin"))

    # Only active dimensions appear in the stratum id
    stratum_id = pl.concat_str(
        [pl.col("stratum_setting"), pl.col("stratum_taxon"), pl.col("stratum_prob_bin")],
        separator="|", ignore_nulls=True,
    )
    return df.with_columns(
        pl.when(stratum_id == "").then(pl.lit(STRATUM_ALL)).otherwise(stratum_id).alias("stratum_id")
    )


def stratum_summary(tracks_with_strata: pl.DataFrame) -> pl.DataFrame:
    """Return the population size per stratum.

    Returns:
        Frame with 'stratum_id, stratum_setting, stratum_taxon, stratum_prob_bin, n_population'.
    """
    return (
        tracks_with_strata.group_by(STRATUM_GROUP_COLUMNS)
        .len().rename({"len": "n_population"})
        .sort(STRATUM_GROUP_COLUMNS[1:], nulls_last=True)
    )


def small_strata(summary: pl.DataFrame, min_per_stratum: int | None) -> list[str]:
    """Return the sorted ids of the strata in 'summary' with fewer than 'min_per_stratum' tracks."""
    if not min_per_stratum or min_per_stratum <= 1:
        return []
    return sorted(summary.filter(pl.col("n_population") < min_per_stratum)["stratum_id"].to_list())


def sample_strata(tracks_with_strata: pl.DataFrame, max_per_stratum: int | None, seed: int,
                  min_per_stratum: int | None = None) -> pl.DataFrame:
    """Draw a seeded random sample of up to 'max_per_stratum' tracks from every stratum.

    Every track gets a random key (see _random_keys()), a stratum keeps its tracks with the smallest
    keys ('sample'), a second key ('order') sets the queue. Strata are drawn independently, and the
    track key breaks ties.

    Args:
        tracks_with_strata: Output of assign_strata().
        max_per_stratum: Maximum tracks per stratum. None or 0 keeps all tracks.
        seed: Random seed for reproducible sampling.
        min_per_stratum: Strata with fewer tracks are excluded.

    Returns:
        Sampled tracks with 'stratum_n_population' and a shuffled 0-based 'sample_order'.
    """
    summary = stratum_summary(tracks_with_strata).select("stratum_id", "n_population")
    df = tracks_with_strata.join(summary, on="stratum_id", how="left").rename({"n_population": "stratum_n_population"})
    excluded = small_strata(summary, min_per_stratum)
    if excluded:
        df = df.filter(~pl.col("stratum_id").is_in(excluded))
    if max_per_stratum and max_per_stratum > 0:
        df = (
            df.with_columns(_random_keys(df, seed, "sample").alias("_draw"))
            .sort(["_draw", *TRACK_KEY_COLUMNS])
            .filter(pl.int_range(pl.len()).over("stratum_id") < max_per_stratum)
            .drop("_draw")
        )
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.UInt32).alias("sample_order"))
    return (
        df.with_columns(_random_keys(df, seed, "order").alias("_order"))
        .sort(["_order", *TRACK_KEY_COLUMNS])
        .drop("_order")
        .with_row_index("sample_order")
    )


def _random_keys(tracks: pl.DataFrame, seed: int, purpose: str) -> pl.Series:
    """Return one 64-bit random key per track, independent of library versions and row order.

    The key is the 8-byte BLAKE2b digest of 'seed, purpose, device_id, session_id, track_id'
    (UTF-8, joined by U+001F) read as a big-endian integer.
    """
    keys = []
    for key in tracks.select(TRACK_KEY_COLUMNS).iter_rows():
        text = _KEY_SEPARATOR.join((str(seed), purpose, *key))
        keys.append(int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest()))
    return pl.Series(keys, dtype=pl.UInt64)

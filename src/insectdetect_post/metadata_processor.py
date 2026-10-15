"""Aggregate classified metadata into per-track candidates and best-prediction results.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Aggregates image-level predictions into per-track candidates with weighted probabilities,
applies confidence/duration/probability filters, and writes both outputs to CSV.

BioCLIP predictions at species rank get the accepted GBIF species name and key of each Tree of
Life species from the species mapping ('gbif_species', 'gbif_species_key'), for joining further
details from the mapping. The per-image '_classified.csv' stays the raw model output.

Functions:
    process_metadata_classified(): Process classified metadata with filtering and size estimation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import polars as pl

from insectdetect_post.constants import RANK_ORDER, SETTING_COLUMN, TRACK_KEY_COLUMNS, OutputLayout
from insectdetect_post.dataset_harmonizer import parse_timestamp_column

# Create module-level logger
logger = logging.getLogger(__name__)

# Columns joined from the species mapping onto BioCLIP species predictions
GBIF_SPECIES_COLUMNS = ["gbif_species", "gbif_species_key"]


def _load_gbif_species(
    progress_callback: Callable[[int, int, str], None] | None = None
) -> pl.DataFrame | None:
    """Return the GBIF species name and key per Tree of Life species, or None if unavailable.

    Only species with a GBIF species key are returned, so both columns stay empty together for
    a Tree of Life species without one.

    Args:
        progress_callback: Optional callback(current, total, message) for a first-use download.

    Returns:
        DataFrame with 'bioclip_species' and GBIF_SPECIES_COLUMNS, or None if the species mapping
        cannot be loaded (warning logged).
    """
    # Imported here, so runs without BioCLIP do not load the GBIF dependencies
    from insectdetect_post.build_region_filter import load_species_mapping

    try:
        mapping = load_species_mapping(progress_callback)
    except (OSError, KeyError, ValueError) as e:
        logger.warning("Species mapping not available, outputs are written without GBIF columns: %s", e)
        return None
    return (
        mapping.filter(pl.col("gbif_species_key").is_not_null())
        .select("bioclip_species", *GBIF_SPECIES_COLUMNS)
    )


def process_metadata_classified(
    metadata_path: Path,
    output_dir: Path,
    min_conf_mean: float | None = None,
    min_dur_s: int | None = None,
    max_dur_s: int | None = None,
    min_prob_weighted: float | None = None,
    frame_width_mm: int | None = None,
    frame_height_mm: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    """Process classified metadata with filtering and size estimation.

    Supports both Ultralytics (top1/top2) and BioCLIP (taxonomic hierarchy)
    classification formats. BioCLIP results carry additional taxonomy columns
    while keeping the standard output shape, at species rank followed by
    GBIF_SPECIES_COLUMNS from the species mapping (left out if it is not
    available). All filtering is applied at track level after aggregation.
    If the mapping is used, its provenance record is copied into the run.

    Args:
        metadata_path: Path to classified metadata CSV.
        output_dir: Output directory for results.
        min_conf_mean: Minimum mean detection confidence per track ID.
        min_dur_s: Minimum track duration in seconds.
        max_dur_s: Maximum track duration in seconds.
        min_prob_weighted: Minimum weighted mean prediction probability per track ID.
        frame_width_mm: Physical frame width in millimeters.
        frame_height_mm: Physical frame height in millimeters.
        progress_callback: Optional callback(current, total, message) for progress updates.

    Returns:
        Dict with keys "candidates_path", "final_path", "out_dir", "tracks_total",
        "tracks_kept", "conf_tracks_removed", "min_dur_tracks_removed",
        "max_dur_tracks_removed", and "pred_tracks_removed".

    Raises:
        ValueError: If no classification columns found.
    """
    if progress_callback:
        progress_callback(0, 100, "Loading metadata...")

    df = pl.read_csv(metadata_path)

    # Auto-detect classification results format
    bioclip_rank_labels = RANK_ORDER[::-1]  # deepest first, the order of the output columns
    active_bioclip_ranks: list[str] = []
    classifier_type: Literal["ultralytics", "bioclip"]
    if "top1" in df.columns and "top1_prob" in df.columns:
        classifier_type = "ultralytics"
        bioclip_columns: list[str] = []
        logger.info("Detected Ultralytics classification format")
    elif "bioclip_top1_prob" in df.columns:
        classifier_type = "bioclip"
        active_bioclip_ranks = [r for r in bioclip_rank_labels if f"bioclip_top1_{r}" in df.columns]
        if not active_bioclip_ranks:
            raise ValueError("BioCLIP format detected but no taxonomic rank columns found")
        bioclip_columns = [f"bioclip_{r}" for r in active_bioclip_ranks]
        logger.info("Detected BioCLIP classification format (preserving taxonomic hierarchy)")
    else:
        raise ValueError(
            "No required classification columns found ('top1'/'top1_prob' or 'bioclip_top1_prob')."
        )

    if progress_callback:
        progress_callback(10, 100, "Preparing metadata...")

    # Convert timestamp to proper datetime if needed
    if df["timestamp"].dtype != pl.Datetime:
        original_count = len(df)
        parsed_ts, _ = parse_timestamp_column("timestamp")
        df = (
            df
            .with_columns(parsed_ts.alias("timestamp"))
            .filter(pl.col("timestamp").is_not_null())
        )
        parsed_count = len(df)
        invalid_count = original_count - parsed_count
        if invalid_count > 0:
            logger.warning(
                "Removed %d of %d rows with unparseable timestamps", invalid_count, original_count
            )

    if progress_callback:
        progress_callback(20, 100, "Calculating bbox sizes...")

    estimate_size = frame_width_mm is not None and frame_height_mm is not None
    if estimate_size:
        # Calculate physical bbox sizes based on frame dimensions in millimeters
        df = (
            df.with_columns([
                ((pl.col("x_max") - pl.col("x_min")) * frame_width_mm).round(4).alias("bbox_size_x"),
                ((pl.col("y_max") - pl.col("y_min")) * frame_height_mm).round(4).alias("bbox_size_y"),
            ])
            .with_columns([
                pl.max_horizontal("bbox_size_x", "bbox_size_y").alias("bbox_length_mm"),
                pl.min_horizontal("bbox_size_x", "bbox_size_y").alias("bbox_width_mm"),
            ])
            .drop(["bbox_size_x", "bbox_size_y"])
        )

    if progress_callback:
        progress_callback(30, 100, "Computing track aggregates...")

    # "setting" (deployment background) is constant per session, extract once here
    has_setting = SETTING_COLUMN in df.columns
    session_settings = (
        df.select(["device_id", "session_id", SETTING_COLUMN])
        .unique(subset=["device_id", "session_id"], keep="first")
        if has_setting else None
    )

    # Track-level aggregates
    agg_exprs = [
        pl.len().alias("track_imgs"),
        pl.min("timestamp").alias("start_time"),
        pl.max("timestamp").alias("end_time"),
        pl.mean("confidence").round(2).alias("det_conf_mean"),
    ]
    if estimate_size:
        agg_exprs += [
            pl.mean("bbox_length_mm").round(3).alias("bbox_length_mm_mean"),
            pl.mean("bbox_width_mm").round(3).alias("bbox_width_mm_mean"),
        ]

    agg_track = (
        df.group_by(TRACK_KEY_COLUMNS)
        .agg(agg_exprs)
        .with_columns([
            (pl.col("end_time") - pl.col("start_time"))
            .dt.total_seconds()
            .round(2)
            .alias("duration_s")
        ])
    )

    if progress_callback:
        progress_callback(40, 100, "Reshaping predictions for candidate aggregates...")

    # Reshape top1/top2 predictions into one row per (crop, candidate) vote
    slot_prefix = "bioclip_" if classifier_type == "bioclip" else ""
    slot_id_suffixes = active_bioclip_ranks if classifier_type == "bioclip" else [""]
    shared_columns = [c for c in df.columns
                      if not c.startswith(f"{slot_prefix}top1") and not c.startswith(f"{slot_prefix}top2")]
    output_columns = shared_columns + ["candidate", "candidate_prob", "vote_slot"] + bioclip_columns

    slot_frames = []
    for slot in (1, 2):
        id_columns = [f"{slot_prefix}top{slot}_{s}" if s else f"{slot_prefix}top{slot}"
                      for s in slot_id_suffixes]
        prob_column = f"{slot_prefix}top{slot}_prob"
        rename_map = {prob_column: "candidate_prob"}
        if classifier_type == "bioclip":
            rename_map.update({f"{slot_prefix}top{slot}_{r}": f"{slot_prefix}{r}" for r in slot_id_suffixes})
        candidate_source = bioclip_columns if classifier_type == "bioclip" else id_columns

        slot_frames.append(
            df.select(shared_columns + id_columns + [prob_column])
            .rename(rename_map)
            .filter(pl.col("candidate_prob").is_not_null())
            .with_columns(
                pl.coalesce([pl.col(c) for c in candidate_source]).alias("candidate"),
                pl.lit(slot).alias("vote_slot"),
            )
            .select(output_columns)
        )
    df_votes = pl.concat(slot_frames, how="vertical")

    if progress_callback:
        progress_callback(50, 100, "Computing candidate aggregates...")

    # Candidate aggregates with weighted probability
    agg_votes = [
        pl.len().alias("candidate_imgs"),
        pl.mean("candidate_prob").alias("candidate_prob_mean"),
        pl.col("candidate_prob").filter(pl.col("vote_slot") == 1).len().alias("top1_imgs"),
        pl.col("candidate_prob").filter(pl.col("vote_slot") == 1).mean().round(4).fill_null(0).alias("top1_prob_mean"),
        pl.col("candidate_prob").filter(pl.col("vote_slot") == 2).len().alias("top2_imgs"),
        pl.col("candidate_prob").filter(pl.col("vote_slot") == 2).mean().round(4).fill_null(0).alias("top2_prob_mean"),
    ]

    df_candidates = (
        df_votes.group_by([*TRACK_KEY_COLUMNS, "candidate"])
        .agg(agg_votes)
        .join(
            agg_track.select([*TRACK_KEY_COLUMNS, "track_imgs"]),
            on=TRACK_KEY_COLUMNS
        )
    )
    if session_settings is not None:
        df_candidates = df_candidates.join(session_settings, on=["device_id", "session_id"])
    df_candidates = (
        df_candidates
        .with_columns([
            (pl.col("candidate_prob_mean") * (pl.col("candidate_imgs") / pl.col("track_imgs")))
            .alias("candidate_prob_weighted")
        ])
        .with_columns(
            pl.col("candidate_prob_mean").round(4),
            pl.col("candidate_prob_weighted").round(4),
        )
    )

    if classifier_type == "bioclip":
        # Re-attach taxonomic hierarchy columns
        taxonomy_map = (
            df_votes.select([*TRACK_KEY_COLUMNS, "candidate", *bioclip_columns])
            .unique(subset=[*TRACK_KEY_COLUMNS, "candidate"], keep="first")
        )
        df_candidates = df_candidates.join(
            taxonomy_map,
            on=[*TRACK_KEY_COLUMNS, "candidate"],
            how="left"
        )

    # Sort candidates per track by weighted and mean probabilities
    df_candidates = df_candidates.sort(
        by=[*TRACK_KEY_COLUMNS, "candidate_prob_weighted", "candidate_prob_mean"],
        descending=[*[False] * len(TRACK_KEY_COLUMNS), True, True]
    )

    # Drop negligible candidates, but always keep each track's own best row
    df_candidates = df_candidates.filter(
        (pl.col("candidate_prob_weighted") > 0)
        | (pl.int_range(pl.len()).over(TRACK_KEY_COLUMNS) == 0)
    )

    if progress_callback:
        progress_callback(70, 100, "Selecting best candidate...")

    # Get best candidate per track
    df_final = (
        df_candidates
        .group_by(TRACK_KEY_COLUMNS)
        .first()
        .join(agg_track, on=TRACK_KEY_COLUMNS)
    )

    if progress_callback:
        progress_callback(80, 100, "Filtering tracks...")

    tracks_total = len(df_final)

    # Apply mean detection confidence filtering
    conf_tracks_removed = 0
    if min_conf_mean is not None and min_conf_mean > 0:
        tracks_before = len(df_final)
        df_final = df_final.filter(pl.col("det_conf_mean") >= min_conf_mean)
        conf_tracks_removed = tracks_before - len(df_final)

    # Apply duration filtering
    min_dur_tracks_removed = 0
    if min_dur_s is not None and min_dur_s > 0:
        tracks_before = len(df_final)
        df_final = df_final.filter(pl.col("duration_s") >= min_dur_s)
        min_dur_tracks_removed = tracks_before - len(df_final)

    max_dur_tracks_removed = 0
    if max_dur_s is not None:
        tracks_before = len(df_final)
        df_final = df_final.filter(pl.col("duration_s") <= max_dur_s)
        max_dur_tracks_removed = tracks_before - len(df_final)

    # Apply weighted prediction probability filtering
    pred_tracks_removed = 0
    if min_prob_weighted is not None and min_prob_weighted > 0:
        tracks_before = len(df_final)
        df_final = df_final.filter(pl.col("candidate_prob_weighted") >= min_prob_weighted)
        pred_tracks_removed = tracks_before - len(df_final)

    if progress_callback:
        progress_callback(90, 100, "Finalizing results...")

    # Sort the final result
    df_final = df_final.sort(TRACK_KEY_COLUMNS)

    # Final column selection
    main_columns = ["device_id", "session_id"]
    if has_setting:
        main_columns.append(SETTING_COLUMN)
    main_columns += [
        "track_id", "track_imgs", "candidate_imgs",
        "candidate", "candidate_prob_weighted", "candidate_prob_mean"
    ]

    track_level_columns = [
        "start_time", "end_time", "duration_s",
        "det_conf_mean",
    ]
    if estimate_size:
        track_level_columns += ["bbox_length_mm_mean", "bbox_width_mm_mean"]

    candidates_columns = main_columns.copy()
    candidates_columns += ["top1_imgs", "top1_prob_mean", "top2_imgs", "top2_prob_mean"]

    final_columns = main_columns + track_level_columns

    # BioCLIP taxonomy columns go at the very end of both outputs
    if classifier_type == "bioclip":
        candidates_columns += bioclip_columns
        final_columns += bioclip_columns

    # Working columns are named "candidate_*" throughout; _candidates.csv uses them as-is,
    # _final.csv renames them to "pred_*" since it has exactly one row per track
    df_candidates = df_candidates.select(candidates_columns)
    df_final = (
        df_final
        .with_columns(pl.col("candidate_prob_mean").round(2), pl.col("candidate_prob_weighted").round(2))
        .select(final_columns)
        .rename({
            "candidate_imgs": "pred_imgs",
            "candidate": "pred",
            "candidate_prob_mean": "pred_prob_mean",
            "candidate_prob_weighted": "pred_prob_weighted",
        })
    )

    # Accepted GBIF species of the predicted Tree of Life species, after the taxonomy columns
    gbif_species = None
    if "bioclip_species" in bioclip_columns:
        def mapping_progress(_pct: int, _total: int, message: str) -> None:
            if progress_callback:
                progress_callback(92, 100, message)

        gbif_species = _load_gbif_species(mapping_progress)
    if gbif_species is not None:
        df_candidates = df_candidates.join(gbif_species, on="bioclip_species", how="left", maintain_order="left")
        df_final = df_final.join(gbif_species, on="bioclip_species", how="left", maintain_order="left")
        logger.info("GBIF species key for %d of %d tracks",
                    df_final["gbif_species_key"].is_not_null().sum(), df_final.height)

    if progress_callback:
        progress_callback(95, 100, "Saving results...")

    layout = OutputLayout(output_dir)
    candidates_path = layout.metadata_dir / f"{metadata_path.stem}_candidates.csv"
    final_path = layout.metadata_dir / f"{metadata_path.stem}_final.csv"
    df_candidates.write_csv(candidates_path)
    df_final.write_csv(final_path)

    if gbif_species is not None:
        from insectdetect_post.build_region_filter import copy_mapping_provenance

        try:
            copy_mapping_provenance(layout.filters_dir)
        except OSError as e:
            logger.warning("Could not copy the species mapping provenance to '%s': %s", layout.filters_dir, e)

    if progress_callback:
        progress_callback(100, 100, "Metadata processing complete")

    return {
        "candidates_path": candidates_path,
        "final_path": final_path,
        "out_dir": layout.metadata_dir,
        "tracks_total": tracks_total,
        "tracks_kept": len(df_final),
        "conf_tracks_removed": conf_tracks_removed,
        "min_dur_tracks_removed": min_dur_tracks_removed,
        "max_dur_tracks_removed": max_dur_tracks_removed,
        "pred_tracks_removed": pred_tracks_removed,
    }

"""Classify images using the BioCLIP 2 TreeOfLife model.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Runs pybioclip's TreeOfLifeClassifier over pre-scanned images in memory-bounded chunks,
optionally restricted by taxa and countries, and writes results to the metadata CSV.

Functions:
    classify_imgs_bioclip(): Classify images in chunks using the BioCLIP 2 model and write results to CSV.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
import psutil
from bioclip import Rank, TreeOfLifeClassifier

from insectdetect_post.build_region_filter import load_region_species
from insectdetect_post.classifier_utils import (
    format_time,
    parse_crop_name,
    save_classification_results,
    validate_metadata,
)
from insectdetect_post.constants import RANK_ORDER

# Create module-level logger
logger = logging.getLogger(__name__)

# BioCLIP Rank and taxon name each filter_arthropods.taxa value maps to for create_taxa_filter()
_TAXA_RANKS: dict[str, tuple[Rank, str]] = {
    "all": (Rank.PHYLUM, "Arthropoda"),
    "Insecta": (Rank.CLASS, "Insecta"),
    "Arachnida": (Rank.CLASS, "Arachnida"),
    "Diplopoda": (Rank.CLASS, "Diplopoda"),
    "Chilopoda": (Rank.CLASS, "Chilopoda"),
    "Isopoda": (Rank.ORDER, "Isopoda"),
}


def _build_taxa_mask(classifier: TreeOfLifeClassifier, taxa: Sequence[str]) -> list[bool]:
    """Return a species mask covering every selected taxon, across taxonomic ranks.

    Values are grouped by rank so create_taxa_filter() is called once per rank, and the resulting
    masks are OR-ed: a species is kept if it matches any selection.

    Args:
        classifier: Loaded TreeOfLifeClassifier whose label data the mask is built against.
        taxa: Selected filter_arthropods.taxa values, as keys of _TAXA_RANKS.

    Returns:
        Boolean mask over the classifier's species labels.
    """
    by_rank: dict[Rank, list[str]] = defaultdict(list)
    for value in taxa:
        rank, name = _TAXA_RANKS[value]
        by_rank[rank].append(name)

    masks = [classifier.create_taxa_filter(rank, names) for rank, names in by_rank.items()]
    if len(masks) == 1:
        return masks[0]
    return [any(entries) for entries in zip(*masks)]


def classify_imgs_bioclip(
    crop_file_list: list[Path],
    metadata_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    rank: str = "species",
    filter_arthropods_enabled: bool = False,
    filter_taxa: Sequence[str] = ("all",),
    filter_countries: Sequence[str] = ("all",),
    device: str = "cpu",
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Path:
    """Classify images in chunks using the BioCLIP 2 model and write results to CSV.

    Returns the top 2 taxonomic predictions per image, up to the requested rank.
    For ranks above species, species-level probabilities are summed to the target rank.
    An image may have fewer than 2 predictions if candidates don't meet min_prob.

    Args:
        crop_file_list: Pre-scanned list of crop file paths.
        metadata_path: Path to metadata CSV.
        output_dir: Output directory for results.
        batch_size: Number of images to process per batch.
        rank: Taxonomic rank to predict (species-level probabilities are summed to this rank).
        filter_arthropods_enabled: If True, restrict predictions by taxa and optionally countries.
        filter_taxa: Taxa to restrict predictions to, as keys of _TAXA_RANKS. A species is kept
            if it matches any of them; 'all' covers the whole phylum Arthropoda.
        filter_countries: ISO 3166-1 alpha-2 country codes. A species is kept if it occurs in
            any of them; 'all' means no region restriction.
        device: Device to run model on ("cpu" or "cuda").
        progress_callback: Optional progress callback.

    Returns:
        Path to classified metadata CSV.
    """
    if not crop_file_list:
        raise ValueError("No crop files provided for classification")

    start_time = time.time()
    bioclip_rank = Rank[rank.upper()]
    active_ranks = RANK_ORDER[:RANK_ORDER.index(rank) + 1]

    # Validate metadata
    validate_metadata(metadata_path, progress_callback)

    # Prepare crop files
    if progress_callback:
        progress_callback(1, 100, "Preparing crop files...")

    crop_files = sorted(crop_file_list)
    total = len(crop_files)
    logger.debug("Using %d pre-scanned crop files", total)

    # Convert to string paths for BioCLIP API
    image_paths_str = [str(p) for p in crop_files]

    logger.info("Found %d crops to classify", total)
    logger.info("Batch size: %d, Device: %s", batch_size, device)

    # Load model
    if progress_callback:
        progress_callback(3, 100, "Loading BioCLIP model...")

    classifier = TreeOfLifeClassifier(device=device)
    logger.info("Loaded BioCLIP TreeOfLife classifier")

    # Progress percentage at which classification starts (raised if a region filter is built)
    cls_start_pct = 5

    if filter_arthropods_enabled:
        # Create taxa filter mask for the requested taxa
        taxa_label = ", ".join(filter_taxa)
        countries_label = ", ".join(filter_countries)
        taxon_mask = _build_taxa_mask(classifier, filter_taxa)

        if "all" not in filter_countries:
            def region_filter_progress(pct: int, _total: int, message: str) -> None:
                """Wrapper callback that maps region filter progress to global progress."""
                nonlocal cls_start_pct
                cls_start_pct = 10
                if progress_callback:
                    # Scale progress from 3% to 10%
                    progress_callback(3 + int(pct / 100 * 7), 100, message)

            # Merge the countries into a single species list, so the mask is built in one pass
            region_species = load_region_species(
                filter_countries, progress_callback=region_filter_progress
            )
            region_mask = classifier.create_taxa_filter(Rank.SPECIES, region_species)
            combined_mask = [t and r for t, r in zip(taxon_mask, region_mask)]
        else:
            combined_mask = taxon_mask

        if not any(combined_mask):
            raise ValueError(
                f"No species match taxa='{taxa_label}' and countries='{countries_label}' -- "
                "combined filter would exclude all predictions."
            )

        classifier.apply_filter(combined_mask)
        logger.info("Applied taxa='%s', countries='%s', filter: %d/%d species kept",
                    taxa_label, countries_label, sum(combined_mask), len(combined_mask))

    # Compute dynamic chunk size based on available RAM
    n_species = classifier.get_txt_embeddings().shape[1]
    bytes_per_image = n_species * 4
    available_ram = psutil.virtual_memory().available
    chunk_size = max(50, min(int(available_ram * 0.7 / bytes_per_image), total))
    logger.info("Available RAM: %.1f GB | species: %d | chunk size: %d",
                available_ram / 1024**3, n_species, chunk_size)

    if progress_callback:
        progress_callback(cls_start_pct, 100, f"Classifying {total} crops (chunk size: {chunk_size})...")

    # Pre-allocate results
    timestamps: list[str | None] = [None] * total
    track_ids: list[int | None] = [None] * total
    rank_predictions: dict[int, dict[str, list[str | None]]] = {
        slot: {r: [None] * total for r in active_ranks} for slot in (1, 2)
    }
    bioclip_probs: dict[int, list[float | None]] = {slot: [None] * total for slot in (1, 2)}

    # Classify in memory-bounded chunks; each chunk's probs dict is freed before the next
    pred_index = 0
    total_chunks = (total + chunk_size - 1) // chunk_size
    cls_start_time = time.time()

    for chunk_start in range(0, total, chunk_size):
        chunk = image_paths_str[chunk_start:chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        logger.debug("Chunk %d/%d: images %d-%d (%d images)",
                     chunk_num, total_chunks,
                     chunk_start + 1, chunk_start + len(chunk), len(chunk))
        chunk_start_time = time.time()

        def bioclip_progress_callback(current: int, _chunk_total: int,
                                      _offset: int = chunk_start):
            """Wrapper callback that maps chunk progress to global progress."""
            if progress_callback:
                global_current = _offset + current
                # Scale progress from cls_start_pct to 95%
                percentage = cls_start_pct + int((global_current / max(total, 1)) * (95 - cls_start_pct))

                elapsed = time.time() - cls_start_time
                rate = global_current / max(elapsed, 0.1)
                remaining = total - global_current
                eta = remaining / rate if rate > 0 and remaining > 0 else 0

                msg = f"Classifying: {global_current}/{total} | {rate:.1f}/s"
                if remaining > 0:
                    msg += f" | ETA: {format_time(eta)}"

                progress_callback(percentage, 100, msg)

        predictions = classifier.predict(
            images=chunk,
            rank=bioclip_rank,
            min_prob=0.001,
            k=2,  # return top 2 predictions
            batch_size=batch_size,
            callback=bioclip_progress_callback
        )

        # Group predictions by file_name
        preds_by_file: dict[str, list[dict]] = defaultdict(list)
        for pred in predictions:
            preds_by_file[pred["file_name"]].append(pred)

        for local_idx, image_path in enumerate(chunk):
            ts_iso, track_id = parse_crop_name(crop_files[chunk_start + local_idx].name)
            timestamps[pred_index] = ts_iso
            track_ids[pred_index] = track_id
            for slot, pred in enumerate(preds_by_file.get(image_path, [])[:2], start=1):
                for r in active_ranks:
                    rank_predictions[slot][r][pred_index] = pred.get(r)
                bioclip_probs[slot][pred_index] = pred.get('score')
            pred_index += 1

        logger.debug("Chunk %d/%d done in %.1fs", chunk_num, total_chunks,
                     time.time() - chunk_start_time)

    if progress_callback:
        progress_callback(96, 100, "Processing results...")

    # Create results DataFrame
    lowest_rank = active_ranks[-1]
    higher_ranks = list(reversed(active_ranks[:-1]))
    df_cls = pl.DataFrame({
        "timestamp": timestamps,
        "track_id": track_ids,
        f"bioclip_top1_{lowest_rank}": rank_predictions[1][lowest_rank],
        "bioclip_top1_prob": bioclip_probs[1],
        f"bioclip_top2_{lowest_rank}": rank_predictions[2][lowest_rank],
        "bioclip_top2_prob": bioclip_probs[2],
        **{f"bioclip_top1_{r}": rank_predictions[1][r] for r in higher_ranks},
        **{f"bioclip_top2_{r}": rank_predictions[2][r] for r in higher_ranks},
    }).with_columns(
        pl.col("bioclip_top1_prob").round(3),
        pl.col("bioclip_top2_prob").round(3),
    )

    # Save results
    out_path = save_classification_results(df_cls, metadata_path, output_dir, progress_callback)

    # Format completion message
    elapsed = time.time() - start_time
    speed = total / max(elapsed, 0.001)

    if progress_callback:
        progress_callback(100, 100, f"Done: {total} crops in {format_time(elapsed)} ({speed:.1f}/s)")

    logger.info("Classification complete: %.1f crops/s", speed)

    return out_path

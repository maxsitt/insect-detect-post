"""Classify images using a classification model supported by the ultralytics library.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Runs Ultralytics YOLO classification via glob-based streaming inference
over pre-scanned images, and writes results to the metadata CSV.

Functions:
    classify_imgs_ultralytics(): Classify images using streaming inference and write results to CSV.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl
from ultralytics import YOLO
from ultralytics.engine.results import Probs, Results

from insectdetect_post.classifier_utils import (
    format_time,
    parse_crop_name,
    save_classification_results,
    validate_metadata,
)

# Create module-level logger
logger = logging.getLogger(__name__)


def classify_imgs_ultralytics(
    crop_file_list: list[Path],
    crop_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    model_path: Path,
    batch_size: int = 8,
    device: str = "cpu",
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Path:
    """Classify images using streaming inference and write results to CSV.

    Returns the top 2 predictions per image.
    Uses glob pattern which properly supports batch_size > 1 and streaming.

    Args:
        crop_file_list: Pre-scanned list of crop file paths.
        crop_dir: Root directory containing images (used for glob pattern).
        metadata_path: Path to metadata CSV.
        output_dir: Output directory for results.
        model_path: Path to ONNX/PT model.
        batch_size: Number of images to process per batch.
        device: Device to run model on ("cpu" or "cuda").
        progress_callback: Optional progress callback.

    Returns:
        Path to classified metadata CSV.
    """
    if not crop_file_list:
        raise ValueError("No crop files provided for classification")
    if not crop_dir.exists() or not crop_dir.is_dir():
        raise ValueError(f"Invalid crop directory: {crop_dir}")

    start_time = time.time()

    # Validate metadata
    validate_metadata(metadata_path, progress_callback)

    # Prepare crop files
    if progress_callback:
        progress_callback(1, 100, "Preparing crop files...")

    crop_files = sorted(crop_file_list)
    total = len(crop_files)
    logger.debug("Using %d pre-scanned crop files", total)

    # Build filename -> index map for O(1) result lookup
    filename_to_idx = {p.name: i for i, p in enumerate(crop_files)}

    logger.info("Found %d crops to classify", total)
    logger.info("Batch size: %d, Device: %s", batch_size, device)

    # Pre-allocate results
    timestamps: list[str | None] = [None] * total
    track_ids: list[int | None] = [None] * total
    top1_labels: list[str | None] = [None] * total
    top1_probs: list[float | None] = [None] * total
    top2_labels: list[str | None] = [None] * total
    top2_probs: list[float | None] = [None] * total

    # Load model
    if progress_callback:
        progress_callback(3, 100, "Loading model...")

    model = YOLO(str(model_path), task="classify")
    logger.info("Loaded %s", model_path.name)

    # Extract input image size from model metadata
    imgsz: int = model.overrides.get("imgsz", 224)
    logger.info("Model input size: %d", imgsz)

    if progress_callback:
        progress_callback(5, 100, f"Classifying {total} crops...")

    # Streaming inference
    processed = 0
    last_pct = 5
    cls_start_time = time.time()

    for result in model.predict(
        source=str(crop_dir / "**/*.jpg"),
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        stream=True,
        verbose=False
    ):
        result = cast(Results, result)
        crop_name = Path(result.path).name
        idx = filename_to_idx.get(crop_name)
        if idx is None:
            continue

        ts_iso, track_id = parse_crop_name(crop_name)
        timestamps[idx] = ts_iso
        track_ids[idx] = track_id
        probs = cast(Probs, result.probs)
        top1_labels[idx] = result.names[probs.top5[0]]
        top1_probs[idx] = float(probs.top5conf[0])
        top2_labels[idx] = result.names[probs.top5[1]]
        top2_probs[idx] = float(probs.top5conf[1])
        processed += 1

        # Progress update (every 1%)
        current_pct = 5 + int((processed / total) * 90)
        if (current_pct > last_pct or processed == total) and progress_callback:
            elapsed = time.time() - cls_start_time
            rate = processed / max(elapsed, 0.1)
            remaining = total - processed
            eta = remaining / rate if rate > 0 else 0

            msg = f"Classifying: {processed}/{total} | {rate:.1f}/s"
            if remaining > 0:
                msg += f" | ETA: {format_time(eta)}"

            progress_callback(current_pct, 100, msg)
            last_pct = current_pct

    # Create results DataFrame
    if progress_callback:
        progress_callback(96, 100, "Processing results...")

    df_cls = pl.DataFrame({
        "timestamp": timestamps,
        "track_id": track_ids,
        "top1": top1_labels,
        "top1_prob": top1_probs,
        "top2": top2_labels,
        "top2_prob": top2_probs
    }).with_columns(
        pl.col("top1_prob").round(3),
        pl.col("top2_prob").round(3),
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

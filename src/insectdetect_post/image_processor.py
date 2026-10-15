"""Process full frames to generate crops and/or annotated overlays for each detection frame.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Generates square or original-aspect crops and/or annotated overlay images for each
detection frame referenced in a metadata CSV, using a thread pool for parallel I/O.

Classes:
    PostConfig: Configuration for image processing (cropping and/or overlays).

Functions:
    make_bbox_square(): Expand bounding box to square while staying within image bounds.
    process_images(): Process images (cropping and/or overlays) with progress reporting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import polars as pl

from insectdetect_post.classifier_utils import format_time
from insectdetect_post.constants import MAX_WORKERS, OutputLayout
from insectdetect_post.dataset_context import DatasetContext
from insectdetect_post.dataset_harmonizer import FILENAME_TIMESTAMP_FORMAT_PY
from insectdetect_post.exceptions import PipelineCancelled

# Create module-level logger
logger = logging.getLogger(__name__)


@dataclass
class PostConfig:
    """Configuration for image processing (cropping and/or overlays)."""
    crop_enabled: bool
    crop_method: Literal["square", "original"]
    overlay_enabled: bool
    output_dir: Path
    img_ext: str = ".jpg"


def make_bbox_square(
    img_w: int,
    img_h: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int
) -> tuple[int, int, int, int]:
    """Expand bounding box to square while staying within image bounds.

    Args:
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        x0: Bounding box left edge.
        y0: Bounding box top edge.
        x1: Bounding box right edge.
        y1: Bounding box bottom edge.

    Returns:
        Square bounding box as (x0, y0, x1, y1), clamped to the image bounds.
    """
    bbox_w = x1 - x0
    bbox_h = y1 - y0

    if bbox_w == bbox_h:
        return x0, y0, x1, y1

    if bbox_w < bbox_h:
        # Expand bbox width to match bbox height
        expansion_per_side = (bbox_h - bbox_w) // 2
        x0_new = max(0, x0 - expansion_per_side)
        x1_new = min(img_w, x1 + (bbox_h - (x1 - x0_new)))
        if x1_new - x0_new < bbox_h:
            x0_new = max(0, x1_new - bbox_h)
        return x0_new, y0, x0_new + bbox_h, y1
    else:
        # Expand bbox height to match bbox width
        expansion_per_side = (bbox_w - bbox_h) // 2
        y0_new = max(0, y0 - expansion_per_side)
        y1_new = min(img_h, y1 + (bbox_w - (y1 - y0_new)))
        if y1_new - y0_new < bbox_w:
            y0_new = max(0, y1_new - bbox_w)
        return x0, y0_new, x1, y0_new + bbox_w


def _timestamp_to_name(ts: str) -> str:
    """Convert ISO timestamp to filename-safe format.

    Args:
        ts: ISO format timestamp (e.g., "2025-07-18T13:06:48.757974")

    Returns:
        Filename-safe timestamp (e.g., "2025-07-18_13-06-48-757974")
    """
    dt = datetime.fromisoformat(ts)
    return dt.strftime(FILENAME_TIMESTAMP_FORMAT_PY)


def _put_outlined_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: cv2.FontFace,
    size: int,
    font_weight: int,
    outline_radius: int
) -> None:
    """Draw text with a black outline and white fill, in-place.

    Args:
        img: Image array to draw on.
        text: Text to draw.
        org: Bottom-left origin of the text, in pixels.
        font: Font face to use.
        size: Font size in pixels.
        font_weight: Font weight.
        outline_radius: Outline thickness in pixels.
    """
    for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
        offset_org = (org[0] + dx * outline_radius, org[1] + dy * outline_radius)
        cv2.putText(img, text, offset_org, (0, 0, 0), font, size, font_weight)
    cv2.putText(img, text, org, (255, 255, 255), font, size, font_weight)


def _draw_overlays(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str],
    confs: Sequence[float] | None,
    track_ids: Sequence[int]
) -> None:
    """Draw bounding boxes and text overlays in-place with adaptive positioning.

    Args:
        img: Image array to draw on.
        boxes: Array of bounding boxes [x0, y0, x1, y1].
        labels: Detection labels.
        confs: Optional confidence scores.
        track_ids: Track IDs for each detection.
    """
    img_h, img_w = img.shape[:2]
    big = img_w > 2000

    # Adaptive text size based on image resolution
    font = cv2.FontFace("sans")
    font_size = 40 if big else 30
    font_weight = 500 if big else 400
    outline_radius = 2
    box_thickness = 3 if big else 2
    above_gap = 6 if big else 4

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        label = str(labels[i])
        track_id = str(track_ids[i])

        if confs is not None and confs[i] is not None:
            label_text = f"{label} {confs[i]:.2f}"
        else:
            label_text = label
        id_text = f"ID: {track_id}"

        # Calculate text heights (ascent) for positioning
        label_rect = cv2.getTextSize((img_w, img_h), label_text, (0, 0), font, font_size, font_weight)
        id_rect = cv2.getTextSize((img_w, img_h), id_text, (0, 0), font, font_size, font_weight)
        label_h = -label_rect[1]
        id_h = -id_rect[1]

        # Determine position: below box if space available, otherwise above
        if y1 + label_h + id_h < img_h * 0.95:
            label_pos = (x0, y1 + label_h)
            id_pos = (x0, y1 + label_h + id_h)
        else:
            label_pos = (x0, y0 - id_h - above_gap)
            id_pos = (x0, y0 - above_gap)

        # Draw label (with confidence) and track ID, both with black outline + white fill
        _put_outlined_text(img, label_text, label_pos, font, font_size, font_weight, outline_radius)
        _put_outlined_text(img, id_text, id_pos, font, font_size, font_weight, outline_radius)

        # Draw bounding box (red)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), box_thickness)


def _batch_write_crops(crop_batch: list[tuple[Path, np.ndarray]]) -> list[bool]:
    """Write multiple crops in batch to amortize per-call I/O overhead.

    Args:
        crop_batch: List of (output_path, crop_image) tuples.

    Returns:
        Per-item write success, aligned by index with crop_batch.
    """
    results = []
    for out_path, crop in crop_batch:
        try:
            ok = bool(cv2.imwrite(str(out_path), crop))
            if not ok:
                logger.error("Failed to write crop %s: imwrite returned False", out_path)
            results.append(ok)
        except Exception:
            logger.exception("Failed to write crop %s", out_path)
            results.append(False)
    return results


def process_images(
    metadata_path: Path,
    post_config: PostConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    context: DatasetContext | None = None
) -> dict[str, Any]:
    """Process images (cropping and/or overlays) with progress reporting.

    Args:
        metadata_path: Path to metadata CSV.
        post_config: Configuration for image processing.
        progress_callback: Optional progress callback.
        context: Dataset context for frame path resolution.

    Returns:
        Dict with keys "frames_processed", "crops_created", "crop_file_list" (paths of
        successfully written crops), "frames_skipped_missing" (no detection frame found),
        "frames_skipped_corrupt" (frame failed to load), and "frames_failed" (unexpected
        per-frame errors).

    Raises:
        ValueError: If context is missing, output_dir is not set,
                    or the metadata CSV is missing required columns.
    """
    start_time = time.time()

    if not context:
        raise ValueError("DatasetContext is required for frame path resolution")

    if progress_callback:
        progress_callback(0, 100, "Loading metadata...")

    df = pl.read_csv(metadata_path)

    req = {"timestamp", "label", "track_id", "x_min", "y_min", "x_max", "y_max"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    if post_config.overlay_enabled and "confidence" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("confidence"))

    if progress_callback:
        progress_callback(5, 100, "Preparing output directories...")

    # Setup output directories
    if not post_config.output_dir:
        raise ValueError("output_dir is required for image processing")

    layout = OutputLayout(post_config.output_dir)
    overlays_dir = layout.overlays_dir
    crops_dir = layout.crops_dir

    if post_config.overlay_enabled:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    if post_config.crop_enabled:
        # Create label subdirectories
        for lbl in df.get_column("label").unique().to_list():
            layout.crop_label_dir(lbl).mkdir(parents=True, exist_ok=True)

    # Group by timestamp for per-frame processing
    groups = list(df.group_by("timestamp", maintain_order=True))
    total_groups = len(groups)

    if progress_callback:
        progress_callback(10, 100, f"Processing {total_groups} frames...")

    logger.info("Using %d worker threads for processing", MAX_WORKERS)

    stats = {
        "frames_processed": 0,
        "crops_created": 0,
        "frames_skipped_missing": 0,
        "frames_skipped_corrupt": 0,
        "frames_failed": 0,
    }

    # Cancellation flag shared across workers
    cancel_event = threading.Event()

    def handle_group(group_data: tuple[pl.DataFrame, int]) -> dict[str, Any]:
        """Process one timestamp with all its detections."""
        sub, group_idx = group_data
        cam_id = sub.item(0, "device_id")
        timestamp = sub.item(0, "timestamp")
        # "filename" can exist but be null here (diagonal_relaxed merge of mixed-format sources)
        filename = sub.item(0, "filename") if "filename" in sub.columns else None
        ts_name = Path(filename).stem if filename is not None else _timestamp_to_name(timestamp)

        # Check for cancellation before any per-group work
        if cancel_event.is_set():
            raise PipelineCancelled()

        # Look up by (device_id, timestamp)
        img_path = context.detection_frame_map.get((cam_id, timestamp))

        if not img_path or not img_path.exists():
            # No detection frame available - skip this group
            logger.warning("No detection frame found for %s (device %s), skipping", ts_name, cam_id)
            return {
                "group_idx": group_idx,
                "ok": False,
                "skip_reason": "missing",
                "stats": {"frames_processed": 0, "crops_created": 0}
            }

        # Check for cancellation before the image load (expensive I/O)
        if cancel_event.is_set():
            raise PipelineCancelled()

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to read image %s (corrupt or unreadable), skipping", img_path)
            return {
                "group_idx": group_idx,
                "ok": False,
                "skip_reason": "corrupt",
                "stats": {"frames_processed": 0, "crops_created": 0}
            }

        # Get image dimensions
        img_h, img_w = img.shape[:2]

        # Denormalize bounding boxes using actual image dimensions
        boxes_norm = sub.select(["x_min", "y_min", "x_max", "y_max"]).to_numpy()
        boxes = np.clip(boxes_norm, 0, 1) * np.array([img_w, img_h, img_w, img_h])
        boxes = boxes.astype(np.int32)

        labels = sub.get_column("label").to_list()
        track_ids = sub.get_column("track_id").to_list()
        confs = sub.get_column("confidence").to_list() if "confidence" in sub.columns else None

        local_stats = {"frames_processed": 0, "crops_created": 0}
        group_abs_crop_paths: list[Path] = []

        # Process crops with batching
        if post_config.crop_enabled:
            crop_batch: list[tuple[Path, np.ndarray]] = []

            for i, (x0, y0, x1, y1) in enumerate(boxes):
                label = labels[i]
                track_id = track_ids[i]
                crop_filename = f"{ts_name}_ID{track_id}_crop{post_config.img_ext}"
                out_path = crops_dir / str(label) / crop_filename

                # Generate crop
                if post_config.crop_method == "square":
                    x0, y0, x1, y1 = make_bbox_square(img_w, img_h, x0, y0, x1, y1)

                if x1 <= x0 or y1 <= y0:
                    logger.warning(
                        "Skipping degenerate crop bbox for %s: (%d, %d, %d, %d)",
                        crop_filename, x0, y0, x1, y1
                    )
                    continue

                crop = img[y0:y1, x0:x1]
                crop_batch.append((out_path, crop))

            # Check for cancellation before the batch write (expensive I/O)
            if cancel_event.is_set():
                raise PipelineCancelled()

            # Batch write crops, then record only the ones that actually succeeded
            if crop_batch:
                write_ok = _batch_write_crops(crop_batch)
                for ok, (out_path, _) in zip(write_ok, crop_batch):
                    if ok:
                        group_abs_crop_paths.append(out_path)
                local_stats["crops_created"] += sum(write_ok)

        # Process overlay
        if post_config.overlay_enabled:
            ov_path = overlays_dir / f"{ts_name}_overlay{post_config.img_ext}"

            # Safe to draw in-place: crops (numpy views into img) are already written to disk above
            _draw_overlays(img, boxes, labels, confs, track_ids)
            cv2.imwrite(str(ov_path), img)
            local_stats["frames_processed"] += 1

        return {
            "group_idx": group_idx,
            "ok": True,
            "abs_crop_paths": group_abs_crop_paths,
            "stats": local_stats
        }

    # Pre-allocate results list
    results: list[dict[str, Any] | None] = [None] * total_groups
    processed_count = 0
    last_update_pct = 0

    # Process groups with threading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        group_list = [(g[1], i) for i, g in enumerate(groups)]
        futures = {ex.submit(handle_group, group_data): group_data[1] for group_data in group_list}

        try:
            for f in as_completed(futures):
                group_idx = futures[f]
                try:
                    result = f.result()
                except PipelineCancelled:
                    raise
                except Exception:
                    logger.exception("Frame group %d failed unexpectedly, skipping", group_idx)
                    result = {
                        "group_idx": group_idx,
                        "ok": False,
                        "skip_reason": "error",
                        "stats": {"frames_processed": 0, "crops_created": 0}
                    }

                results[group_idx] = result
                processed_count += 1

                # Throttled progress updates: only fire when the displayed percent changes
                current_pct = 10 + int((processed_count / total_groups) * 80)
                if (
                    progress_callback
                    and (current_pct != last_update_pct or processed_count == total_groups)
                ):
                    elapsed = time.time() - start_time
                    frames_per_sec = processed_count / max(elapsed, 0.1)
                    remaining_frames = total_groups - processed_count
                    eta_seconds = remaining_frames / max(frames_per_sec, 0.1)

                    msg_parts = [f"Processing: {processed_count}/{total_groups} frames"]
                    if frames_per_sec > 0:
                        msg_parts.append(f"{frames_per_sec:.1f} fps")
                    if processed_count < total_groups:
                        msg_parts.append(f"ETA: {format_time(eta_seconds)}")

                    progress_callback(current_pct, 100, " | ".join(msg_parts))
                    last_update_pct = current_pct

                # Aggregate stats
                if "stats" in result:
                    for key, value in result["stats"].items():
                        stats[key] += value

                if not result.get("ok", True):
                    reason = result.get("skip_reason")
                    if reason == "missing":
                        stats["frames_skipped_missing"] += 1
                    elif reason == "corrupt":
                        stats["frames_skipped_corrupt"] += 1
                    elif reason == "error":
                        stats["frames_failed"] += 1

        except PipelineCancelled:
            cancel_event.set()
            for future in futures:
                future.cancel()
            raise

    if progress_callback:
        progress_callback(90, 100, "Finalizing crop file list...")

    # Build crop file list from write results
    if post_config.crop_enabled:
        all_crop_paths = [
            abs_path
            for result in results
            if result and result.get("abs_crop_paths")
            for abs_path in result["abs_crop_paths"]
        ]
        logger.debug("Collected %d crop file paths", len(all_crop_paths))
    else:
        all_crop_paths = []

    # Calculate final statistics
    elapsed_total = time.time() - start_time

    frames_skipped_total = (
        stats["frames_skipped_missing"] + stats["frames_skipped_corrupt"] + stats["frames_failed"]
    )

    if progress_callback:
        final_msg = f"Complete: {total_groups} frames in {format_time(elapsed_total)}"
        if stats['crops_created'] > 0:
            final_msg += f" | {stats['crops_created']} crops created"
        if frames_skipped_total > 0:
            final_msg += f" | {frames_skipped_total} frames skipped"
        progress_callback(100, 100, final_msg)

    return {
        "frames_processed": stats["frames_processed"],
        "crops_created": stats["crops_created"],
        "crop_file_list": all_crop_paths,
        "frames_skipped_missing": stats["frames_skipped_missing"],
        "frames_skipped_corrupt": stats["frames_skipped_corrupt"],
        "frames_failed": stats["frames_failed"],
    }

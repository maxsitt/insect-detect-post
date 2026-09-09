"""Run the post-processing pipeline: image + metadata processing, classification, and crop sorting.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Runs image processing, classification, metadata processing/aggregation, and crop sorting
in sequence, skipping steps disabled in the config. Emits Qt signals for progress and
completion/error status so the GUI can track a run without blocking.

Classes:
    PipelineRunner: Runs pipeline steps with progress reporting and cancellation support.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import polars as pl
from PySide6.QtCore import QObject, Signal

from insectdetect_post.asset_manager import ensure_asset
from insectdetect_post.classifier_utils import parse_crop_name
from insectdetect_post.config import AppConfig
from insectdetect_post.constants import (
    MAX_WORKERS,
    MODELS_JSON,
    MODELS_PATH,
    RANK_ORDER,
    OutputLayout,
)
from insectdetect_post.dataset_context import CropOnlyEntry, DatasetContext
from insectdetect_post.exceptions import PipelineCancelled
from insectdetect_post.image_processor import PostConfig, process_images
from insectdetect_post.metadata_processor import process_metadata_classified

# Create module-level logger
logger = logging.getLogger(__name__)


class PipelineRunner(QObject):
    """Runs pipeline steps with progress reporting and cancellation support."""
    progress = Signal(int)
    progress_message = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        config_stem: str = "config",
        context: DatasetContext | None = None
    ) -> None:
        """Initialize the pipeline runner for a single run.

        Args:
            config: Fully resolved app configuration for this run.
            config_stem: Filename stem of the active config file.
            context: Pre-scanned dataset context, if already available.

        Raises:
            RuntimeError: If config.source_path is not set.
        """
        super().__init__()
        if not config.source_path:
            raise RuntimeError("source_path is not configured")
        self.source_dir = Path(config.source_path)
        self.config = config
        self.config_stem = config_stem
        self._context: DatasetContext | None = context
        self._cancelled = False
        self._output_root: Path | None = None
        self._layout: OutputLayout | None = None
        self.run_timestamp: str | None = None
        self._merged_metadata_path_value: Path | None = None
        self._classified_metadata_path: Path | None = None
        self._final_metadata_path_value: Path | None = None
        self._crop_file_cache: list[Path] = []
        self._total_steps: int = 0
        self._current_step_index: int = 0

    @property
    def context(self) -> DatasetContext:
        """Cached dataset context; raises if not yet initialized."""
        if self._context is None:
            raise RuntimeError("context has not been initialized")
        return self._context


    @property
    def _merged_metadata_path(self) -> Path:
        """Path to the merged metadata CSV; raises if not yet created."""
        if self._merged_metadata_path_value is None:
            raise RuntimeError("_merged_metadata_path has not been initialized")
        return self._merged_metadata_path_value


    @property
    def _final_metadata_path(self) -> Path:
        """Path to the processed final metadata CSV; raises if not yet created."""
        if self._final_metadata_path_value is None:
            raise RuntimeError("_final_metadata_path has not been initialized")
        return self._final_metadata_path_value


    @property
    def output_root(self) -> Path:
        """Root output directory for this run; raises if not yet created."""
        if self._output_root is None:
            raise RuntimeError("output_root has not been initialized")
        return self._output_root


    @property
    def layout(self) -> OutputLayout:
        """Directory layout for this run's output root; raises if not yet created."""
        if self._layout is None:
            raise RuntimeError("layout has not been initialized")
        return self._layout


    def run(self) -> None:
        """Execute complete post-processing pipeline with progress reporting."""
        log_file_handler = None

        try:
            self._context = self._ensure_context()
            self._create_output_structure()
            self._save_pipeline_config()

            # Set up log file handler scoped to this package only
            log_file = self.output_root / f"{self.run_timestamp}_run.log"
            log_file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            log_file_handler.setLevel(logging.DEBUG)
            log_file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logging.getLogger("insectdetect_post").addHandler(log_file_handler)

            logger.info("=" * 60)
            logger.info("Starting post-processing pipeline")
            logger.info("Source: %s", self.source_dir)
            logger.info("Output: %s", self.output_root)
            logger.info("=" * 60)

            # Create merged metadata
            self._merged_metadata_path_value = self._create_merged_metadata()

            # Determine steps
            steps = []
            ultralytics_enabled = self.config.classification.ultralytics.enabled
            bioclip_enabled = self.config.classification.bioclip.enabled
            classification_enabled = ultralytics_enabled or bioclip_enabled

            # Step 1: Image Processing
            if self._should_process_images():
                steps.append(("Image Processing", self._process_images))

            # Step 2: Classification
            if ultralytics_enabled:
                steps.append(("Ultralytics Classification", self._run_ultralytics_classification))
            elif bioclip_enabled:
                steps.append(("BioCLIP Classification", self._run_bioclip_classification))

            # Step 3: Metadata Processing (aggregates classification results)
            meta_config = self.config.metadata
            if classification_enabled:
                steps.append(("Metadata Processing", self._process_metadata))
            elif (meta_config.filter_tracks.enabled or meta_config.filter_predictions.enabled
                  or meta_config.estimate_size.enabled):
                logger.warning(
                    "Metadata settings are enabled but no classifier is - metadata processing "
                    "aggregates classification results and will be skipped"
                )

            # Step 4: Sort crops (only if classification was done)
            if classification_enabled and self.config.classification.sort_crops.enabled:
                steps.append(("Sort Crops by Individual Prediction", self._sort_crops_wrapper))
            elif classification_enabled and self.config.classification.sort_tracks.enabled:
                steps.append(("Sort Tracks by Final Prediction", self._sort_tracks_wrapper))

            if not steps:
                logger.warning("No processing steps enabled")
                self.finished.emit()
                return

            logger.info("Enabled steps: %s", ', '.join(name for name, _ in steps))

            # Execute steps
            pipeline_stats = {
                "run_timestamp": self.run_timestamp,
                "source_dir": str(self.source_dir),
                "output_dir": str(self.output_root),
                "steps_completed": [],
                "total_duration_seconds": 0,
            }
            pipeline_start = time.time()
            self._total_steps = len(steps)

            for i, (step_name, step_func) in enumerate(steps):
                if self._cancelled:
                    raise PipelineCancelled()

                self._current_step_index = i
                logger.info("Step %d/%d: %s", i + 1, len(steps), step_name)
                self.progress_message.emit(f"Step {i+1}/{len(steps)}: {step_name}")

                step_start = time.time()
                step_func()
                step_duration = time.time() - step_start

                pipeline_stats["steps_completed"].append({
                    "name": step_name,
                    "duration_seconds": round(step_duration, 2)
                })

                logger.info("%s completed in %.1fs", step_name, step_duration)

            pipeline_stats["total_duration_seconds"] = round(time.time() - pipeline_start, 2)
            self._save_pipeline_stats(pipeline_stats)

            logger.info("=" * 60)
            logger.info("Pipeline completed successfully")
            logger.info("Total time: %.1fs", pipeline_stats['total_duration_seconds'])
            logger.info("=" * 60)

            self.progress.emit(100)
            self.progress_message.emit("Pipeline completed")
            self.finished.emit()

        except PipelineCancelled:
            logger.info("Pipeline cancelled")
            self.error.emit("Pipeline cancelled")
        except Exception as e:
            logger.exception("Pipeline error")
            self.error.emit(str(e))
        finally:
            if log_file_handler:
                logging.getLogger("insectdetect_post").removeHandler(log_file_handler)
                log_file_handler.close()


    def cancel(self) -> None:
        """Request cancellation of the pipeline."""
        if not self._cancelled:
            self._cancelled = True
            logger.info("Pipeline cancellation requested")


    def _ensure_context(self) -> DatasetContext:
        """Ensure dataset context exists, create if needed.

        Returns:
            The cached or freshly scanned DatasetContext.
        """
        if self._context is None:
            logger.warning("No cached context, performing fresh scan")
            self._context = DatasetContext.from_inspection(
                self.source_dir,
                progress_callback=lambda percent, message: self._progress_callback(percent, 100, message)
            )
        return self.context


    def _create_output_structure(self) -> Path:
        """Create timestamped output directory structure.

        Returns:
            Path to the output root directory.
        """
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        source_name = self.source_dir.name
        output_dir_name = f"{self.run_timestamp}_{source_name}_processed"

        if not self.config.output_path:
            raise RuntimeError("output_path is not configured")
        processed_root = Path(self.config.output_path) / "data_processed"
        self._output_root = processed_root / output_dir_name
        self._layout = OutputLayout(self._output_root)
        processed_root.mkdir(exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        layout = self.layout
        layout.metadata_dir.mkdir(exist_ok=True)
        if self.config.processing.crop.enabled:
            layout.crops_dir.mkdir(parents=True, exist_ok=True)
        if self.config.processing.overlay.enabled:
            layout.overlays_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output directory: %s", self.output_root)
        return self.output_root


    def _create_merged_metadata(self) -> Path:
        """Create merged metadata from dataset context.

        Returns:
            Path to the merged metadata CSV.
        """
        df = self.context.get_metadata_df()
        if df is None:
            raise RuntimeError("No metadata available in context")

        metadata_filename = f"{self.source_dir.name}_metadata_merged.csv"
        metadata_path = self.layout.metadata_dir / metadata_filename
        df.write_csv(metadata_path)
        logger.info("Created merged metadata: %s", metadata_path.name)
        return metadata_path


    def _save_pipeline_config(self) -> None:
        """Save current configuration as JSON snapshot."""
        config_filename = f"{self.run_timestamp}_{self.config_stem}.json"
        config_path = self.output_root / config_filename

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        logger.debug("Saved config: %s", config_filename)


    def _save_pipeline_stats(self, stats: dict[str, Any]) -> None:
        """Save processing statistics.

        Args:
            stats: Pipeline run statistics to serialize as JSON.
        """
        stats_filename = f"{self.run_timestamp}_stats.json"
        stats_path = self.output_root / stats_filename

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.debug("Saved stats: %s", stats_filename)


    def _progress_callback(self, current: int, _total: int, message: str) -> None:
        """Progress callback that checks for cancellation.

        Args:
            current: Current progress value (0-100).
            _total: Unused; kept to match the (current, total, message) callback signature.
            message: Status message to display.
        """
        if self._cancelled:
            raise PipelineCancelled()
        self.progress.emit(current)
        if self._total_steps > 0:
            step_label = f"[{self._current_step_index + 1}/{self._total_steps}] "
            self.progress_message.emit(step_label + message)
        else:
            self.progress_message.emit(message)


    def _should_process_images(self) -> bool:
        """Check if image processing should run.

        Returns:
            True if cropping/overlay is enabled and there is source material to process.
        """
        crop_enabled = self.config.processing.crop.enabled
        overlay_enabled = self.config.processing.overlay.enabled

        if not crop_enabled and not overlay_enabled:
            return False

        has_frames = bool(self.context.detection_frames)
        has_crop_only = crop_enabled and bool(self.context.crop_only_map)

        if not has_frames and not has_crop_only:
            logger.warning("No detection frames or crop-only files available for image processing")
            return False

        return True


    def _process_images(self) -> None:
        """Execute image processing (cropping/overlay).

        For mixed datasets (detection frames + crop-only files):
        1. Generate crops from detection frames
        2. Copy crop-only files from source (using context.crop_only_map)
        """
        post_config = PostConfig(
            crop_enabled=self.config.processing.crop.enabled,
            crop_method=self.config.processing.crop.method,
            overlay_enabled=self.config.processing.overlay.enabled,
            output_dir=self.output_root
        )

        logger.info("Config: crop=%s, overlay=%s, method=%s",
                    post_config.crop_enabled, post_config.overlay_enabled, post_config.crop_method)

        # Step 1: Process detection frames (generate crops + overlays)
        if len(self.context.detection_frames) > 0:
            result = process_images(
                metadata_path=self._merged_metadata_path,
                post_config=post_config,
                progress_callback=self._progress_callback,
                context=self.context
            )

            logger.info("Processed %d frames", result['frames_processed'])
            logger.info("Created %d crops", result['crops_created'])

            if result.get('frames_skipped_missing'):
                logger.warning(
                    "Skipped %d frames with no detection frame available", result['frames_skipped_missing']
                )
            if result.get('frames_skipped_corrupt'):
                logger.warning(
                    "Skipped %d frames with unreadable/corrupt image data", result['frames_skipped_corrupt']
                )
            if result.get('frames_failed'):
                logger.warning(
                    "Skipped %d frames due to unexpected processing errors", result['frames_failed']
                )

            # Cache outputs for subsequent steps
            if post_config.crop_enabled and result.get('crop_file_list'):
                self._crop_file_cache = result['crop_file_list']
                logger.debug("Cached %d crop paths", len(self._crop_file_cache))
        else:
            result = {"crops_created": 0}

        # Step 2: Copy crop-only files from source
        if post_config.crop_enabled and len(self.context.crop_only_map) > 0:
            copied = self._copy_crop_only_files()
            logger.info("Copied %d crop-only files from source", copied)

        # Step 3: Build crop_path column from all crop files on disk
        if post_config.crop_enabled and self._crop_file_cache:
            self._write_crops_metadata()


    def _copy_crop_only_files(self) -> int:
        """Copy crop-only files from source to output directory.

        These are detections that have existing crop files but no detection frame.
        Preserves label directory structure.

        Returns:
            Number of files successfully copied.
        """
        if not self.context.crop_only_map:
            return 0

        crops_dir = self.layout.crops_dir
        crops_dir.mkdir(parents=True, exist_ok=True)

        def copy_crop(entry: CropOnlyEntry) -> Path | None:
            """Copy a single crop-only file. Runs in a worker thread."""
            src_path = entry.path
            try:
                if not src_path.exists():
                    return None

                # Check for corrupt/unreadable image files
                if cv2.imread(str(src_path)) is None:
                    logger.warning("Corrupt crop file %s, skipping", src_path.name)
                    return None

                # Create destination
                label_dir = self.layout.crop_label_dir(entry.label)
                label_dir.mkdir(parents=True, exist_ok=True)
                dest_path = label_dir / src_path.name

                shutil.copy2(src_path, dest_path)
                return dest_path

            except Exception:
                logger.exception("Failed to copy crop %s", src_path.name)
                return None

        total = len(self.context.crop_only_map)
        copied = 0
        processed = 0
        last_update_pct = -1

        self._progress_callback(0, 100, f"Copying {total} crop-only files...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(copy_crop, entry) for entry in self.context.crop_only_map.values()]
            for future in as_completed(futures):
                dest_path = future.result()
                if dest_path is not None:
                    copied += 1
                    self._crop_file_cache.append(dest_path)

                processed += 1
                current_pct = int((processed / total) * 100)

                # Throttled progress updates: only fire when the displayed percent changes
                if current_pct != last_update_pct or processed == total:
                    self._progress_callback(
                        current_pct, 100, f"Copying crop-only files: {processed}/{total}"
                    )
                    last_update_pct = current_pct

        return copied


    def _write_crops_metadata(self) -> None:
        """Add a crop_path column (derived from _crop_file_cache) to the merged metadata CSV."""
        df = pl.read_csv(self._merged_metadata_path)

        crop_path_map: dict[tuple[str, int], str] = {}
        for crop_path in self._crop_file_cache:
            ts_iso, track_id = parse_crop_name(crop_path.name)
            if track_id is not None:
                crop_path_map[(ts_iso, track_id)] = self.layout.relative(crop_path)

        crop_path_df = pl.DataFrame({
            "timestamp": [key[0] for key in crop_path_map],
            "track_id": [key[1] for key in crop_path_map],
            "crop_path": list(crop_path_map.values()),
        })

        df = df.join(crop_path_df, on=["timestamp", "track_id"], how="left")
        df.write_csv(self._merged_metadata_path)
        logger.info("Updated merged metadata with crop paths: %s", self._merged_metadata_path.name)


    def _run_ultralytics_classification(self) -> None:
        """Classify all cached crops with the configured Ultralytics model."""
        from insectdetect_post.classifier_ultralytics import classify_imgs_ultralytics

        if not self._crop_file_cache:
            raise RuntimeError(
                "No crop images available for classification. "
                "Enable 'Crop Detections' to generate/copy crops."
            )

        crop_dir = self.layout.crops_dir
        metadata_path = self._merged_metadata_path
        cls_config = self.config.classification.ultralytics
        model_name = cls_config.model
        batch_size = cls_config.batch_size
        device = self.config.device

        model_path = MODELS_PATH / model_name
        try:
            model_path = ensure_asset(
                model_name, MODELS_JSON, progress_callback=self._progress_callback
            )
        except KeyError as e:
            # Not a registered asset: fall back to a user-supplied model file if present
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}") from e
            logger.debug("Model '%s' is not a registered asset, using the local file.", model_name)

        logger.info("Crop directory: %s", crop_dir)
        logger.info("Total crops: %d", len(self._crop_file_cache))
        logger.info("Model: %s", model_name)
        logger.info("Batch size: %d", batch_size)
        logger.info("Device: %s", device)

        result_path = classify_imgs_ultralytics(
            crop_file_list=self._crop_file_cache,
            crop_dir=crop_dir,
            metadata_path=metadata_path,
            output_dir=self.output_root,
            model_path=model_path,
            batch_size=batch_size,
            device=device,
            progress_callback=self._progress_callback
        )

        logger.info("Results saved: %s", result_path.name)
        self._classified_metadata_path = result_path


    def _run_bioclip_classification(self) -> None:
        """Classify all cached crops with the BioCLIP 2 model."""
        from insectdetect_post.classifier_bioclip import classify_imgs_bioclip

        if not self._crop_file_cache:
            raise RuntimeError(
                "No crop images available for classification. "
                "Enable 'Crop Detections' to generate/copy crops."
            )

        metadata_path = self._merged_metadata_path
        cls_config = self.config.classification.bioclip
        batch_size = cls_config.batch_size
        device = self.config.device
        rank = cls_config.rank
        filter_arthropods = cls_config.filter_arthropods

        logger.info("Crop directory: %s", self.layout.crops_dir)
        logger.info("Total crops: %d", len(self._crop_file_cache))
        logger.info("Using BioCLIP 2 model")
        logger.info("Rank: %s", rank)
        logger.info("Batch size: %d", batch_size)
        logger.info("Device: %s", device)

        result_path = classify_imgs_bioclip(
            crop_file_list=self._crop_file_cache,
            metadata_path=metadata_path,
            output_dir=self.output_root,
            batch_size=batch_size,
            rank=rank,
            filter_arthropods_enabled=filter_arthropods.enabled,
            filter_taxon=filter_arthropods.taxon,
            filter_country=filter_arthropods.country,
            device=device,
            progress_callback=self._progress_callback
        )

        logger.info("Results saved: %s", result_path.name)
        self._classified_metadata_path = result_path


    def _process_metadata(self) -> None:
        """Post-process and optionally filter classified metadata."""
        if not self._classified_metadata_path or not self._classified_metadata_path.exists():
            raise FileNotFoundError("No classified metadata found")

        classified_csv = self._classified_metadata_path
        meta_config = self.config.metadata
        filter_tracks = meta_config.filter_tracks
        min_conf = filter_tracks.min_det_conf if filter_tracks.enabled else None
        min_duration = filter_tracks.min_dur_s if filter_tracks.enabled else None
        max_duration = filter_tracks.max_dur_s if filter_tracks.enabled else None
        filter_pred = meta_config.filter_predictions
        min_prob = filter_pred.min_prob_weighted if filter_pred.enabled else None
        estimate_size = meta_config.estimate_size
        frame_width_mm = estimate_size.frame_width_mm if estimate_size.enabled else None
        frame_height_mm = estimate_size.frame_height_mm if estimate_size.enabled else None

        logger.info("Processing: %s", classified_csv.name)

        result = process_metadata_classified(
            metadata_path=classified_csv,
            output_dir=self.output_root,
            min_conf_mean=min_conf,
            min_dur_s=min_duration,
            max_dur_s=max_duration,
            min_prob_weighted=min_prob,
            frame_width_mm=frame_width_mm,
            frame_height_mm=frame_height_mm,
            progress_callback=self._progress_callback
        )

        self._final_metadata_path_value = result['final_path']

        logger.info("Saved: %s", result['candidates_path'].name)
        logger.info("Saved: %s", result['final_path'].name)
        logger.info("Kept %d/%d tracks", result['tracks_kept'], result['tracks_total'])
        logger.info("  - removed %d (below mean detection confidence threshold)", result['conf_tracks_removed'])
        logger.info("  - removed %d (below minimum duration threshold)", result['min_dur_tracks_removed'])
        logger.info("  - removed %d (above maximum duration threshold)", result['max_dur_tracks_removed'])
        logger.info("  - removed %d (below weighted prediction probability threshold)", result['pred_tracks_removed'])


    def _sort_crops_wrapper(self) -> None:
        """Sort crops by individual prediction using the cached classified metadata path."""
        if not self._classified_metadata_path:
            logger.warning("No classified metadata available, skipping sort crops")
            return

        if not self._crop_file_cache:
            logger.warning("No crop directory available, skipping sort crops")
            return

        self._sort_crops_by_prediction(self._classified_metadata_path, self.layout.crops_dir)


    def _sort_crops_by_prediction(self, classified_metadata_path: Path, crop_dir: Path) -> None:
        """Move crop images to subdirectories based on each crop's individual prediction.

        Ultralytics: flat layout   → crops/top1_label/file.jpg
        BioCLIP:     nested layout → crops/kingdom/phylum/.../rank/file.jpg

        Args:
            classified_metadata_path: Path to the classifier's output metadata CSV.
            crop_dir: Root crops directory used to resolve relative crop_path values.
        """
        self._progress_callback(0, 100, "Reading individual prediction results...")
        logger.info("Sorting crops based on individual prediction results...")
        logger.info("Crop directory: %s", crop_dir)

        lazy = pl.scan_csv(classified_metadata_path, infer_schema_length=0)
        schema_cols = lazy.collect_schema().names()

        if "crop_path" not in schema_cols:
            logger.warning("No 'crop_path' column found, skipping sort")
            return

        # Build the ordered list of columns that define the destination path
        path_cols = [f"bioclip_top1_{rank}" for rank in RANK_ORDER if f"bioclip_top1_{rank}" in schema_cols]
        if not path_cols:
            # Ultralytics classifier output
            if "top1" not in schema_cols:
                logger.warning("No 'top1' or BioCLIP taxonomy columns found, skipping sort")
                return
            path_cols = ["top1"]

        logger.info("Sorting by columns: %s", path_cols)

        needed = ["crop_path"] + path_cols
        df_valid = (
            lazy
            .select(needed)
            .filter(pl.all_horizontal(pl.col(c).is_not_null() for c in needed))
            .collect()
        )

        if len(df_valid) == 0:
            logger.warning("No valid crop paths to sort")
            return

        moved, errors, crop_path_updates = self._move_crops(
            df_valid, crop_dir, lambda row: [str(row[col]) for col in path_cols]
        )

        logger.info("Sorted %d crops into prediction subdirectories", moved)
        if errors > 0:
            logger.warning("Failed to sort %d crops", errors)

        # Cleanup empty directories
        self._progress_callback(90, 100, f"Cleaning up directories ({moved} crops sorted)...")
        self._cleanup_empty_dirs(crop_dir)

        if crop_path_updates:
            self._update_crop_paths(classified_metadata_path, crop_path_updates)
            self._update_crop_paths(self._merged_metadata_path, crop_path_updates)

        self._progress_callback(100, 100, f"Sorted {moved} crops into prediction subdirectories")


    def _sort_tracks_wrapper(self) -> None:
        """Sort crops by track-level final prediction, using cached metadata paths."""
        if not self._classified_metadata_path:
            logger.warning("No classified metadata available, skipping sort tracks")
            return

        if not self._crop_file_cache:
            logger.warning("No crop directory available, skipping sort tracks")
            return

        if not self._final_metadata_path_value:
            logger.warning("No final metadata available, skipping sort tracks")
            return

        self._sort_tracks_by_final_prediction(
            self._classified_metadata_path, self._final_metadata_path, self.layout.crops_dir
        )


    def _sort_tracks_by_final_prediction(
        self,
        classified_metadata_path: Path,
        final_metadata_path: Path,
        crop_dir: Path
    ) -> None:
        """Move each track's crops into one subdirectory named by the track's final prediction.

        Ultralytics: flat layout   → crops/pred/track/file.jpg.
        BioCLIP:     nested layout → crops/kingdom/phylum/.../rank/track/file.jpg.
        Tracks filtered out during metadata processing are left unsorted.

        Args:
            classified_metadata_path: Path to the classifier's output metadata CSV.
            final_metadata_path: Path to the metadata processing '..._final.csv' output.
            crop_dir: Root crops directory used to resolve relative crop_path values.
        """
        self._progress_callback(0, 100, "Reading final prediction results...")
        logger.info("Sorting crops based on track-level final predictions...")
        logger.info("Crop directory: %s", crop_dir)

        final_lazy = pl.scan_csv(final_metadata_path, infer_schema_length=0)
        final_cols = final_lazy.collect_schema().names()

        id_cols = ["device_id", "session_id", "track_id"]
        missing_final = [c for c in id_cols if c not in final_cols]
        if missing_final:
            logger.warning("Missing columns %s in final metadata, skipping sort", missing_final)
            return

        # Build the ordered list of columns that define the destination path
        path_cols = [f"bioclip_{rank}" for rank in RANK_ORDER if f"bioclip_{rank}" in final_cols]
        if not path_cols:
            # Ultralytics classifier output
            if "pred" not in final_cols:
                logger.warning("No 'pred' or BioCLIP taxonomy columns found, skipping sort")
                return
            path_cols = ["pred"]

        logger.info("Sorting by columns: %s", path_cols)

        needed_final = id_cols + path_cols
        df_final = (
            final_lazy
            .select(needed_final)
            .filter(pl.all_horizontal(pl.col(c).is_not_null() for c in needed_final))
            .collect()
        )

        if len(df_final) == 0:
            logger.warning("No valid track predictions to sort by")
            return

        crop_lazy = pl.scan_csv(classified_metadata_path, infer_schema_length=0)
        crop_cols = crop_lazy.collect_schema().names()
        required_crop = ["crop_path"] + id_cols
        missing_crop = [c for c in required_crop if c not in crop_cols]
        if missing_crop:
            logger.warning("Missing columns %s in classified metadata, skipping sort", missing_crop)
            return

        df_valid = (
            crop_lazy
            .select(required_crop)
            .filter(pl.all_horizontal(pl.col(c).is_not_null() for c in required_crop))
            .collect()
            .join(df_final, on=["device_id", "session_id", "track_id"], how="inner")
        )

        if len(df_valid) == 0:
            logger.warning("No valid crop paths to sort")
            return

        def build_dest_dir(row: dict[str, Any]) -> list[str]:
            """Build the taxonomy + track id path segments for one crop's destination."""
            track_dir_name = f"{row['device_id']}_{row['session_id']}_ID{row['track_id']}"
            return [str(row[col]) for col in path_cols] + [track_dir_name]

        moved, errors, crop_path_updates = self._move_crops(df_valid, crop_dir, build_dest_dir)

        logger.info("Sorted %d crops into track-level prediction subdirectories", moved)
        if errors > 0:
            logger.warning("Failed to sort %d crops", errors)

        # Cleanup empty directories
        self._progress_callback(90, 100, f"Cleaning up directories ({moved} crops sorted)...")
        self._cleanup_empty_dirs(crop_dir)

        if crop_path_updates:
            self._update_crop_paths(classified_metadata_path, crop_path_updates)
            self._update_crop_paths(self._merged_metadata_path, crop_path_updates)

        self._progress_callback(100, 100, f"Sorted {moved} crops into track-level prediction subdirectories")


    def _move_crops(
        self,
        df_valid: pl.DataFrame,
        crop_dir: Path,
        build_dest_dir: Callable[[dict[str, Any]], Sequence[str]],
    ) -> tuple[int, int, dict[str, str]]:
        """Move each row's crop into a subdirectory relative to its current parent.

        Args:
            df_valid: Rows with a "crop_path" column plus any columns build_dest_dir needs.
            crop_dir: Root crops directory used to resolve relative crop_path values.
            build_dest_dir: Returns the destination path segments for a given row.

        Returns:
            Tuple of (moved count, error count, {old_rel_path: new_rel_path}).
        """
        crops_prefix = self.layout.relative(self.layout.crops_dir) + "/"
        total = len(df_valid)
        moved = 0
        errors = 0
        crop_path_updates: dict[str, str] = {}
        last_update_pct = -1

        for i, row in enumerate(df_valid.iter_rows(named=True)):
            if self._cancelled:
                raise PipelineCancelled()

            current_pct = int((i / total) * 90)
            if current_pct != last_update_pct:
                self._progress_callback(current_pct, 100, f"Sorting crops: {i}/{total}")
                last_update_pct = current_pct

            try:
                crop_rel_path = row["crop_path"]

                if crop_rel_path.startswith(crops_prefix):
                    src_path = self.output_root / crop_rel_path
                else:
                    src_path = Path(crop_rel_path)
                    if not src_path.is_absolute():
                        src_path = crop_dir / crop_rel_path

                if not src_path.exists():
                    continue

                dest_dir = src_path.parent
                for segment in build_dest_dir(row):
                    dest_dir = dest_dir / segment
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / src_path.name
                shutil.move(src_path, dest_path)
                moved += 1

                # Record new location to update the classified metadata CSV below
                crop_path_updates[crop_rel_path] = self.layout.relative(dest_path)

            except Exception:
                errors += 1
                if errors <= 5:
                    logger.exception("Failed to sort crop")

        return moved, errors, crop_path_updates


    def _update_crop_paths(self, metadata_path: Path, updates: dict[str, str]) -> None:
        """Rewrite crop_path values in a metadata CSV after moving files.

        Args:
            metadata_path: CSV file to update in place.
            updates: Mapping of old to new relative crop_path values.
        """
        df = pl.read_csv(metadata_path)
        if "crop_path" not in df.columns:
            return

        mapping_df = pl.DataFrame({
            "crop_path": list(updates.keys()),
            "crop_path_new": list(updates.values()),
        })

        df = (
            df.join(mapping_df, on="crop_path", how="left")
            .with_columns(pl.coalesce(["crop_path_new", "crop_path"]).alias("crop_path"))
            .drop("crop_path_new")
        )
        df.write_csv(metadata_path)
        logger.debug("Updated crop_path for %d sorted crops in %s", len(updates), metadata_path.name)


    def _cleanup_empty_dirs(self, root_dir: Path) -> None:
        """Remove empty subdirectories under root_dir, bottom-up so emptied parents are removed too.

        Re-checks each directory with iterdir() since walk()'s listing is a snapshot
        taken before removal started.

        Args:
            root_dir: Directory to clean; never removed itself even if empty.
        """
        for dir_path, _dirnames, _filenames in root_dir.walk(top_down=False):
            if dir_path == root_dir:
                continue
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                logger.debug("Removed empty directory: %s", dir_path.name)

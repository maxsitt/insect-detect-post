"""GUI application for configuring and running the post-processing pipeline.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

This module provides a PySide6-based GUI for configuring and running the
post-processing pipeline, including image processing, classification,
metadata processing/aggregation, and crop sorting.

Classes:
    ConfigWidget: Configuration and dataset-inspection widget for the pipeline.
    MainWindow: Main application window with pipeline controls and log viewer.

Functions:
    main: Main entry point for the post-processing GUI application.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import qt_themes
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qt_logging import LogCache, LogViewer
from qt_parameters import ComboParameter, ParameterForm, PathParameter

from insectdetect_post.asset_manager import list_registered_filenames
from insectdetect_post.config import (
    AppConfig,
    check_config_changes,
    ensure_config_files,
    get_field_constraints,
    get_field_literals,
    load_config_selector,
    load_config_yaml,
    update_config_selector,
    update_config_yaml,
)
from insectdetect_post.constants import (
    BASE_PATH,
    COLOR_BTN_CANCEL,
    COLOR_BTN_RESET,
    COLOR_BTN_RUN,
    COLOR_BTN_SAVE,
    COLOR_BTN_UPDATE,
    COLOR_GROUP_CLASSIFICATION,
    COLOR_GROUP_METADATA,
    COLOR_GROUP_PROCESSING,
    CONFIG_SELECTOR_PATH,
    CONFIGS_PATH,
    MODELS_JSON,
    MODELS_PATH,
    PROG_BAR_STYLESHEET,
    SETTINGS_APP,
    SETTINGS_ORG,
    THEME_DEFAULT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from insectdetect_post.dataset_context import DatasetContext
from insectdetect_post.dataset_inspector import DatasetInspector
from insectdetect_post.gui_utils import (
    create_groupbox,
    create_num_param,
    create_param_form,
    extract_enabled_values,
    resolve_and_validate_path,
    restore_enabled_values,
)
from insectdetect_post.multi_combo_parameter import MultiComboParameter
from insectdetect_post.pipeline_runner import PipelineRunner
from insectdetect_post.styled_button import StyledButton
from insectdetect_post.update import UpdateInfo, sync_command
from insectdetect_post.update_runner import UpdateChecker, UpdateInstaller

# Create module-level logger
logger = logging.getLogger(__name__)

# Field constraints, defaults and literals extracted from AppConfig
_CONSTRAINT_PATHS: list[str] = [
    "classification.bioclip.batch_size",
    "classification.ultralytics.batch_size",
    "metadata.filter_tracks.min_det_conf",
    "metadata.filter_tracks.min_dur_s",
    "metadata.filter_tracks.max_dur_s",
    "metadata.filter_predictions.min_prob_weighted",
    "metadata.estimate_size.frame_width_mm",
    "metadata.estimate_size.frame_height_mm",
]
_FIELD_CONSTRAINTS: dict[str, dict[str, int | float | None]] = {
    path: get_field_constraints(AppConfig, *path.split("."))
    for path in _CONSTRAINT_PATHS
}
_DEFAULTS = AppConfig()
BIOCLIP_RANK_OPTIONS = get_field_literals(AppConfig, "classification", "bioclip", "rank")
FILTER_TAXA_OPTIONS = get_field_literals(
    AppConfig, "classification", "bioclip", "filter_arthropods", "taxa"
)
FILTER_REGION_OPTIONS = get_field_literals(
    AppConfig, "classification", "bioclip", "filter_arthropods", "countries"
)


def _get_constraints(path: str) -> tuple[int | float, int | float]:
    """Return (min, max) for a pre-loaded field constraint path; raises if unset."""
    c = _FIELD_CONSTRAINTS[path]
    min_val, max_val = c["min"], c["max"]
    if min_val is None or max_val is None:
        raise RuntimeError(f"Missing min/max constraints for '{path}'")
    return min_val, max_val


def _load_theme() -> str:
    """Return the persisted theme name, falling back to the default if not available."""
    theme_name = str(QSettings(SETTINGS_ORG, SETTINGS_APP).value("theme", THEME_DEFAULT))
    if theme_name not in qt_themes.get_themes():
        logger.warning("Theme '%s' not available, using '%s'", theme_name, THEME_DEFAULT)
        return THEME_DEFAULT
    return theme_name


def _save_theme(theme_name: str) -> None:
    """Persist the selected theme name for the next application start."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("theme", theme_name)


class ConfigWidget(QWidget):
    """Configuration and dataset-inspection widget for the pipeline."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    run_enabled_changed = Signal(bool)
    inspection_active_changed = Signal(bool)

    # UI widget attributes — all assigned unconditionally by _create_ui_layout()
    source_path_param: PathParameter
    output_path_param: PathParameter
    paths_form: ParameterForm
    config_select: QComboBox
    gpu_checkbox: QCheckBox
    processing_form: ParameterForm
    classification_form: ParameterForm
    metadata_form: ParameterForm
    crop_box: Any
    overlay_box: Any
    bioclip_box: Any
    filter_arthropods_box: Any
    ultralytics_box: Any
    sort_crops_box: Any
    sort_tracks_box: Any
    filter_tracks_box: Any
    filter_predictions_box: Any
    estimate_size_box: Any

    def __init__(self) -> None:
        """Initialize the configuration widget with config and model directories."""
        super().__init__()
        self._config: AppConfig | None = None
        self._config_updates: dict[str, Any] = {}
        self.config_active: str = ""
        self._configs: list[str] = []
        self._models: list[str] = []
        self._dataset_inspector: DatasetInspector | None = None
        self._inspector_thread: QThread | None = None
        self._last_inspected_path: Path | None = None
        self._cuda_available: bool | None = None
        self._can_process_images: bool = False
        self._can_classify_images: bool = False
        self._can_process_metadata: bool = False
        self._is_ui_locked: bool = False
        self._loading_config: bool = False
        self._metadata_boxes: list[tuple[Any, str]] = []
        self.dataset_context: DatasetContext | None = None

        # Get active config from config selector (creates config files if missing)
        config_selector = load_config_selector()
        self.config_active = config_selector.config_active

        # Get available configs and models
        self._configs = sorted([f.name for f in CONFIGS_PATH.glob("*.yaml")
                                if f.name != CONFIG_SELECTOR_PATH.name])
        local_models = {f.name for fmt in ["*.pt", "*.onnx"] for f in MODELS_PATH.glob(fmt)}
        try:
            registered_models = set(list_registered_filenames(MODELS_JSON))
        except FileNotFoundError:
            registered_models = set()
        self._models = sorted(local_models | registered_models)

        # Create UI layout
        self._create_ui_layout()

        # Defer GPU/CUDA check and config loading until the event loop starts
        QTimer.singleShot(0, self._initialize_after_display)

    @property
    def source_path(self) -> Path | None:
        """Get resolved source path or None if invalid."""
        path_str = self.paths_form.values().get("source_path", "")
        return resolve_and_validate_path(path_str)

    @property
    def output_path(self) -> Path | None:
        """Get resolved output path or None if invalid."""
        path_str = self.paths_form.values().get("output_path", "")
        return resolve_and_validate_path(path_str)

    @property
    def device(self) -> str:
        """Get the selected compute device ('cuda' or 'cpu')."""
        if self._cuda_available and self.gpu_checkbox.isChecked():
            return "cuda"
        return "cpu"

    @property
    def _is_crop_enabled(self) -> bool:
        """Check if crop detections is currently enabled."""
        return self.crop_box.checkbox.isChecked() if self.crop_box else False

    @property
    def _is_classification_enabled(self) -> bool:
        """Check if any classification method is currently enabled."""
        bioclip_enabled = (
            self.bioclip_box.checkbox.isChecked() if self.bioclip_box else False
        )
        ultralytics_enabled = (
            self.ultralytics_box.checkbox.isChecked() if self.ultralytics_box else False
        )
        return bioclip_enabled or ultralytics_enabled

    def _initialize_after_display(self) -> None:
        """Initialize CUDA check and load config after GUI is displayed."""
        self.status_updated.emit("Checking GPU availability and loading config...")
        QApplication.processEvents()

        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            logger.info("GPU/CUDA %s", "available" if self._cuda_available else "not available")
        except ImportError:
            self._cuda_available = False
            logger.info("PyTorch not installed, CUDA disabled")

        self.gpu_checkbox.setEnabled(self._cuda_available)
        self._load_config(self.config_active)

        status = "GPU available" if self._cuda_available else "GPU not available"
        self.status_updated.emit(f"Ready - {status}")

        self.source_path_param.value_changed.connect(self._on_source_path_change)
        self.output_path_param.value_changed.connect(self._on_output_path_change)

    def _create_ui_layout(self) -> None:
        """Create UI layout for the config widget."""

        # Main vertical layout for the config widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Top row horizontal layout for source + output path and config selector + GPU checkbox
        top_row_layout = QHBoxLayout()
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(6)

        # Combined form for source + output paths
        self.paths_form = ParameterForm("paths")
        self.paths_form.set_flat(True)

        self.source_path_param = PathParameter("source_path")
        self.source_path_param.set_label("Source Path:")
        self.source_path_param.set_method(PathParameter.Method.EXISTING_DIR)
        self.source_path_param.setToolTip("Source folder containing images and metadata")
        self.paths_form.add_parameter(self.source_path_param)

        self.output_path_param = PathParameter("output_path")
        self.output_path_param.set_label("Output Path:")
        self.output_path_param.set_method(PathParameter.Method.EXISTING_DIR)
        self.output_path_param.setToolTip("Output folder where all results will be saved")
        self.paths_form.add_parameter(self.output_path_param)

        # Config selector + GPU checkbox layout
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # Config selector
        config_select_layout = QFormLayout()
        config_select_layout.setContentsMargins(0, 0, 0, 0)
        self.config_select = QComboBox()
        self.config_select.addItems(self._configs)
        self.config_select.setCurrentText(self.config_active)
        self.config_select.currentTextChanged.connect(self._on_config_change)
        config_select_layout.addRow("Active Config:", self.config_select)
        right_layout.addLayout(config_select_layout)

        # GPU checkbox
        gpu_layout = QHBoxLayout()
        gpu_layout.setContentsMargins(0, 0, 0, 0)
        gpu_layout.setSpacing(6)
        self.gpu_checkbox = QCheckBox("Enable GPU")
        self.gpu_checkbox.setToolTip("Enable GPU (Requires NVIDIA GPU + CUDA)")
        self.gpu_checkbox.setEnabled(False)  # will be updated after CUDA check
        gpu_layout.addWidget(self.gpu_checkbox)
        gpu_layout.addStretch()
        right_layout.addLayout(gpu_layout)

        top_row_layout.addWidget(self.paths_form, 2)
        top_row_layout.addLayout(right_layout, 1)
        main_layout.addLayout(top_row_layout)

        # Horizontal layout for config sections
        config_sections_layout = QHBoxLayout()
        config_sections_layout.setContentsMargins(0, 0, 0, 0)
        config_sections_layout.setSpacing(6)

        # Initialize parameter forms and create parameters for each config section
        self.processing_form = ParameterForm("processing")
        self.classification_form = ParameterForm("classification")
        self.metadata_form = ParameterForm("metadata")
        self._create_processing_params()
        self._create_classification_params()
        self._create_metadata_params()

        # Create group boxes for each config section
        processing_groupbox = create_groupbox(
            "Image Processing",
            COLOR_GROUP_PROCESSING,
            self.processing_form
        )
        classification_groupbox = create_groupbox(
            "Classification",
            COLOR_GROUP_CLASSIFICATION,
            self.classification_form
        )
        metadata_groupbox = create_groupbox(
            "Metadata Processing",
            COLOR_GROUP_METADATA,
            self.metadata_form
        )

        config_sections_layout.addWidget(processing_groupbox, 1)
        config_sections_layout.addWidget(classification_groupbox, 1)
        config_sections_layout.addWidget(metadata_groupbox, 1)
        main_layout.addLayout(config_sections_layout, 1)

    def _create_processing_params(self) -> None:
        """Create parameter forms for image processing settings."""
        crop_form, self.crop_box = create_param_form(
            self.processing_form, "crop", "Crop Detections",
            tooltip="Save individual detections as separate .jpg files - cropped from original "
                    "frames, or copied as-is for detections that only have an existing crop file"
        )

        crop_method_select = ComboParameter("method")
        crop_method_select.set_label("Crop Method")
        crop_method_select.set_items(("square", "original"))
        crop_method_select.set_default(_DEFAULTS.processing.crop.method)
        crop_method_select.setToolTip(
            "'square' avoids distortion during resizing for classification and is recommended "
            "(only applies to detections cropped from full frames)"
        )
        crop_form.add_parameter(crop_method_select)

        _, self.overlay_box = create_param_form(
            self.processing_form, "overlay", "Draw Overlays",
            tooltip="Draw overlays on full frames (bounding box, label, confidence, track ID)"
        )

        if self.crop_box:
            self.crop_box.checkbox.toggled.connect(self._on_crop_checkbox_toggle)

    def _create_classification_params(self) -> None:
        """Create parameter forms for classification settings."""
        bioclip_form, self.bioclip_box = create_param_form(
            self.classification_form, "bioclip", "BioCLIP 2",
            tooltip="Use BioCLIP 2 model for classification"
        )

        lo, hi = _get_constraints("classification.bioclip.batch_size")
        bioclip_batch_param = create_num_param(
            "batch_size", "Batch Size",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.classification.bioclip.batch_size,
            tooltip="Number of images processed in parallel (higher = faster but more memory)"
        )
        bioclip_form.add_parameter(bioclip_batch_param)

        rank_select = ComboParameter("rank")
        rank_select.set_label("Rank")
        rank_select.set_items(BIOCLIP_RANK_OPTIONS)
        rank_select.set_default(_DEFAULTS.classification.bioclip.rank)
        rank_select.setToolTip(
            "Taxonomic level for prediction. For ranks above species, species-level "
            "probabilities are summed up to the target rank"
        )
        bioclip_form.add_parameter(rank_select)

        filter_arthropods_form, self.filter_arthropods_box = create_param_form(
            bioclip_form, "filter_arthropods", "Filter Arthropods",
            tooltip="Restrict predictions to the selected arthropod groups (and optionally a region)"
        )
        taxa_select = MultiComboParameter("taxa")
        taxa_select.set_label("Taxa")
        taxa_select.set_items(FILTER_TAXA_OPTIONS)
        taxa_select.set_exclusive_value("all")
        taxa_select.set_min_selection(1)  # an empty selection is rejected by the config schema
        taxa_select.set_default(tuple(_DEFAULTS.classification.bioclip.filter_arthropods.taxa))
        taxa_select.setToolTip("Force BioCLIP to only predict species within the selected taxa")
        filter_arthropods_form.add_parameter(taxa_select)

        countries_select = MultiComboParameter("countries")
        countries_select.set_label("Countries")
        countries_select.set_items(FILTER_REGION_OPTIONS)
        countries_select.set_exclusive_value("all")
        countries_select.set_min_selection(1)  # an empty selection is rejected by the config schema
        countries_select.set_default(tuple(_DEFAULTS.classification.bioclip.filter_arthropods.countries))
        countries_select.setToolTip(
            "Force BioCLIP to only predict species within the selected countries, based on GBIF "
            "occurrence records\n(species list is built and cached automatically on first use)"
        )
        filter_arthropods_form.add_parameter(countries_select)

        ultralytics_form, self.ultralytics_box = create_param_form(
            self.classification_form, "ultralytics", "Ultralytics"
        )

        model_select = ComboParameter("model")
        model_select.set_label("Model File")
        model_select.set_items(self._models)
        default_model = _DEFAULTS.classification.ultralytics.model
        if default_model in self._models:
            model_select.set_default(default_model)
        elif self._models:
            model_select.set_default(self._models[0])
        else:
            model_select.set_default("No models found")
            logger.warning(
                "No model files (.pt or .onnx) found in %s and none registered in %s",
                MODELS_PATH, MODELS_JSON
            )
        model_select.setToolTip(
            "Model weights file in 'models' directory\n"
            "(downloaded and cached automatically on first use)"
        )
        ultralytics_form.add_parameter(model_select)

        lo, hi = _get_constraints("classification.ultralytics.batch_size")
        ultralytics_batch_param = create_num_param(
            "batch_size", "Batch Size",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.classification.ultralytics.batch_size,
            tooltip="Number of images processed in parallel (higher = faster but more memory)"
        )
        ultralytics_form.add_parameter(ultralytics_batch_param)

        # Connect classification checkboxes (handles both mutual exclusion and crop auto-enable)
        self.bioclip_box.checkbox.toggled.connect(self._on_classification_toggle)
        self.ultralytics_box.checkbox.toggled.connect(self._on_classification_toggle)

        _sort_crops_form, self.sort_crops_box = create_param_form(
            self.classification_form, "sort_crops", "Sort Crops by Individual Prediction",
            tooltip="Move crops into subdirectories based on the individual prediction"
        )

        _sort_tracks_form, self.sort_tracks_box = create_param_form(
            self.classification_form, "sort_tracks", "Sort Tracks by Final Prediction",
            tooltip="Move all crops of a track into one subdirectory based on the track's final prediction"
        )

        # Connect sort checkboxes (mutual exclusion between sort_crops and sort_tracks)
        self.sort_crops_box.checkbox.toggled.connect(self._on_sort_mode_toggle)
        self.sort_tracks_box.checkbox.toggled.connect(self._on_sort_mode_toggle)

    def _create_metadata_params(self) -> None:
        """Create parameter forms for metadata processing settings."""
        filter_tracks_form, self.filter_tracks_box = create_param_form(
            self.metadata_form, "filter_tracks", "Filter Tracks",
            tooltip="Filter tracking IDs based on mean detection confidence and tracking duration"
        )

        lo, hi = _get_constraints("metadata.filter_tracks.min_det_conf")
        filter_tracks_form.add_parameter(create_num_param(
            "min_det_conf", "Minimum Confidence",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.filter_tracks.min_det_conf,
            param_type="float",
            tooltip="Minimum required mean detection confidence to retain a tracking ID"
        ))

        lo, hi = _get_constraints("metadata.filter_tracks.min_dur_s")
        filter_tracks_form.add_parameter(create_num_param(
            "min_dur_s", "Minimum Duration (s)",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.filter_tracks.min_dur_s,
            tooltip="Minimum required tracking duration (seconds) to retain a tracking ID"
        ))

        lo, hi = _get_constraints("metadata.filter_tracks.max_dur_s")
        filter_tracks_form.add_parameter(create_num_param(
            "max_dur_s", "Maximum Duration (s)",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.filter_tracks.max_dur_s,
            tooltip="Maximum allowed tracking duration (seconds) to retain a tracking ID"
        ))

        filter_predictions_form, self.filter_predictions_box = create_param_form(
            self.metadata_form, "filter_predictions", "Filter Predictions",
            tooltip="Filter tracking IDs based on weighted probability of the final prediction"
        )

        lo, hi = _get_constraints("metadata.filter_predictions.min_prob_weighted")
        filter_predictions_form.add_parameter(create_num_param(
            "min_prob_weighted", "Minimum Probability",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.filter_predictions.min_prob_weighted,
            param_type="float",
            tooltip="Minimum required weighted probability to retain a tracking ID"
        ))

        size_form, self.estimate_size_box = create_param_form(
            self.metadata_form, "estimate_size", "Estimate Size",
            tooltip="Estimate physical size based on bounding box dimensions and frame size")

        lo, hi = _get_constraints("metadata.estimate_size.frame_width_mm")
        size_form.add_parameter(create_num_param(
            "frame_width_mm", "Frame Width (mm)",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.estimate_size.frame_width_mm,
            tooltip="Physical width of the frame (millimeters) - '1' returns relative size"
        ))

        lo, hi = _get_constraints("metadata.estimate_size.frame_height_mm")
        size_form.add_parameter(create_num_param(
            "frame_height_mm", "Frame Height (mm)",
            min_val=lo, max_val=hi,
            default=_DEFAULTS.metadata.estimate_size.frame_height_mm,
            tooltip="Physical height of the frame (millimeters) - '1' returns relative size"
        ))

        # Base tooltips kept for _sync_metadata_lock_state(), which appends a hint while locked
        self._metadata_boxes = [
            (self.filter_tracks_box,
             "Filter tracking IDs based on mean detection confidence and tracking duration"),
            (self.filter_predictions_box,
             "Filter tracking IDs based on weighted probability of the final prediction"),
            (self.estimate_size_box,
             "Estimate physical size based on bounding box dimensions and frame size"),
        ]

    @Slot(bool)
    def _on_classification_toggle(self, checked: bool) -> None:
        """Handle classification checkbox toggle.

        Enforces mutual exclusion between BioCLIP and Ultralytics,
        then updates the crop checkbox's auto-enable/lock state.

        Skipped while bulk-loading config values.
        """
        if self._loading_config:
            return

        sender = self.sender()

        # Handle mutual exclusion between classifiers
        if checked:
            if sender == self.bioclip_box.checkbox and self.ultralytics_box:
                self.ultralytics_box.checkbox.blockSignals(True)
                self.ultralytics_box.checkbox.setChecked(False)
                self.ultralytics_box.checkbox.blockSignals(False)
                self.ultralytics_box.frame.setEnabled(False)
            elif sender == self.ultralytics_box.checkbox and self.bioclip_box:
                self.bioclip_box.checkbox.blockSignals(True)
                self.bioclip_box.checkbox.setChecked(False)
                self.bioclip_box.checkbox.blockSignals(False)
                self.bioclip_box.frame.setEnabled(False)
        else:
            if sender == self.bioclip_box.checkbox and self.ultralytics_box:
                self.ultralytics_box.frame.setEnabled(True)
            elif sender == self.ultralytics_box.checkbox and self.bioclip_box:
                self.bioclip_box.frame.setEnabled(True)

        # Auto-enable and lock the crop checkbox
        self._sync_crop_lock_state()

        # Metadata processing aggregates classification results
        self._sync_metadata_lock_state()

    @Slot(bool)
    def _on_sort_mode_toggle(self, checked: bool) -> None:
        """Enforce mutual exclusion between 'Sort Crops' and 'Sort Tracks'.

        Skipped while bulk-loading config values.
        """
        if self._loading_config:
            return

        sender = self.sender()

        if checked:
            if sender == self.sort_crops_box.checkbox and self.sort_tracks_box:
                self.sort_tracks_box.checkbox.blockSignals(True)
                self.sort_tracks_box.checkbox.setChecked(False)
                self.sort_tracks_box.checkbox.blockSignals(False)
                self.sort_tracks_box.frame.setEnabled(False)
            elif sender == self.sort_tracks_box.checkbox and self.sort_crops_box:
                self.sort_crops_box.checkbox.blockSignals(True)
                self.sort_crops_box.checkbox.setChecked(False)
                self.sort_crops_box.checkbox.blockSignals(False)
                self.sort_crops_box.frame.setEnabled(False)
        else:
            if sender == self.sort_crops_box.checkbox and self.sort_tracks_box:
                self.sort_tracks_box.frame.setEnabled(True)
            elif sender == self.sort_tracks_box.checkbox and self.sort_crops_box:
                self.sort_crops_box.frame.setEnabled(True)

    def _sync_crop_lock_state(self) -> None:
        """Auto-enable and lock/unlock the crop checkbox based on classification state."""
        if not self.crop_box:
            return

        classification_enabled = self._is_classification_enabled
        already_locked = not self.crop_box.checkbox.isEnabled()

        if classification_enabled:
            if not self.crop_box.checkbox.isChecked():
                logger.info("Auto-enabling 'Crop Detections' (required for classification)")
                self.crop_box.checkbox.blockSignals(True)
                self.crop_box.checkbox.setChecked(True)
                self.crop_box.checkbox.blockSignals(False)

            self.crop_box.frame.setEnabled(self._can_classify_images)
            self.crop_box.checkbox.setEnabled(False)
            self.crop_box.setToolTip(
                "Save individual detections as separate .jpg files - cropped from original "
                "frames, or copied as-is for detections that only have an existing crop file"
                "\n\nRequired for classification - disable classification to unlock this setting"
            )
            if not already_locked:
                logger.debug("Crop checkbox locked (required by classification)")
        else:
            self.crop_box.checkbox.setEnabled(True)
            self.crop_box.setToolTip(
                "Save individual detections as separate .jpg files - cropped from original "
                "frames, or copied as-is for detections that only have an existing crop file"
            )
            if already_locked:
                logger.debug("Crop checkbox unlocked (no classification active)")

        self._on_crop_checkbox_toggle()

    def _sync_metadata_lock_state(self) -> None:
        """Lock/unlock the metadata settings based on classification state.

        Metadata processing aggregates per-image classification results into per-track
        results, so the pipeline skips the step entirely when no classifier is enabled.
        """
        classification_enabled = self._is_classification_enabled
        available = self._can_process_metadata and classification_enabled
        force_uncheck = self.dataset_context is not None

        for box, tooltip in self._metadata_boxes:
            if not box:
                continue

            was_checked = box.checkbox.isChecked()
            self._toggle_box(box, available, force_uncheck=force_uncheck)
            if was_checked and not box.checkbox.isChecked():
                logger.debug("Metadata setting auto-disabled (requires an enabled classifier)")

            box.setToolTip(tooltip if classification_enabled else (
                f"{tooltip}"
                "\n\nRequires an enabled classifier - metadata processing aggregates "
                "classification results"
            ))

    @Slot()
    def _on_crop_checkbox_toggle(self) -> None:
        """Handle crop checkbox toggle - update sort_crops/sort_tracks availability.

        Skipped while bulk-loading config values.
        """
        if self._loading_config:
            return

        if not self.crop_box:
            return

        sort_checkable = self.crop_box.checkbox.isChecked() and self._is_classification_enabled
        sort_available = sort_checkable and self._can_classify_images
        dataset_known = self.dataset_context is not None
        force_uncheck = not sort_checkable or dataset_known

        for box, label in [(self.sort_crops_box, "Sort crops"), (self.sort_tracks_box, "Sort tracks")]:
            if not box:
                continue

            box.checkbox.setEnabled(sort_available)
            if box.checkbox.isChecked():
                box.frame.setEnabled(sort_available)

            if not sort_available and force_uncheck and box.checkbox.isChecked():
                box.checkbox.blockSignals(True)
                box.checkbox.setChecked(False)
                box.checkbox.blockSignals(False)
                logger.debug("%s auto-disabled", label)

    @Slot(str)
    def _on_output_path_change(self, _new_path: str) -> None:
        """Handle output path selection - update UI state."""
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        """Update UI state based on lock state and dataset capabilities."""
        if self._is_ui_locked:
            self._disable_all_controls()
            return

        self._enable_controls_by_capability()

    def _disable_all_controls(self) -> None:
        """Disable all controls (used when UI is locked)."""
        self.paths_form.setEnabled(False)
        self.config_select.setEnabled(False)
        self.gpu_checkbox.setEnabled(False)
        self.processing_form.setEnabled(False)
        self.classification_form.setEnabled(False)
        self.metadata_form.setEnabled(False)

    def _enable_controls_by_capability(self) -> None:
        """Enable controls based on dataset capabilities."""
        # Capability flags default to False before a dataset is inspected
        dataset_known = self.dataset_context is not None

        # Main controls (always enabled when unlocked)
        self.paths_form.setEnabled(True)
        self.config_select.setEnabled(True)
        if self._cuda_available:
            self.gpu_checkbox.setEnabled(True)

        # Classification controls - resolved first, since crop's lock state and its own
        # capability gate below both depend on the post-toggle classification state
        self.classification_form.setEnabled(True)
        for box in [self.bioclip_box, self.ultralytics_box]:
            self._toggle_box(box, self._can_classify_images, force_uncheck=dataset_known)
        self._sync_crop_lock_state()

        # Image processing controls
        self.processing_form.setEnabled(True)
        if not self._is_classification_enabled:
            self._toggle_box(self.crop_box, self._can_process_images, force_uncheck=dataset_known)
        self._toggle_box(self.overlay_box, self._can_process_images, force_uncheck=dataset_known)

        # Metadata controls - also gated on classification, whose results they aggregate
        self.metadata_form.setEnabled(True)
        self._sync_metadata_lock_state()

        # Update sort_crops/sort_tracks availability based on crop checkbox state
        self._on_crop_checkbox_toggle()

        # Run button
        can_run = self._can_process_images or self._can_classify_images or self._can_process_metadata
        self.run_enabled_changed.emit(
            can_run and self.dataset_context is not None and self.output_path is not None
        )

    def _toggle_box(self, box: Any | None, enabled: bool, force_uncheck: bool = True) -> None:
        """Enable or disable a parameter box (checkbox + form).

        Args:
            box: Parameter box to toggle, no-op if None.
            enabled: Whether the box should be enabled.
            force_uncheck: Whether to uncheck when disabling.
        """
        if box is None:
            return

        if not enabled and force_uncheck:
            box.checkbox.blockSignals(True)
            box.checkbox.setChecked(False)
            box.checkbox.blockSignals(False)

        box.checkbox.setEnabled(enabled)
        box.frame.setEnabled(enabled and box.checkbox.isChecked())

    def lock_ui(self) -> None:
        """Lock UI components in config widget."""
        self._is_ui_locked = True
        self._update_ui_state()
        logger.debug("Config widget locked")

    def unlock_ui(self) -> None:
        """Unlock UI components in config widget."""
        self._is_ui_locked = False
        self._update_ui_state()
        logger.debug("Config widget unlocked")

    @Slot(str)
    def _on_config_change(self, config_name: str) -> None:
        """Switch to selected config file and activate config parameters."""
        if config_name == self.config_active:
            return

        if self._config and check_config_changes(self._config, self.sync_config()):
            reply = QMessageBox.warning(
                self, "Unsaved Changes",
                "You have unsaved configuration changes!\n"
                "Do you want to save them before switching?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Cancel:
                self.config_select.setCurrentText(self.config_active)
                return
            if reply == QMessageBox.StandardButton.Save and not self.save_config():
                self.config_select.setCurrentText(self.config_active)
                return

        # Reset last inspected path to ensure dataset inspection runs for the new config
        self._last_inspected_path = None
        self._activate_config(config_name)

    def _load_config(self, config_name: str, loaded_config: AppConfig | None = None) -> None:
        """Load configuration parameters from config file and apply them to the UI."""
        if loaded_config is None:
            config_path = CONFIGS_PATH / config_name
            if not config_path.exists():
                QMessageBox.warning(self, "Config Not Found", f"Config file '{config_name}' not found!")
                return

            try:
                loaded_config = load_config_yaml(config_path)
            except Exception as e:
                logger.exception("Failed to load config '%s'", config_name)
                QMessageBox.critical(self, "Invalid Config", str(e))
                return

        self._config = loaded_config
        self._config_updates = self._config.model_dump()

        self.paths_form.set_values({
            "source_path": self._config_updates.get("source_path") or "",
            "output_path": self._config_updates.get("output_path") or ""
        })

        config_device = self._config_updates.get("device", "cpu")
        if self._cuda_available is not None:
            if config_device == "cuda" and not self._cuda_available:
                logger.warning("Config specifies 'cuda' but GPU/CUDA is not available")
                QMessageBox.warning(
                    self, "GPU/CUDA Not Available",
                    "Configuration is set to use GPU ('cuda'), but CUDA is not available.\n\n"
                    "Device has been automatically changed to 'cpu'.\n"
                    "Please install PyTorch with CUDA support if you want to use GPU acceleration."
                )
                self._config_updates["device"] = "cpu"
                config_device = "cpu"
            elif config_device == "cpu" and self._cuda_available:
                logger.info("GPU/CUDA is available but config uses 'cpu'")
                QMessageBox.information(
                    self, "GPU/CUDA Available",
                    "CUDA-compatible GPU detected!\n\n"
                    "Your configuration is currently set to use CPU.\n"
                    "Enable GPU acceleration to significantly increase classification speed."
                )

        self.gpu_checkbox.setChecked(config_device == "cuda")

        # Populate parameter forms with enabled values from the loaded config
        processing_values = extract_enabled_values(self._config_updates.get("processing", {}))
        classification_values = extract_enabled_values(self._config_updates.get("classification", {}))
        metadata_values = extract_enabled_values(self._config_updates.get("metadata", {}))

        # Suppress cascade handlers while bulk-populating values
        self._loading_config = True
        try:
            self.processing_form.set_values(processing_values)
            self.classification_form.set_values(classification_values)
            self.metadata_form.set_values(metadata_values)
        finally:
            self._loading_config = False

        if self._is_classification_enabled:
            self._on_classification_toggle(True)
        else:
            self._on_crop_checkbox_toggle()

        source_path_str = self._config_updates.get("source_path") or ""
        source_dir = resolve_and_validate_path(source_path_str)

        # Only start dataset inspection if path is different from last inspected source
        if source_dir and self._last_inspected_path != source_dir:
            self._start_dataset_inspection(source_dir)
            return

        if not source_dir and self.dataset_context is not None:
            logger.info("Clearing cached dataset context (no valid source path in loaded config)")
            self._reset_dataset_capabilities()

        # Update UI state immediately if no source path or same path with existing context
        self._update_ui_state()

    @Slot()
    def reset_to_defaults(self) -> None:
        """Reset all parameters to their defaults, preserving source/output paths."""
        defaults = _DEFAULTS.model_dump()
        self.gpu_checkbox.setChecked(False)

        self._loading_config = True
        try:
            self.processing_form.set_values(extract_enabled_values(defaults.get("processing", {})))
            self.classification_form.set_values(extract_enabled_values(defaults.get("classification", {})))
            self.metadata_form.set_values(extract_enabled_values(defaults.get("metadata", {})))
        finally:
            self._loading_config = False

        if self._is_classification_enabled:
            self._on_classification_toggle(True)
        else:
            self._on_crop_checkbox_toggle()

    def sync_config(self) -> dict[str, Any]:
        """Sync live form values into _config_updates and return the result."""
        self._config_updates["source_path"] = self.source_path.as_posix() if self.source_path else None
        self._config_updates["output_path"] = self.output_path.as_posix() if self.output_path else None
        self._config_updates["device"] = self.device
        self._config_updates["processing"] = restore_enabled_values(self.processing_form.values())
        self._config_updates["classification"] = restore_enabled_values(self.classification_form.values())
        self._config_updates["metadata"] = restore_enabled_values(self.metadata_form.values())
        return self._config_updates

    def has_unsaved_changes(self) -> bool:
        """Return True if the current form state differs from the last saved config."""
        if not self._config:
            return False
        return check_config_changes(self._config, self.sync_config())

    def save_to_active_file(self) -> bool:
        """Save current configuration directly to the active config file without prompting."""
        return self._save_to_file(CONFIGS_PATH / self.config_active)

    def save_config(self) -> bool:
        """Save modified configuration parameters to config file."""
        if not self._config:
            return False

        if not check_config_changes(self._config, self.sync_config()):
            QMessageBox.information(
                self, "No Changes",
                "Configuration has not been modified!"
            )
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Save Configuration")
        msg.setText(f"Save changes to '{self.config_active}'?")

        save_btn = msg.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        save_as_btn = msg.addButton("Save As...", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        clicked = msg.clickedButton()

        if clicked == save_btn:
            return self._save_to_file(CONFIGS_PATH / self.config_active)
        if clicked == save_as_btn:
            return self._create_new_config()

        return False

    def _save_to_file(self, config_path: Path) -> bool:
        """Save configuration to specified file path."""
        try:
            validated_config = update_config_yaml(config_path, self.sync_config())
            logger.info("Configuration saved to '%s'", config_path.name)
            self._activate_config(config_path.name, loaded_config=validated_config)
            return True
        except Exception:
            logger.exception("Failed to save configuration to '%s'", config_path.name)
            QMessageBox.critical(
                self, "Save Error",
                "Failed to save configuration!\n\nCheck the log output for details."
            )
            return False

    def _prompt_for_new_filename(self) -> str | None:
        """Prompt for a valid new config filename, returning None if cancelled."""
        while True:
            filename, ok = QInputDialog.getText(
                self, "New Config File",
                "Save config file as:",
                text="config_custom"
            )
            filename = filename.strip()

            if not ok or not filename:
                return None

            if not all(c.isalnum() or c in "_-" for c in filename):
                QMessageBox.warning(
                    self, "Invalid Filename",
                    "Filename can only contain letters, numbers, hyphens, and underscores.\n"
                    "Please try again."
                )
                continue

            if filename == "config_selector":
                QMessageBox.warning(
                    self, "Cannot Overwrite",
                    "Cannot overwrite config selector!\n"
                    "Please choose a different name."
                )
                continue

            return filename

    def _create_new_config(self) -> bool:
        """Save modified config parameters to a new file."""
        while True:
            filename = self._prompt_for_new_filename()
            if filename is None:
                QMessageBox.warning(self, "Cancelled", "New config file not saved!")
                return False

            config_new_path = CONFIGS_PATH / f"{filename}.yaml"
            if config_new_path.exists():
                reply = QMessageBox.warning(
                    self, "File Exists",
                    f"File '{filename}.yaml' already exists.\n"
                    "Do you want to overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    continue

            return self._save_to_file(config_new_path)

    def _activate_config(self, config_name: str, loaded_config: AppConfig | None = None) -> None:
        """Activate the specified config file and update config selector."""
        self._configs = sorted([f.name for f in CONFIGS_PATH.glob("*.yaml")
                               if f.name != CONFIG_SELECTOR_PATH.name])
        self.config_select.blockSignals(True)
        self.config_select.clear()
        self.config_select.addItems(self._configs)
        self.config_select.setCurrentText(config_name)
        self.config_select.blockSignals(False)

        update_config_selector(config_name)

        self.config_active = config_name
        self._load_config(self.config_active, loaded_config=loaded_config)

        logger.info("Configuration '%s' activated", config_name)

    @Slot(str)
    def _on_source_path_change(self, new_path: str) -> None:
        """Handle source path selection - trigger background dataset inspection."""
        if not new_path:
            return

        source_dir = resolve_and_validate_path(new_path)
        if not source_dir:
            logger.error("Selected path does not exist or is not a directory: %s", new_path)
            return

        logger.debug("Source path changed to: %s", source_dir)
        self._start_dataset_inspection(source_dir)

    def cancel_inspection(self, timeout_ms: int = 1000) -> None:
        """Cancel any running dataset inspection and wait for the thread to stop."""
        if self._inspector_thread and self._inspector_thread.isRunning():
            if self._dataset_inspector:
                self._dataset_inspector.cancel()
            self._cleanup_inspection_thread(timeout_ms)
            self.progress_updated.emit(0)
            self.status_updated.emit("Inspection cancelled")
            self.inspection_active_changed.emit(False)

    def _cleanup_inspection_thread(self, timeout_ms: int = 1000) -> None:
        """Stop and delete the inspector thread/worker, releasing their Qt-side resources."""
        if self._inspector_thread:
            self._inspector_thread.quit()
            if not self._inspector_thread.wait(timeout_ms):
                logger.warning("Inspector thread not responding, forcing termination...")
                self._inspector_thread.terminate()
                self._inspector_thread.wait(500)
            self._inspector_thread.deleteLater()
            self._inspector_thread = None

        if self._dataset_inspector:
            self._dataset_inspector.deleteLater()
            self._dataset_inspector = None

    def _reset_dataset_capabilities(self) -> None:
        """Reset cached dataset context and capability flags to their initial state."""
        self.dataset_context = None
        self._last_inspected_path = None
        self._can_process_images = False
        self._can_classify_images = False
        self._can_process_metadata = False

    def _start_dataset_inspection(self, data_dir: Path) -> None:
        """Start background dataset inspection."""
        if self._last_inspected_path == data_dir:
            logger.debug("Skipping re-inspection of same path: %s", data_dir)
            return

        # Cancel any already running inspection before starting a new one
        self.cancel_inspection(timeout_ms=2000)

        # Clear stale capabilities from any previously inspected dataset
        self._reset_dataset_capabilities()

        self._last_inspected_path = data_dir
        self._inspector_thread = QThread()
        self._dataset_inspector = DatasetInspector(data_dir)
        self._dataset_inspector.moveToThread(self._inspector_thread)

        # Connect signals
        self._inspector_thread.started.connect(self._dataset_inspector.run)
        self._dataset_inspector.finished.connect(self._on_inspection_finish)
        self._dataset_inspector.error.connect(self._on_inspection_error)
        self.progress_updated.emit(0)
        self.status_updated.emit("Inspecting dataset...")
        self.inspection_active_changed.emit(True)
        self._dataset_inspector.progress.connect(self.progress_updated)
        self._dataset_inspector.progress_message.connect(self.status_updated)

        # Start dataset inspection
        logger.info("Starting dataset inspection...")
        self._inspector_thread.start()

    @Slot(object)
    def _on_inspection_finish(self, context: DatasetContext) -> None:
        """Handle completed dataset inspection: log results and update UI state."""
        # Log summary
        logger.info("\n%s", context.get_summary_str(include_details=True))

        # Update capabilities
        self.dataset_context = context
        has_detection_frames = len(context.detection_frames) > 0
        has_crop_frames = len(context.crop_frames) > 0
        has_metadata = len(context.metadata_files) > 0
        self._can_process_images = has_detection_frames
        self._can_classify_images = (has_detection_frames or has_crop_frames) and has_metadata
        self._can_process_metadata = has_metadata

        if not self._can_process_images:
            if has_crop_frames:
                logger.info("Crops-only dataset - image processing disabled, classification available")
            else:
                logger.warning("No detection frames found - image processing disabled")
        if not self._can_classify_images:
            if not has_metadata:
                logger.warning("No metadata found - classification disabled")
            else:
                logger.warning("No detection or crop frames - classification disabled")
        if not self._can_process_metadata:
            logger.warning("No metadata found - metadata processing disabled")

        # Update progress label
        if context.has_metadata:
            frame_parts = [f"{context.total_detection_frames} detection frames"]
            if context.num_crop_frames > 0:
                frame_parts.append(f"{context.num_crop_frames} crops")
            self.status_updated.emit("Inspection complete: " + ", ".join(frame_parts))
        else:
            self.status_updated.emit(
                f"Inspection complete: {context.total_timelapse_frames} timelapse frames (no metadata)"
            )

        self.inspection_active_changed.emit(False)
        self._cleanup_inspection_thread()
        self._update_ui_state()

    @Slot(str)
    def _on_inspection_error(self, error_msg: str) -> None:
        """Handle dataset inspection error."""
        logger.error("=" * 60)
        logger.error("Dataset Inspection Failed")
        logger.error("Error: %s", error_msg)
        logger.error("=" * 60)

        self._reset_dataset_capabilities()

        self.progress_updated.emit(0)
        self.status_updated.emit(f"Inspection failed: {error_msg}")
        self.inspection_active_changed.emit(False)

        self._cleanup_inspection_thread()
        self._update_ui_state()


class MainWindow(QMainWindow):
    """Main application window with pipeline controls and log viewer."""
    config_widget: ConfigWidget
    save_btn: StyledButton
    reset_btn: StyledButton
    run_btn: StyledButton
    cancel_btn: StyledButton
    update_btn: StyledButton
    theme_select: QComboBox
    progress_bar: QProgressBar
    progress_label: QLabel
    log_viewer: LogViewer

    def __init__(self) -> None:
        """Initialize the main application window."""
        super().__init__()
        self._pipeline_thread: QThread | None = None
        self._pipeline_runner: PipelineRunner | None = None
        self._update_thread: QThread | None = None
        self._update_worker: UpdateChecker | UpdateInstaller | None = None
        self._update_info: UpdateInfo | None = None
        self._update_available: bool = False
        self._inspection_active: bool = False

        # Connect root logger to LogCache
        self._log_cache: LogCache = LogCache()
        self._log_cache.connect_logger(logging.getLogger())

        # Set main window properties
        self.setWindowTitle("Insect Detect Post-Processing")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.7)
        h = int(screen.height() * 0.8)
        self.resize(w, h)
        icon_path = BASE_PATH / "resources" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create UI layout
        self._create_ui_layout()

        # Check for updates once the main window is displayed
        QTimer.singleShot(1000, self._start_update_check)

    @property
    def _is_pipeline_running(self) -> bool:
        """Check if pipeline is currently running."""
        return (
            self._pipeline_thread is not None
            and self._pipeline_thread.isRunning()
        )

    def _create_ui_layout(self) -> None:
        """Create UI layout for the main application window."""

        # Main vertical layout for the central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Add config widget
        self.config_widget = ConfigWidget()
        self.config_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(self.config_widget, 0)

        # Create control buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(6, 0, 6, 0)
        button_layout.setSpacing(10)

        self.save_btn = StyledButton("Save Config", COLOR_BTN_SAVE)
        self.reset_btn = StyledButton("Reset to Defaults", COLOR_BTN_RESET)
        self.run_btn = StyledButton("Run Processing", COLOR_BTN_RUN)
        self.cancel_btn = StyledButton("Cancel", COLOR_BTN_CANCEL)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        self.save_btn.clicked.connect(self.save_config)
        self.reset_btn.clicked.connect(self.config_widget.reset_to_defaults)
        self.run_btn.clicked.connect(self.run_pipeline)
        self.cancel_btn.clicked.connect(self.cancel_running_task)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()

        # Update button (right-aligned, hidden until an update is available)
        self.update_btn = StyledButton("Update available", COLOR_BTN_UPDATE)
        self.update_btn.setToolTip("Install the latest version from GitHub")
        self.update_btn.setVisible(False)
        self.update_btn.clicked.connect(self._on_update_click)
        button_layout.addWidget(self.update_btn)

        # Theme selector (right-aligned in the control button row)
        self.theme_select = QComboBox()
        self.theme_select.addItems(sorted(qt_themes.get_themes()))
        self.theme_select.setCurrentText(_load_theme())
        self.theme_select.setToolTip("Select the color theme of the application")
        self.theme_select.setFixedWidth(150)
        self.theme_select.currentTextChanged.connect(self._on_theme_change)
        button_layout.addWidget(QLabel("Theme:"))
        button_layout.addWidget(self.theme_select)
        main_layout.addLayout(button_layout, 0)

        # Create progress bar and label
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(6, 6, 6, 0)
        progress_layout.setSpacing(2)
        self.progress_bar = QProgressBar()
        self._style_progress_bar()
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout, 0)

        # Create log viewer
        self.log_viewer = LogViewer(self._log_cache)
        self.log_viewer.set_levels([logging.ERROR, logging.WARNING, logging.INFO])
        self.log_viewer.setMinimumHeight(100)
        self.log_viewer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.log_viewer, 1)

        # Connect ConfigWidget signals to MainWindow widgets
        self.config_widget.progress_updated.connect(self.progress_bar.setValue)
        self.config_widget.status_updated.connect(self.progress_label.setText)
        self.config_widget.run_enabled_changed.connect(self.run_btn.setEnabled)
        self.config_widget.inspection_active_changed.connect(self._on_inspection_active_change)

    def _style_progress_bar(self) -> None:
        """Apply the progress bar stylesheet with colors from the active theme."""
        theme = qt_themes.get_theme() or qt_themes.get_theme(THEME_DEFAULT)
        if theme is None:
            return
        border = theme.crust if theme.is_dark_theme() else theme.surface2
        background, text = theme.base, theme.text
        if border is None or background is None or text is None:
            return
        self.progress_bar.setStyleSheet(PROG_BAR_STYLESHEET.format(
            border=border.name(), background=background.name(), text=text.name()
        ))

    def _repolish_styled_widgets(self) -> None:
        """Re-apply stylesheets so styled widgets pick up the palette of the active theme."""
        for widget in self.findChildren(QWidget):
            if widget.styleSheet() and widget is not self.progress_bar:
                widget.setStyleSheet(widget.styleSheet())

    @Slot(str)
    def _on_theme_change(self, theme_name: str) -> None:
        """Apply the selected color theme to the application and persist the selection."""
        qt_themes.set_theme(theme_name, style=None)
        self._repolish_styled_widgets()
        self._style_progress_bar()
        _save_theme(theme_name)

    @Slot(bool)
    def _on_inspection_active_change(self, active: bool) -> None:
        """Enable cancelling a running dataset inspection and block updates while it runs."""
        self._inspection_active = active
        self.cancel_btn.setEnabled(active)
        self._sync_update_btn()

    def _set_update_available(self, available: bool) -> None:
        """Show or hide the update button."""
        self._update_available = available
        self.update_btn.setVisible(available)
        self._sync_update_btn()

    def _sync_update_btn(self) -> None:
        """Enable the update button only if an update is available and no task is running."""
        self.update_btn.setEnabled(self._update_available and not self._inspection_active
                                   and not self._is_pipeline_running)

    def _set_controls_locked(self, locked: bool) -> None:
        """Lock or unlock all controls while an update is installed/finished/failed."""
        self.save_btn.setEnabled(not locked)
        self.reset_btn.setEnabled(not locked)
        if locked:
            self.run_btn.setEnabled(False)
            self.update_btn.setEnabled(False)
            self.config_widget.lock_ui()
        else:
            # Unlocking re-emits run_enabled_changed, which restores the run button
            self.config_widget.unlock_ui()
            self._sync_update_btn()

    def _start_update_worker(self, worker: UpdateChecker | UpdateInstaller) -> None:
        """Move an update worker to its own thread and start it."""
        thread = QThread()
        self._update_thread = thread
        self._update_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()

    def _cleanup_update_thread(self) -> None:
        """Stop and delete the update thread/worker, releasing their Qt-side resources."""
        if self._update_thread:
            self._update_thread.quit()
            if not self._update_thread.wait(5000):
                logger.warning("Update thread not responding, forcing termination...")
                self._update_thread.terminate()
                self._update_thread.wait(500)
            self._update_thread.deleteLater()
            self._update_thread = None

        if self._update_worker:
            self._update_worker.deleteLater()
            self._update_worker = None

    def _start_update_check(self) -> None:
        """Check for available updates in the background."""
        checker = UpdateChecker()
        checker.finished.connect(self._on_update_check_finish)
        checker.error.connect(self._on_update_check_error)
        self._start_update_worker(checker)

    @Slot(object)
    def _on_update_check_finish(self, info: UpdateInfo | None) -> None:
        """Show the update button if an update is available."""
        self._cleanup_update_thread()
        self._update_info = info
        if info is None:
            logger.debug("No updates available")
            return

        logger.info("Update available: %d new commit(s) on GitHub", len(info.commits))
        self._set_update_available(True)

    @Slot(str)
    def _on_update_check_error(self, error_msg: str) -> None:
        """Log a failed update check without interruption."""
        self._cleanup_update_thread()
        logger.debug("Update check skipped: %s", error_msg)

    @Slot()
    def _on_update_click(self) -> None:
        """Show the available update and install it if confirmed."""
        info = self._update_info
        if info is None:
            return

        if info.version_local and info.version_remote and info.version_local != info.version_remote:
            text = f"Insect Detect Post {info.version_local} → {info.version_remote}"
        else:
            text = "A new version of Insect Detect Post is available."

        informative = (f"{len(info.commits)} new commit(s) will be applied.\n\n"
                       "Your configuration and profiles are not affected.")
        if info.local_changes:
            informative += (f"\n\nYour local changes to {len(info.local_changes)} file(s) will be "
                            "stashed and restored afterwards.")

        details = "\n".join([*info.commits, "",
                             *(f"{status:<8} {path}" for status, path in info.changes)])

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Available")
        msg.setText(text)
        msg.setInformativeText(informative)
        msg.setDetailedText(details)
        install_btn = msg.addButton("Install Update", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(install_btn)
        msg.exec()

        if msg.clickedButton() is install_btn:
            self._install_update(info)

    def _install_update(self, info: UpdateInfo) -> None:
        """Lock the interface and install the update in the background."""
        logger.info("Installing update...")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Installing update...")
        self._set_controls_locked(True)

        installer = UpdateInstaller(info)
        installer.finished.connect(self._on_update_install_finish)
        installer.error.connect(self._on_update_install_error)
        self._start_update_worker(installer)

    def _reset_progress(self, status: str) -> None:
        """Return the progress bar to its determinate state and show a status message."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText(status)

    @Slot()
    def _on_update_install_finish(self) -> None:
        """Prompt for a restart after the update was installed."""
        self._cleanup_update_thread()
        self._reset_progress("Update installed")
        self._set_controls_locked(False)
        self._set_update_available(False)
        logger.info("Update complete!")

        info, self._update_info = self._update_info, None
        version = f" to version {info.version_remote}" if info and info.version_remote else ""

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Installed")
        msg.setText(f"Insect Detect Post was updated{version}.")

        if info and info.needs_sync:
            command = sync_command()
            msg.setInformativeText(
                "This update changed the dependencies. Close the application and run the "
                f"following command to finish the update:\n\n{command}"
            )
            copy_btn = msg.addButton("Copy Command", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            if msg.clickedButton() is copy_btn:
                QApplication.clipboard().setText(command)
                logger.info("Copied '%s' to the clipboard", command)
        else:
            msg.setInformativeText("Restart the application to use the new version.")
            restart_btn = msg.addButton("Restart Now", QMessageBox.ButtonRole.AcceptRole)
            msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(restart_btn)
            msg.exec()
            if msg.clickedButton() is restart_btn:
                self._restart_application()

    @Slot(str)
    def _on_update_install_error(self, error_msg: str) -> None:
        """Report a failed update installation."""
        self._cleanup_update_thread()
        self._reset_progress("Update failed")
        self._set_controls_locked(False)
        logger.error(error_msg)
        QMessageBox.critical(self, "Update Failed", error_msg)

    def _restart_application(self) -> None:
        """Start a new instance of the application and close the current one."""
        try:
            process = subprocess.Popen([sys.executable, "-m", "insectdetect_post.post_processing_gui"],
                                       cwd=BASE_PATH)
        except OSError:
            logger.exception("Could not restart the application")
            QMessageBox.warning(self, "Restart Failed",
                                "Could not restart automatically. Please start the application again.")
            return

        # Stop the new instance again if closing the current one was cancelled
        if not self.close():
            logger.info("Restart cancelled.")
            process.terminate()

    def _validate_paths(self, config: dict[str, Any]) -> str | None:
        """Validate source and output paths from config.

        Args:
            config: Configuration dictionary.

        Returns:
            Error message string if validation fails, None if paths are valid.
        """
        source_path_str = config.get("source_path", "")
        if not source_path_str or not source_path_str.strip():
            return "No source path specified in configuration"

        source_dir = resolve_and_validate_path(source_path_str)
        if not source_dir:
            return f"Invalid source directory:\n{source_path_str}"

        output_path_str = config.get("output_path", "")
        if not output_path_str or not output_path_str.strip():
            return "No output path specified in configuration"

        output_dir = resolve_and_validate_path(output_path_str)
        if not output_dir:
            return f"Invalid output directory:\n{output_path_str}"

        if output_dir == source_dir or output_dir.is_relative_to(source_dir):
            return "Output directory cannot be nested in the source directory"

        return None

    @Slot()
    def save_config(self) -> None:
        """Save configuration (feedback handled by ConfigWidget)."""
        self.config_widget.save_config()

    @Slot()
    def run_pipeline(self) -> None:
        """Run the processing pipeline with cached context."""
        if not self.config_widget.dataset_context:
            QMessageBox.warning(self, "No Dataset", "Please select a dataset directory first.")
            return

        logger.info("Initializing pipeline...")
        self.progress_label.setText("Initializing pipeline...")
        self.progress_bar.setValue(0)

        if self._is_pipeline_running:
            logger.warning("Pipeline already running!")
            return

        # Clean up any leftover threads from previous runs
        if self._pipeline_thread:
            self._pipeline_thread.quit()
            self._pipeline_thread.wait(1000)
            self._pipeline_thread.deleteLater()
            self._pipeline_thread = None
        if self._pipeline_runner:
            self._pipeline_runner.deleteLater()
            self._pipeline_runner = None

        # Prompt to save only if there are unsaved changes
        if self.config_widget.has_unsaved_changes():
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Unsaved Changes")
            msg.setText("You have unsaved configuration changes.")
            save_btn = msg.addButton("Save && Run", QMessageBox.ButtonRole.AcceptRole)
            msg.addButton("Run without saving", QMessageBox.ButtonRole.ActionRole)
            msg.addButton(QMessageBox.StandardButton.Cancel)
            msg.exec()
            clicked = msg.clickedButton()

            if msg.buttonRole(clicked) == QMessageBox.ButtonRole.RejectRole:
                return
            if clicked == save_btn and not self.config_widget.save_to_active_file():
                return

        config_dict = self.config_widget.sync_config()
        try:
            config = AppConfig.model_validate(config_dict)
        except Exception as e:
            QMessageBox.critical(self, "Invalid Configuration",
                                 f"Configuration validation failed:\n\n{e}")
            logger.exception("Configuration validation failed")
            return

        # Validate source and output paths
        error_msg = self._validate_paths(config_dict)

        if error_msg:
            QMessageBox.critical(self, "Invalid Path", error_msg)
            logger.error(error_msg.replace('\n', ' '))
            return

        # Create pipeline thread
        self._pipeline_thread = QThread()

        # Get cached context from config widget
        context = self.config_widget.dataset_context

        if context is None:
            logger.warning("No cached dataset context, will scan during pipeline run")

        # Create pipeline runner
        config_stem = Path(self.config_widget.config_active).stem
        self._pipeline_runner = PipelineRunner(
            config,
            config_stem=config_stem,
            context=context
        )
        self._pipeline_runner.moveToThread(self._pipeline_thread)

        # Connect signals
        self._pipeline_thread.started.connect(self._pipeline_runner.run)
        self._pipeline_runner.progress.connect(self.progress_bar.setValue)
        self._pipeline_runner.progress_message.connect(self.progress_label.setText)
        self._pipeline_runner.finished.connect(self._on_pipeline_finish)
        self._pipeline_runner.error.connect(self._on_pipeline_error)

        # Update UI state
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
        self.config_widget.lock_ui()

        # Start pipeline
        self._pipeline_thread.start()
        logger.info("Starting pipeline...")

    @Slot()
    def cancel_running_task(self) -> None:
        """Cancel whichever task is currently running: pipeline or dataset inspection."""
        self.cancel_btn.setEnabled(False)

        if self._pipeline_runner and self._is_pipeline_running:
            self.progress_label.setText("Cancelling pipeline...")
            self._pipeline_runner.cancel()
        else:
            self.progress_label.setText("Cancelling inspection...")
            self.config_widget.cancel_inspection()

    @Slot(str)
    def _on_pipeline_error(self, error_msg: str) -> None:
        """Handle pipeline errors."""
        if error_msg != "Pipeline cancelled":
            logger.error(error_msg)
            QMessageBox.critical(self, "Pipeline Error", error_msg)

        # Always trigger cleanup
        self._on_pipeline_finish()

    @Slot()
    def _on_pipeline_finish(self) -> None:
        """Handle pipeline completion with comprehensive cleanup."""
        logger.info("Cleaning up pipeline...")

        # Clean up thread
        if self._pipeline_thread:
            self._pipeline_thread.quit()

            if not self._pipeline_thread.wait(3000):
                logger.warning("Thread not responding, forcing termination...")
                self._pipeline_thread.terminate()

                if not self._pipeline_thread.wait(1000):
                    logger.error("Thread could not be terminated!")

            # Delete thread object
            self._pipeline_thread.deleteLater()
            self._pipeline_thread = None

        # Clean up pipeline runner
        if self._pipeline_runner:
            self._pipeline_runner.deleteLater()
            self._pipeline_runner = None

        # Reset UI state
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self._sync_update_btn()
        self.config_widget.unlock_ui()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        logger.info("Pipeline stopped")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event with pipeline cleanup."""

        # Never interrupt a running update, as this could leave a partially updated installation
        if isinstance(self._update_worker, UpdateInstaller):
            QMessageBox.information(self, "Update Running",
                                    "The update is still being installed. Please wait until it is finished.")
            event.ignore()
            return

        # Clean up update and inspector threads if running
        self._cleanup_update_thread()
        self.config_widget.cancel_inspection()

        if self._is_pipeline_running:
            reply = QMessageBox.question(
                self, "Pipeline Running",
                "Pipeline is still running. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Disconnect signals first to prevent callbacks during shutdown
                try:
                    if self._pipeline_runner:
                        self._pipeline_runner.progress.disconnect()
                        self._pipeline_runner.progress_message.disconnect()
                        self._pipeline_runner.finished.disconnect()
                        self._pipeline_runner.error.disconnect()
                except (RuntimeError, AttributeError):
                    pass

                if self._pipeline_runner:
                    self._pipeline_runner.cancel()

                if self._pipeline_thread:
                    self._pipeline_thread.quit()
                    if not self._pipeline_thread.wait(2000):
                        logger.warning("Forcing pipeline termination on exit...")
                        self._pipeline_thread.terminate()
                        self._pipeline_thread.wait(500)

                # Clean up objects
                if self._pipeline_runner:
                    self._pipeline_runner.deleteLater()
                if self._pipeline_thread:
                    self._pipeline_thread.deleteLater()

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main() -> None:
    """Main entry point for the post-processing GUI application."""

    # Set logging levels for root logger and insectdetect_post logger
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("insectdetect_post").setLevel(logging.DEBUG)

    # Create config files with default values if they are missing
    try:
        ensure_config_files()
    except OSError:
        logger.exception("Could not create config files in %s", CONFIGS_PATH)
        sys.exit(1)

    if not MODELS_PATH.exists():
        logger.error("Model directory not found at %s", MODELS_PATH)
        sys.exit(1)

    # Start Qt application
    app = QApplication(sys.argv)
    qt_themes.set_theme(_load_theme())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

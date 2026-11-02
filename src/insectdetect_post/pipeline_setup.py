"""Prepare a processing run: check the paths, adapt the settings to the dataset, check the result.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Holds the rules that decide which settings of a config can run on a given dataset, so every
caller runs a config with the same settings and gets the same output. The GUI settings panel
applies the same rules while it is edited. adapt_run_config() applies them to a config file.

Classes:
    DatasetCapabilities: Processing steps a dataset supports, derived from its inspection.

Functions:
    resolve_and_validate_path(): Resolve a path string to an existing directory.
    validate_paths(): Check the source and output directories of a run.
    adapt_run_config(): Adapt a config to what a dataset supports.
    available_models(): Return the model files that can be selected in a config.
    cuda_available(): Return True if PyTorch is installed with CUDA support and a GPU is available.
    check_run_config(): Check what would otherwise only fail once the run has started.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insectdetect_post.asset_manager import list_registered_filenames
from insectdetect_post.config import AppConfig
from insectdetect_post.constants import MODELS_JSON, MODELS_PATH
from insectdetect_post.dataset_context import DatasetContext

# Create module-level logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetCapabilities:
    """Processing steps a dataset supports, derived from its inspection.

    - can_process_images:   detection frames to crop and draw overlays on
    - can_classify_images:  detection frames or crops, and metadata to link them to
    - can_process_metadata: metadata to aggregate
    - can_run:              a run is possible at all (every run starts from the merged metadata)
    """
    can_process_images: bool
    can_classify_images: bool
    can_process_metadata: bool

    @classmethod
    def from_context(cls, context: DatasetContext) -> DatasetCapabilities:
        """Derive the capabilities from a dataset inspection."""
        has_detection_frames = len(context.detection_frames) > 0
        has_crop_frames = len(context.crop_frames) > 0
        has_metadata = len(context.metadata_files) > 0
        return cls(
            can_process_images=has_detection_frames,
            can_classify_images=(has_detection_frames or has_crop_frames) and has_metadata,
            can_process_metadata=has_metadata,
        )

    @property
    def can_run(self) -> bool:
        """Return True if a run is possible: every run starts by merging the metadata files."""
        return self.can_process_metadata


def resolve_and_validate_path(path_str: str | None) -> Path | None:
    """Resolve and validate a path string.

    Strips surrounding whitespace and double quotes (e.g. from a path pasted from the
    file manager) and resolves relative paths against the current working directory.

    Args:
        path_str: Path string to validate.

    Returns:
        Resolved Path object if valid directory, None otherwise.
    """
    if not path_str or not path_str.strip():
        return None

    stripped = path_str.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        stripped = stripped[1:-1].strip()

    path_obj = Path(stripped)
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()

    return path_obj if path_obj.exists() and path_obj.is_dir() else None


def validate_paths(source_path: str | None, output_path: str | None) -> str | None:
    """Check the source and output directories of a run.

    Args:
        source_path: Source directory as given in the config.
        output_path: Output directory as given in the config.

    Returns:
        Error message string if validation fails, None if paths are valid.
    """
    if not source_path or not source_path.strip():
        return "No source path specified in the config"

    source_dir = resolve_and_validate_path(source_path)
    if not source_dir:
        return f"Invalid source directory:\n{source_path}"

    if not output_path or not output_path.strip():
        return "No output path specified in the config"

    output_dir = resolve_and_validate_path(output_path)
    if not output_dir:
        return f"Invalid output directory:\n{output_path}"

    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        return "Output directory cannot be nested in the source directory"

    return None


def adapt_run_config(config: AppConfig, context: DatasetContext) -> tuple[AppConfig, list[str]]:
    """Adapt a config to what a dataset supports.

    Applies these rules in order, each one to the result of the previous ones:
    1. Classification ('bioclip', 'ultralytics') is disabled if the dataset has no metadata,
       or neither detection frames nor crops.
    2. Cropping ('crop') is enabled if a classifier is still enabled, as classification runs
       on the crops. Otherwise it is disabled if the dataset has no detection frames.
    3. Overlays ('overlay') are disabled if the dataset has no detection frames.
    4. Metadata processing ('filter_crops', 'filter_tracks', 'estimate_size') and sorting
       ('sort_crops', 'sort_tracks') are disabled if no classifier is enabled, as they work
       on the classification results.

    Only these 'enabled' switches change, all other values of a setting are kept.
    Existing source and output paths are written as resolved paths with forward slashes.

    Args:
        config: Configuration of the run (paths already checked with validate_paths()).
        context: Inspection of the run's source directory.

    Returns:
        The adapted config, and one message per changed switch (empty if nothing changed).
    """
    caps = DatasetCapabilities.from_context(context)
    config_dict = config.model_dump()
    processing: dict[str, Any] = config_dict["processing"]
    classification: dict[str, Any] = config_dict["classification"]
    metadata: dict[str, Any] = config_dict["metadata"]
    changes: list[str] = []

    def set_enabled(section: str, settings: dict[str, Any], key: str, enabled: bool, reason: str) -> None:
        if settings[key]["enabled"] != enabled:
            settings[key]["enabled"] = enabled
            changes.append(f"{section}.{key} {'enabled' if enabled else 'disabled'}: {reason}")

    if not caps.can_classify_images:
        reason = ("no metadata in the dataset" if not caps.can_process_metadata
                  else "no detection frames or crops in the dataset")
        for key in ("bioclip", "ultralytics"):
            set_enabled("classification", classification, key, False, reason)
    classify = classification["bioclip"]["enabled"] or classification["ultralytics"]["enabled"]

    if classify:
        set_enabled("processing", processing, "crop", True, "required for classification")
    elif not caps.can_process_images:
        set_enabled("processing", processing, "crop", False, "no detection frames in the dataset")
    if not caps.can_process_images:
        set_enabled("processing", processing, "overlay", False, "no detection frames in the dataset")

    if not classify:
        for key in ("filter_crops", "filter_tracks", "estimate_size"):
            set_enabled("metadata", metadata, key, False, "requires an enabled classifier")
        for key in ("sort_crops", "sort_tracks"):
            set_enabled("classification", classification, key, False, "requires an enabled classifier")

    for key in ("source_path", "output_path"):
        resolved = resolve_and_validate_path(config_dict[key])
        if resolved is not None:
            config_dict[key] = resolved.as_posix()

    return AppConfig.model_validate(config_dict), changes


def available_models() -> list[str]:
    """Return the model files that can be selected in a config.

    Model files (.pt or .onnx) in the models directory, and models registered in models.json
    (downloaded on first use).

    Returns:
        Sorted model filenames.
    """
    local_models = {f.name for fmt in ["*.pt", "*.onnx"] for f in MODELS_PATH.glob(fmt)}
    try:
        registered_models = set(list_registered_filenames(MODELS_JSON))
    except FileNotFoundError:
        registered_models = set()
    return sorted(local_models | registered_models)


def cuda_available() -> bool:
    """Return True if PyTorch is installed with CUDA support and a GPU is available."""
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch is not installed, CUDA is not available")
        return False
    return torch.cuda.is_available()


def check_run_config(config: AppConfig, context: DatasetContext) -> str | None:
    """Check what would otherwise only fail once the run has started.

    The dataset needs metadata (every run starts by merging it), device 'cuda' needs an available
    GPU, and the model of an enabled 'classification.ultralytics' has to be available (it is loaded
    only after image processing). Pass the config returned by adapt_run_config(), so a classifier
    the dataset rules switched off is not checked.

    Args:
        config: Configuration of the run, adapted to the dataset by adapt_run_config().
        context: Inspection of the run's source directory.

    Returns:
        Error message string if the run cannot work, None otherwise.
    """
    if not DatasetCapabilities.from_context(context).can_run:
        return "The dataset has no metadata files, so nothing can be processed."
    if config.device == "cuda" and not cuda_available():
        return ("The config uses device 'cuda', but no GPU with CUDA is available "
                "(or PyTorch is installed without CUDA support). Set 'device: cpu' to run on the CPU.")
    ultralytics = config.classification.ultralytics
    if ultralytics.enabled:
        models = available_models()
        if ultralytics.model not in models:
            available = ", ".join(models) or "none"
            return (
                f"Model '{ultralytics.model}' not found: it is not registered in '{MODELS_JSON.name}' "
                f"and there is no such file in '{MODELS_PATH}'.\nAvailable models: {available}"
            )
    return None

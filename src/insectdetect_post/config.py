"""Classes and functions for configuration file management.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    AppConfig and nested models: Validated Pydantic models for post-processing configuration.

Functions:
    get_field_constraints(): Extract numeric constraints for a nested field from a Pydantic model.
    get_field_literals(): Extract allowed Literal values for a nested field from a Pydantic model.
    ensure_config_files(): Create the default config and config selector file if they are missing.
    load_config_selector(): Load the config selector file and return a validated ConfigSelectorModel.
    load_config_yaml(): Load a YAML config file, clamp out-of-range values and return a validated AppConfig.
    check_config_changes(): Return True if two configs differ.
    update_config_selector(): Update the config selector file to point to a different config file.
    update_config_yaml(): Merge updates into the active config file, write back and re-validate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from insectdetect_post.constants import (
    CONFIG_DEFAULT_PATH,
    CONFIG_SELECTOR_PATH,
    CONFIGS_PATH,
    get_bioclip_country_options,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)

class _ConfigDumper(yaml.SafeDumper):
    """SafeDumper that writes tuples as plain YAML sequences.

    Config files are read with yaml.safe_load(), so the writer has to stay within the same
    safe subset. The default Dumper would emit a '!!python/tuple' tag that safe_load then
    refuses to read back. Tuples reach this point because qt-parameters returns them for
    list-valued widgets, and GUI values are written before Pydantic coerces them to lists.
    """


_ConfigDumper.add_representer(tuple, yaml.representer.SafeRepresenter.represent_list)

# YAML dump settings for consistent formatting when writing config files
_YAML_DUMP_KWARGS: dict[str, Any] = {
    "Dumper": _ConfigDumper,
    "default_flow_style": False,
    "allow_unicode": True,
    "sort_keys": False,
    "width": 120,
}


class ConfigSelectorModel(BaseModel):
    """Selector file model that stores the filename of the active config file."""
    config_active: str = "config.yaml"


class CropConfig(BaseModel):
    """Save individual detections as separate files if enabled.

    Detections are cropped from full images where available. Detections that only have an
    existing crop file (no full image, e.g. crop-only datasets) are copied through as-is.

    Methods (applies to detections cropped from full images only):
    - square:   crop to a square bounding box - can improve subsequent classification (default)
    - original: crop to the original detection bounding box aspect ratio
    """
    enabled: bool = True
    method: Literal["square", "original"] = "square"


class OverlayConfig(BaseModel):
    """Draw bounding boxes and metadata overlays on full images and save as copies if enabled."""
    enabled: bool = False


class ProcessingConfig(BaseModel):
    """Post-processing settings applied to full images."""
    crop: CropConfig = CropConfig()
    overlay: OverlayConfig = OverlayConfig()


class BioclipFilterArthropodsConfig(BaseModel):
    """Filter BioCLIP predictions by taxa and/or a region.

    'taxa': One or more arthropod groups to restrict predictions to.
            'all' covers the whole phylum Arthropoda and supersedes any other selection.
    'countries': One or more country codes recognized by the GBIF API. A species is kept if it
                 occurs in any of them; 'all' means no region restriction.
    """
    enabled: bool = True
    taxa: list[Literal["all", "Insecta", "Arachnida", "Diplopoda", "Chilopoda", "Isopoda"]] = (
        Field(default=["Insecta", "Arachnida"], min_length=1)
    )
    countries: list[Literal[get_bioclip_country_options()]] = (  # pyright: ignore[reportInvalidTypeForm]
        Field(default=["all"], min_length=1)
    )

    @field_validator("taxa", "countries")
    @classmethod
    def _normalize_selection(cls, values: list[str]) -> list[str]:
        """Collapse to 'all' if selected, otherwise drop duplicates while keeping the order."""
        if "all" in values:
            return ["all"]
        return list(dict.fromkeys(values))


class BioclipConfig(BaseModel):
    """BioCLIP classification settings via the pybioclip package.

    Rank controls the taxonomic level at which predictions are returned.
    For ranks above species, species-level probabilities are summed up to
    the target rank (e.g. all species in a genus are summed for genus-level).
    """
    enabled: bool = True
    batch_size: int = Field(default=16, ge=1, le=256)
    rank: Literal["kingdom", "phylum", "class", "order", "family", "genus", "species"] = "species"
    filter_arthropods: BioclipFilterArthropodsConfig = BioclipFilterArthropodsConfig()


class UltralyticsConfig(BaseModel):
    """Ultralytics YOLO classification settings."""
    enabled: bool = False
    batch_size: int = Field(default=16, ge=1, le=256)
    model: str = "platform_insect-detect_yolo26s-cls_v1-0-0.onnx"


class SortCropsConfig(BaseModel):
    """Move cropped detections into subdirectories based on the individual prediction."""
    enabled: bool = False


class SortTracksConfig(BaseModel):
    """Move all crops of a track into one subdirectory based on the track's final prediction."""
    enabled: bool = True


class ClassificationConfig(BaseModel):
    """Classification settings applied to cropped detections."""
    bioclip: BioclipConfig = BioclipConfig()
    ultralytics: UltralyticsConfig = UltralyticsConfig()
    sort_crops: SortCropsConfig = SortCropsConfig()
    sort_tracks: SortTracksConfig = SortTracksConfig()

    @model_validator(mode="after")
    def validate_classification(self) -> ClassificationConfig:
        """Validate classifier and sort settings.

        Raises ValueError if:
        - Both BioCLIP and Ultralytics are enabled simultaneously (mutually exclusive).
        - Both sort_crops and sort_tracks are enabled simultaneously (mutually exclusive).
        - sort_crops or sort_tracks is enabled but no classifier is enabled.
        """
        if self.bioclip.enabled and self.ultralytics.enabled:
            raise ValueError(
                "'classification.bioclip' and 'classification.ultralytics' are mutually "
                "exclusive. Enable only one classifier per run."
            )
        if self.sort_crops.enabled and self.sort_tracks.enabled:
            raise ValueError(
                "'classification.sort_crops' and 'classification.sort_tracks' are mutually "
                "exclusive. Enable only one sorting strategy per run."
            )
        if self.sort_crops.enabled and not (self.bioclip.enabled or self.ultralytics.enabled):
            raise ValueError(
                "'classification.sort_crops' requires 'bioclip' or 'ultralytics' to be enabled."
            )
        if self.sort_tracks.enabled and not (self.bioclip.enabled or self.ultralytics.enabled):
            raise ValueError(
                "'classification.sort_tracks' requires 'bioclip' or 'ultralytics' to be enabled."
            )
        return self


class FilterTrackingConfig(BaseModel):
    """Filter tracking IDs based on mean detection confidence and total tracking duration."""
    enabled: bool = False
    min_det_conf: float = Field(default=0.2, ge=0.0, le=1.0)
    min_dur_s: int = Field(default=2, ge=0, le=600)
    max_dur_s: int = Field(default=3600, ge=1, le=21600)

    @model_validator(mode="after")
    def min_must_be_less_than_max(self) -> FilterTrackingConfig:
        """Ensure min_dur_s is strictly less than max_dur_s."""
        if self.min_dur_s >= self.max_dur_s:
            raise ValueError(
                f"metadata.filter_tracks.min_dur_s ({self.min_dur_s}) "
                f"must be less than max_dur_s ({self.max_dur_s})"
            )
        return self


class FilterPredictionConfig(BaseModel):
    """Filter tracking IDs based on weighted mean probability of the final prediction."""
    enabled: bool = False
    min_prob_weighted: float = Field(default=0.2, ge=0.0, le=1.0)


class EstimateSizeConfig(BaseModel):
    """Estimate physical size based on bounding box dimensions and frame size in millimeters."""
    enabled: bool = False
    frame_width_mm: int = Field(default=230, ge=10, le=1000)
    frame_height_mm: int = Field(default=130, ge=10, le=1000)


class MetadataConfig(BaseModel):
    """Metadata processing settings applied to the final results."""
    filter_tracks: FilterTrackingConfig = FilterTrackingConfig()
    filter_predictions: FilterPredictionConfig = FilterPredictionConfig()
    estimate_size: EstimateSizeConfig = EstimateSizeConfig()


class AppConfig(BaseModel):
    """Validated model containing all post-processing configuration settings.

    Loaded from a config YAML file. Missing keys are filled with
    default values automatically. Unknown keys are silently ignored.
    Out-of-range numeric values are clamped to their allowed bounds.

    - source_path: Full path to source directory containing images and metadata.
    - output_path: Full path to output directory where all results will be saved.
    - device: Device for model inference. 'cuda' requires a GPU with CUDA support.
    """
    source_path: str | None = None
    output_path: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    processing: ProcessingConfig = ProcessingConfig()
    classification: ClassificationConfig = ClassificationConfig()
    metadata: MetadataConfig = MetadataConfig()

    @field_validator("source_path", "output_path", mode="before")
    @classmethod
    def coerce_empty_path_to_none(cls, v: object) -> object:
        """Treat empty or whitespace-only path strings as unset (None)."""
        if isinstance(v, str) and not v.strip():
            return None
        return v


def get_field_constraints(
    model_cls: type[BaseModel],
    *field_path: str
) -> dict[str, int | float | None]:
    """Extract numeric constraints for a nested field from a Pydantic model.

    Traverses the model class hierarchy following the given field path and
    reads ge, le, gt, lt and multiple_of metadata from the final field.

    Args:
        model_cls:   Root Pydantic model class to start traversal from.
        *field_path: Sequence of field name strings forming the path to the
                     target field (e.g. 'metadata', 'filter_tracks', 'min_dur_s').

    Returns:
        Dict with keys 'min', 'max' and 'multiple_of', all as int/float or None.
        'min' reflects ge (or gt + 1 as fallback), 'max' reflects le (or lt - 1 as fallback).
        All values are None if no constraints are defined or the path is invalid.
    """
    current_cls = model_cls
    field_info = None

    for key in field_path:
        field_info = current_cls.model_fields.get(key)
        if field_info is None:
            return {"min": None, "max": None, "multiple_of": None}
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current_cls = annotation

    if field_info is None:
        return {"min": None, "max": None, "multiple_of": None}

    ge: int | float | None = None
    le: int | float | None = None
    gt: int | float | None = None
    lt: int | float | None = None
    multiple_of: int | float | None = None
    for meta in field_info.metadata:
        if hasattr(meta, "ge"):
            ge = meta.ge
        if hasattr(meta, "le"):
            le = meta.le
        if hasattr(meta, "gt"):
            gt = meta.gt
        if hasattr(meta, "lt"):
            lt = meta.lt
        if hasattr(meta, "multiple_of"):
            multiple_of = meta.multiple_of

    return {
        "min": ge if ge is not None else (gt + 1 if gt is not None else None),
        "max": le if le is not None else (lt - 1 if lt is not None else None),
        "multiple_of": multiple_of,
    }


def get_field_literals(model_cls: type[BaseModel], *field_path: str) -> tuple[Any, ...]:
    """Extract Literal type args for a nested field from a Pydantic model.

    Traverses the model class hierarchy following the given field path and
    reads the allowed values from a Literal-annotated field.

    Args:
        model_cls:   Root Pydantic model class to start traversal from.
        *field_path: Sequence of field name strings forming the path to the
                     target field (e.g. 'classification', 'sort_crops', 'level').

    A list-valued field (e.g. 'list[Literal[...]]', as used for multi-select settings) is
    unwrapped one level, so the allowed values are returned rather than the Literal itself.

    Returns:
        Tuple of allowed literal values, or empty tuple if the path is invalid
        or the field is not annotated with Literal.
    """
    current_cls = model_cls
    field_info = None

    for key in field_path:
        field_info = current_cls.model_fields.get(key)
        if field_info is None:
            return ()
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current_cls = annotation

    if field_info is None:
        return ()

    annotation = field_info.annotation
    args = get_args(annotation)
    if get_origin(annotation) in (list, tuple) and args:
        return get_args(args[0])
    return args


def _clamp_raw(
    raw: dict[str, object],
    model_cls: type[BaseModel],
    path: str = ""
) -> tuple[dict[str, object], list[str]]:
    """Recursively clamp numeric values in raw to the bounds defined in model_cls.

    Args:
        raw:       Raw dict loaded from YAML, modified in-place where clamping occurs.
        model_cls: Pydantic model class whose field constraints are used as bounds.
        path:      Dot-separated key path prefix used in correction messages.

    Returns:
        Tuple of (modified raw dict, list of human-readable correction messages).
    """
    corrections: list[str] = []

    for key, field_info in model_cls.model_fields.items():
        if key not in raw or raw[key] is None:
            continue

        value = raw[key]
        full_path = f"{path}.{key}" if path else key
        annotation = field_info.annotation

        # Recurse into nested models
        is_nested_model: bool = (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and isinstance(value, dict)
        )
        if is_nested_model:
            raw[key], sub_corrections = _clamp_raw(
                cast(dict[str, object], value),
                cast(type[BaseModel], annotation),
                full_path
            )
            corrections.extend(sub_corrections)
            continue

        # Skip lists and non-numeric values
        origin = getattr(annotation, "__origin__", None)
        if origin is list or not isinstance(value, (int, float)):
            continue

        # Extract constraints via get_field_constraints (single-field path from current model)
        constraints = get_field_constraints(model_cls, key)
        lower = constraints["min"]
        upper = constraints["max"]
        multiple_of = constraints["multiple_of"]

        clamped = cast("int | float", value)
        if lower is not None and clamped < lower:
            clamped = lower
        if upper is not None and clamped > upper:
            clamped = upper

        # Snap to multiple_of after clamping, ensure result stays >= lower
        if multiple_of is not None:
            decimal_places = (len(str(multiple_of).rstrip("0").rsplit(".", maxsplit=1)[-1])
                              if "." in str(multiple_of) else 0)
            snapped = round(round(clamped / multiple_of) * multiple_of, decimal_places)
            if snapped != clamped:
                clamped = snapped
            if lower is not None and clamped < lower:
                clamped += multiple_of

        if clamped != value:
            corrections.append(f"  {full_path}: {value!r} -> {clamped!r}")
            raw[key] = clamped

    return raw, corrections


def _deep_update(base: dict[str, object], updates: dict[str, object]) -> None:
    """Recursively merge updates into base dict in-place.

    Nested dicts are merged rather than replaced. All other value
    types (including lists) are overwritten directly.
    """
    for key, value in updates.items():
        base_value = base.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            _deep_update(base_value, value)
            base[key] = base_value
        else:
            base[key] = value


def ensure_config_files() -> None:
    """Create the configs directory, default config file and config selector file if missing.

    Config files are generated from the AppConfig defaults on first launch.
    Deleting a config file restores its default values on the next start.

    Raises:
        OSError: If the configs directory or the config files could not be created.
    """
    CONFIGS_PATH.mkdir(parents=True, exist_ok=True)

    if not CONFIG_DEFAULT_PATH.exists():
        with open(CONFIG_DEFAULT_PATH, "w", encoding="utf-8") as f:
            yaml.dump(AppConfig().model_dump(), f, **_YAML_DUMP_KWARGS)
        logger.info("Created default config file '%s'.", CONFIG_DEFAULT_PATH)

    if not CONFIG_SELECTOR_PATH.exists():
        update_config_selector(CONFIG_DEFAULT_PATH.name)
        logger.info("Created config selector file '%s'.", CONFIG_SELECTOR_PATH)


def load_config_selector() -> ConfigSelectorModel:
    """Load config selector file, validate and return a ConfigSelectorModel.

    Missing config files are created with their default values. If the selector file is
    unreadable or points to a config file that no longer exists, it falls back to the
    default config file and is updated accordingly.

    Returns:
        ConfigSelectorModel with the validated active config filename.
    """
    ensure_config_files()
    config_default = CONFIG_DEFAULT_PATH.name

    try:
        with open(CONFIG_SELECTOR_PATH, "r", encoding="utf-8") as f:
            raw: dict[str, object] = yaml.safe_load(f) or {}
        selector = ConfigSelectorModel.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as e:
        logger.warning("Could not read config selector file '%s' (%s). Falling back to '%s'.",
                       CONFIG_SELECTOR_PATH, e, config_default)
        update_config_selector(config_default)
        return ConfigSelectorModel(config_active=config_default)

    if not (CONFIGS_PATH / selector.config_active).exists():
        logger.warning("Active config file '%s' not found in '%s'. Falling back to '%s'.",
                       selector.config_active, CONFIGS_PATH, config_default)
        update_config_selector(config_default)
        return ConfigSelectorModel(config_active=config_default)

    return selector


def load_config_yaml(config_path: Path) -> AppConfig:
    """Load a YAML config file, clamp out-of-range values and return a validated AppConfig.

    Missing keys are filled with Pydantic defaults. Out-of-range numeric values
    are clamped to their allowed bounds and a warning is logged for each correction.

    Args:
        config_path: Path to the active config YAML file.

    Returns:
        Validated AppConfig reflecting the final on-disk state.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, object] = yaml.safe_load(f) or {}

    raw, corrections = _clamp_raw(raw, AppConfig)
    if corrections:
        logger.warning(
            "Config values in '%s' were clamped to their allowed range:\n%s",
            config_path, "\n".join(corrections)
        )

    try:
        config = AppConfig.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid configuration in '{config_path}':\n{e}") from e

    validated_dict = config.model_dump()
    if corrections or raw != validated_dict:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(validated_dict, f, **_YAML_DUMP_KWARGS)

    return config


def check_config_changes(
    original: AppConfig | dict[str, object],
    updates: AppConfig | dict[str, object]
) -> bool:
    """Return True if original and updates represent different configurations.

    Args:
        original: Original configuration (AppConfig or dict).
        updates:  Updated configuration to compare against original (AppConfig or dict).

    Returns:
        True if the configurations differ, False if they are identical.
    """
    original_dict = original if isinstance(original, dict) else original.model_dump()
    updates_dict = updates if isinstance(updates, dict) else updates.model_dump()
    return json.dumps(original_dict, sort_keys=True) != json.dumps(updates_dict, sort_keys=True)


def update_config_selector(config_active: str) -> None:
    """Update the config selector file to point to a different config file.

    Raises FileNotFoundError if the specified config file does not exist
    in the configs/ directory.

    Args:
        config_active: Filename of the config file to set as active (e.g. 'config.yaml').
    """
    config_active_path = CONFIGS_PATH / config_active
    if not config_active_path.exists():
        available = [p.name for p in CONFIGS_PATH.glob("*.yaml")
                     if p.name != CONFIG_SELECTOR_PATH.name]
        raise FileNotFoundError(
            f"Cannot set active config to '{config_active}': file not found in '{CONFIGS_PATH}'.\n"
            f"Available config files: {available}"
        )

    selector = ConfigSelectorModel(config_active=config_active)
    with open(CONFIG_SELECTOR_PATH, "w", encoding="utf-8") as f:
        yaml.dump(selector.model_dump(), f, **_YAML_DUMP_KWARGS)


def update_config_yaml(config_path: Path, config_updates: dict[str, object]) -> AppConfig:
    """Merge updates into a config file, write back and re-validate.

    If the file does not exist (e.g. when creating a new config), an empty base
    dict is used and config_updates is written as-is. Otherwise the current file
    is read first and config_updates is merged into it recursively via _deep_update().
    The result is written back and immediately re-validated, which fills any missing
    keys with Pydantic defaults and clamps out-of-range values.

    Args:
        config_path:    Path to the config YAML file to write (need not exist yet).
        config_updates: Nested dict of changes to apply.

    Returns:
        Re-validated AppConfig reflecting the updated file content.
    """
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, object] = yaml.safe_load(f) or {}
    else:
        raw = {}
    _deep_update(raw, config_updates)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, **_YAML_DUMP_KWARGS)

    return load_config_yaml(config_path)

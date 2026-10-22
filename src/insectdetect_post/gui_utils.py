"""Helper functions for GUI modules.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Provides the theme persistence shared by the processing and validation GUIs,
styled widget and parameter form factories, conversion between nested config
dicts and flat ParameterForm values, and source/output path validation.

Functions:
    active_theme(): Return the applied theme, or the default theme before one is applied.
    load_theme(): Return the stored theme name.
    save_theme(): Store the selected theme name.
    create_groupbox(): Create a styled QGroupBox for a config ParameterForm.
    create_param_form(): Create a child parameter form and add it to the parent form.
    create_num_param(): Create a numeric parameter with slider.
    extract_enabled_values(): Extract enabled flags from nested dicts for ParameterForm compatibility.
    restore_enabled_values(): Restore enabled flags back into nested dicts for YAML compatibility.
    resolve_and_validate_path(): Resolve and validate a path string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import qt_themes
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QGroupBox, QVBoxLayout
from qt_parameters import CollapsibleBox, FloatParameter, IntParameter, ParameterForm

from insectdetect_post.constants import SETTINGS_APP, SETTINGS_ORG, THEME_DEFAULT

# Create module-level logger
logger = logging.getLogger(__name__)

# QSettings key of the theme (shared by the processing and the validation window)
_SETTINGS_KEY_THEME = "theme"


def active_theme() -> qt_themes.Theme:
    """Return the applied theme, or the default theme before one is applied."""
    theme = qt_themes.get_theme() or qt_themes.get_theme(THEME_DEFAULT)
    if theme is None:
        raise RuntimeError(f"Default theme '{THEME_DEFAULT}' is not available")
    return theme


def load_theme() -> str:
    """Return the stored theme name, or the default if unavailable.

    Returns:
        Name of a theme that exists in qt_themes.
    """
    theme_name = str(QSettings(SETTINGS_ORG, SETTINGS_APP).value(_SETTINGS_KEY_THEME, THEME_DEFAULT))
    if theme_name not in qt_themes.get_themes():
        logger.warning("Theme '%s' not available, using '%s'", theme_name, THEME_DEFAULT)
        return THEME_DEFAULT
    return theme_name


def save_theme(theme_name: str) -> None:
    """Store the selected theme name.

    Args:
        theme_name: Name of the selected theme.
    """
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(_SETTINGS_KEY_THEME, theme_name)


def create_groupbox(title: str, color: str, form: ParameterForm) -> QGroupBox:
    """Create a styled QGroupBox for a config ParameterForm.

    Args:
        title: Display title shown on the group box header.
        color: Border/accent color as a CSS color string (e.g. hex code).
        form: ParameterForm to embed inside the group box.

    Returns:
        Styled QGroupBox containing the given form.
    """
    box = QGroupBox(title)
    box.setStyleSheet(f"""
        QGroupBox {{
            border: 2px solid {color};
            border-radius: 8px;
            font-weight: bold;
            margin-top: 0.5em;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 5px;
            padding: 0 3px 0 3px;
        }}
    """)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 6, 0, 0)
    layout.addWidget(form)
    return box


def create_param_form(
    parent_form: ParameterForm,
    name: str,
    title: str,
    tooltip: str = "",
    has_checkbox: bool = True
) -> tuple[ParameterForm, CollapsibleBox]:
    """Create a child parameter form and add it to the parent form.

    Args:
        parent_form: Form to nest the new child form under.
        name: Internal parameter name of the child form.
        title: Display title shown on the collapsible box header.
        tooltip: Optional tooltip for the collapsible box.
        has_checkbox: Add a checkbox to the box header, tied to the section's enabled state.

    Returns:
        Tuple of (child form, its collapsible box container).
    """
    child_form = ParameterForm(name)
    form_box = parent_form.add_form(child_form, checkable=has_checkbox)
    form_box.set_title(title)
    form_box.set_collapsible(False)
    if tooltip:
        form_box.setToolTip(tooltip)
    return child_form, form_box


def create_num_param(
    name: str,
    label: str,
    min_val: float,
    max_val: float,
    default: float,
    param_type: Literal["int", "float"] = "int",
    tooltip: str = "",
    slider_max: float | None = None
) -> IntParameter | FloatParameter:
    """Create a numeric parameter with slider.

    Args:
        name: Internal parameter name.
        label: Display label shown next to the slider.
        min_val: Minimum value for both the line edit and slider.
        max_val: Maximum value of the line edit, and of the slider without 'slider_max'.
        default: Default value.
        param_type: "int" for an IntParameter, "float" for a FloatParameter.
        tooltip: Optional tooltip for the parameter.
        slider_max: Lower slider maximum for the usual range (larger values can still be typed).

    Returns:
        Configured IntParameter or FloatParameter instance.
    """
    param = FloatParameter(name) if param_type == "float" else IntParameter(name)
    param.set_label(label)
    if slider_max is None:
        slider_max = max_val
    if isinstance(param, FloatParameter):
        param.set_decimals(2)
        param.set_line_min(float(min_val))
        param.set_line_max(float(max_val))
        param.set_slider_min(float(min_val))
        param.set_slider_max(float(slider_max))
        param.set_default(float(default))
    else:
        param.set_line_min(int(min_val))
        param.set_line_max(int(max_val))
        param.set_slider_min(int(min_val))
        param.set_slider_max(int(slider_max))
        param.set_default(int(default))
    if tooltip:
        param.setToolTip(tooltip)
    return param


def extract_enabled_values(config: dict[str, Any]) -> dict[str, Any]:
    """Extract enabled flags from nested dicts for ParameterForm compatibility.

    Args:
        config: Config data as a nested dict (e.g. an AppConfig section dump).

    Returns:
        Flattened dict with a "<key>_enabled" entry alongside each nested "key" dict.
    """
    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict) and "enabled" in value:
            rest = {k: v for k, v in value.items() if k != "enabled"}
            result[key] = extract_enabled_values(rest)
            result[f"{key}_enabled"] = value["enabled"]
        else:
            result[key] = value
    return result


def restore_enabled_values(config: dict[str, Any]) -> dict[str, Any]:
    """Restore enabled flags back into nested dicts for YAML compatibility.

    Args:
        config: Config data as a flattened dict, as produced by extract_enabled_values().

    Returns:
        Nested dict matching the original config data structure.
    """
    result: dict[str, Any] = {}
    for key, value in config.items():
        if key.endswith("_enabled"):
            base_key = key[:-8]
            if base_key not in result:
                result[base_key] = {}
            result[base_key]["enabled"] = value
    for key, value in config.items():
        if not key.endswith("_enabled"):
            if isinstance(value, dict):
                nested = restore_enabled_values(value)
                if key in result:
                    result[key].update(nested)
                else:
                    result[key] = nested
            else:
                result[key] = value
    return result


def resolve_and_validate_path(path_str: str) -> Path | None:
    """Resolve and validate a path string.

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

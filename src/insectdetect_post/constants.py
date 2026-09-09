"""Shared constants used across insect-detect-post modules.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/
"""

import csv
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until a directory containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not locate project root (no {marker} found)")


# Paths to root directory and subdirectories for configs, models, and filters
BASE_PATH: Path = _find_project_root()
CONFIGS_PATH: Path = BASE_PATH / "configs"
MODELS_PATH: Path = BASE_PATH / "models"
FILTERS_PATH: Path = BASE_PATH / "filters"

# Path to the config selector file that stores the filename of the active config file
CONFIG_SELECTOR_PATH: Path = CONFIGS_PATH / "config_selector.yaml"

# Path to the default config file, generated from the AppConfig defaults on first launch
CONFIG_DEFAULT_PATH: Path = CONFIGS_PATH / "config.yaml"

# Paths to the release assets registries
MODELS_JSON: Path = MODELS_PATH / "models.json"
FILTER_ASSETS_JSON: Path = FILTERS_PATH / "filter_assets.json"


@dataclass(frozen=True)
class OutputLayout:
    """Directory layout for a single pipeline run's output directory."""
    root: Path

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def crops_dir(self) -> Path:
        return self.root / "images" / "crops"

    @property
    def overlays_dir(self) -> Path:
        return self.root / "images" / "overlays"

    def crop_label_dir(self, label: str) -> Path:
        return self.crops_dir / str(label)

    def relative(self, path: Path) -> str:
        """POSIX-style path relative to the output root."""
        return path.relative_to(self.root).as_posix()


# Taxonomic rank hierarchy, broadest to most specific, used to compare rank specificity
RANK_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Phylum filter for GBIF taxon resolution, used by filters/resolve_tol_gbif_species.py and
# build_region_filter.py. Set to 'None' to resolve/query the full Tree of Life species list
PHYLUM_FILTER: str | None = "Arthropoda"

# GBIF backbone taxonKeys for known PHYLUM_FILTER values, used by build_region_filter.py
# to restrict occurrence facet queries server-side for efficiency
PHYLUM_TAXON_KEYS: dict[str, int] = {
    "Arthropoda": 54,
}

# Minimum GBIF occurrence records required for a taxon to be included in a region filter
MIN_OCCURRENCE_COUNT: int = 3


@cache
def get_bioclip_country_options() -> tuple[str, ...]:
    """Return the country codes accepted by the GBIF API, prefixed with 'all'.

    'all' means no BioCLIP region restriction. The CSV is read on first call rather than at
    import, so that importing any other constant from this module does not require it to exist.

    Raises:
        FileNotFoundError: If the country code CSV has not been generated yet.
    """
    csv_path = FILTERS_PATH / "gbif_country_codes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"'{csv_path}' not found. Run 'filters/resolve_gbif_country_codes.py' first."
        )
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return ("all", *(row["iso2"] for row in csv.DictReader(f)))

# Default thread-pool size for image processing operations (e.g. crop/overlay)
MAX_WORKERS: int = min((os.cpu_count() or 4) + 4, 32)

# GUI styling: main window minimum size
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800

# GUI styling: group box accent colors
COLOR_GROUP_PROCESSING = "#4DD0E1"
COLOR_GROUP_CLASSIFICATION = "#7986CB"
COLOR_GROUP_METADATA = "#4DB6AC"

# GUI styling: control button colors
COLOR_BTN_SAVE = "#455A8A"
COLOR_BTN_RUN = "#00796B"
COLOR_BTN_CANCEL = "#D32F2F"
COLOR_BTN_RESET = "#546E7A"
COLOR_BTN_UPDATE = "#FFA000"

# GUI settings: QSettings organization/application names for persisted user preferences
SETTINGS_ORG = "insect-detect"
SETTINGS_APP = "insect-detect-post"

# GUI styling: default qt-themes theme (used if no theme is persisted)
THEME_DEFAULT = "github_dark"

# GUI styling: progress bar stylesheet template (colors filled in from the active theme)
PROG_BAR_STYLESHEET = """
    QProgressBar {{
        border: 1px solid {border};
        border-radius: 5px;
        text-align: center;
        background-color: {background};
        color: {text};
        font-weight: bold;
    }}
    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #26A69A, stop:1 #00897B);
        border-radius: 4px;
    }}
"""

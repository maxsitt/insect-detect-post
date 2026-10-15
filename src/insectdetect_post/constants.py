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

    @property
    def validation_dir(self) -> Path:
        return self.root / "validation"

    @property
    def filters_dir(self) -> Path:
        return self.root / "filters"

    def crop_label_dir(self, label: str) -> Path:
        return self.crops_dir / str(label)

    def relative(self, path: Path) -> str:
        """POSIX-style path relative to the output root."""
        return path.relative_to(self.root).as_posix()


# Columns identifying one track across all metadata CSVs (used for e.g. group by, join and sort)
TRACK_KEY_COLUMNS = ["device_id", "session_id", "track_id"]

# Metadata column of the per-session camera setting (e.g. platform type or flower species)
SETTING_COLUMN = "setting"

# Taxonomic rank hierarchy, broadest to most specific, used to compare rank specificity
RANK_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Phylum filter for GBIF taxon resolution, used by filters/build_bioclip_gbif_species_mapping.py and
# build_region_filter.py. Set to 'None' to resolve/query the full Tree of Life species list
PHYLUM_FILTER: str | None = "Arthropoda"

# GBIF checklist that taxon names are matched against and occurrences are interpreted with:
# Catalogue of Life Extended Release (COL XR), the GBIF default since replacing the GBIF Backbone
GBIF_CHECKLIST_KEY: str = "7ddf754f-d193-4cc9-b351-99906754a03b"

# COL XR taxon keys (alphanumeric strings) for known PHYLUM_FILTER values, used by
# build_region_filter.py to restrict occurrence facet queries server-side for efficiency
PHYLUM_TAXON_KEYS: dict[str, str] = {
    "Arthropoda": "RT",
}

# Minimum GBIF occurrence records required for a taxon to be included in a region filter
MIN_OCCURRENCE_COUNT: int = 3

# GBIF continent vocabulary (https://api.gbif.org/v1/enumeration/basic/Continent), prefixed with 'all'
BIOCLIP_CONTINENT_OPTIONS: tuple[str, ...] = (
    "all", "AFRICA", "ANTARCTICA", "ASIA", "EUROPE", "NORTH_AMERICA", "OCEANIA", "SOUTH_AMERICA",
)


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

# Validation: defaults for stratified subsampling
VALIDATION_PROB_BIN_EDGES_DEFAULT = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VALIDATION_MIN_PER_STRATUM_DEFAULT = 5
VALIDATION_MAX_PER_STRATUM_DEFAULT = 50
VALIDATION_SEED_DEFAULT = 42

# Validation: effort cap (s) per timed interval of a track (a break adds at most this much)
VALIDATION_EFFORT_IDLE_CAP_S = 300

# Validation: comparable tracks below which the statistics warn of a small sample
VALIDATION_SMALL_SAMPLE_TRACKS = 30

# Validation: 'same individual' link candidates: maximum gap and overlap (s) between two tracks
VALIDATION_SAME_INDIVIDUAL_WINDOW_S = 10.0
VALIDATION_SAME_INDIVIDUAL_OVERLAP_S = 2.0

# Validation: maximum length in characters of the per-track notes (one line)
VALIDATION_NOTES_MAX_CHARS = 100

# Validation: crop playback intervals (ms) in single-image mode
VALIDATION_PLAYBACK_INTERVALS_MS = (250, 500, 1000)
VALIDATION_PLAYBACK_INTERVAL_DEFAULT_MS = 500

# GUI settings: QSettings organization/application names for persisted user preferences
SETTINGS_ORG = "insect-detect"
SETTINGS_APP = "insect-detect-post"

# GUI styling: main window minimum size
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800

# GUI styling: default qt-themes theme (used if no theme is persisted)
THEME_DEFAULT = "github_dark"

# GUI styling: group box accent colors
COLOR_GROUP_PROCESSING = "#4DD0E1"
COLOR_GROUP_CLASSIFICATION = "#7986CB"
COLOR_GROUP_METADATA = "#4DB6AC"
COLOR_GROUP_VALIDATION = "#9E8CC4"

# GUI styling: control button colors
COLOR_BTN_SAVE = "#455A8A"
COLOR_BTN_RUN = "#00796B"
COLOR_BTN_CANCEL = "#D32F2F"
COLOR_BTN_RESET = "#546E7A"
COLOR_BTN_UPDATE = "#FFA000"
COLOR_BTN_VALIDATE = "#5E548E"
COLOR_BTN_HELP = "#2E8BA8"

# GUI styling: colors of the validation verdict types (keys: validation_session.VERDICT_TYPES)
COLOR_VERDICT = {
    "agree": "#43A047",
    "partial": "#7CB342",
    "corrected": "#FB8C00",
    "false_positive": "#E53935",
    # Same color, the flag mark tells them apart
    "indeterminate": "#8D6E63",
    "not_assessable": "#8D6E63",
}

# GUI styling: colors of the validation flags (low image quality, multiple / same individual)
COLOR_FLAG_LOW_QUALITY = "#9575CD"
COLOR_FLAG_MULTIPLE_INDIVIDUALS = "#26A69A"
COLOR_FLAG_SAME_INDIVIDUAL = "#4F9DDE"

# GUI styling: validation error text on dark / light themes
COLOR_ERROR_TEXT_DARK = "#EF9A9A"
COLOR_ERROR_TEXT_LIGHT = "#B71C1C"

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

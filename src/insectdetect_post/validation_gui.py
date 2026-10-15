"""Validation GUI for systematic human validation of classified tracks.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Two pages: setup (run, validator, track selection, subsample, sessions) and validation (track
list, crops, track info, verdict form). Changes are saved automatically. A statistics dialog
summarizes the session and exports the tables.

Classes:
    RunLoadWorker: Loads a run (and optionally the taxonomy index) in a background thread.
    SetupPage: First page: pick the run and validator, select and subsample the tracks, start a session.
    TrackInfoPanel: Header above the crops: the current track's metadata, prediction and candidate taxa.
    VerdictPanel: Form for the verdict on the current track: one-click actions, flags, taxon per rank, notes.
    ValidationPage: Second page, where tracks are validated: track list, crops, info and verdict panel.
    StatsDialog: Dialog with the session's verdicts, model agreement and accuracy, with CSV export.
    ValidationWindow: Standalone or opened from the processing GUI; switches between both pages.

Functions:
    main(): Standalone entry point ('uv run --no-sync validate').
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl
import qt_themes
from PySide6.QtCharts import QBarCategoryAxis, QBarSet, QChart, QChartView, QPercentBarSeries, QValueAxis
from PySide6.QtCore import (
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QSettings,
    QStringListModel,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QResizeEvent,
    QShortcut,
    QShowEvent,
    QTextDocumentFragment,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qt_parameters import BoolParameter, ComboParameter, ParameterForm, PathParameter, StringParameter

from insectdetect_post.constants import (
    BASE_PATH,
    COLOR_BTN_CANCEL,
    COLOR_BTN_HELP,
    COLOR_BTN_RESET,
    COLOR_BTN_RUN,
    COLOR_BTN_SAVE,
    COLOR_BTN_UPDATE,
    COLOR_BTN_VALIDATE,
    COLOR_ERROR_TEXT_DARK,
    COLOR_ERROR_TEXT_LIGHT,
    COLOR_FLAG_LOW_QUALITY,
    COLOR_FLAG_MULTIPLE_INDIVIDUALS,
    COLOR_FLAG_SAME_INDIVIDUAL,
    COLOR_GROUP_CLASSIFICATION,
    COLOR_GROUP_METADATA,
    COLOR_GROUP_PROCESSING,
    COLOR_GROUP_VALIDATION,
    COLOR_VERDICT,
    SETTING_COLUMN,
    SETTINGS_APP,
    SETTINGS_ORG,
    THEME_DEFAULT,
    TRACK_KEY_COLUMNS,
    VALIDATION_MAX_PER_STRATUM_DEFAULT,
    VALIDATION_MIN_PER_STRATUM_DEFAULT,
    VALIDATION_NOTES_MAX_CHARS,
    VALIDATION_PLAYBACK_INTERVAL_DEFAULT_MS,
    VALIDATION_PLAYBACK_INTERVALS_MS,
    VALIDATION_PROB_BIN_EDGES_DEFAULT,
    VALIDATION_SAME_INDIVIDUAL_WINDOW_S,
    VALIDATION_SEED_DEFAULT,
    VALIDATION_SMALL_SAMPLE_TRACKS,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from insectdetect_post.exceptions import RunSearchCancelled, ValidationRunError
from insectdetect_post.gui_utils import (
    configure_logging,
    create_groupbox,
    create_num_param,
    load_theme,
    resolve_and_validate_path,
    save_theme,
)
from insectdetect_post.multi_combo_parameter import MultiComboParameter
from insectdetect_post.styled_button import StyledButton
from insectdetect_post.update_button import UpdateButton
from insectdetect_post.validation_run import TrackKey, ValidationRun
from insectdetect_post.validation_sampling import (
    NONE_LABEL,
    assign_strata,
    available_strata_ranks,
    filter_settings,
    filter_taxa,
    parse_prob_edges,
    sample_strata,
    setting_counts,
    setting_strata_available,
    small_strata,
    stratum_summary,
)
from insectdetect_post.validation_session import (
    DIFFERENT_TAXA_IN_TRACK,
    ID_QUALIFIER_AGG,
    ID_QUALIFIER_INC,
    IMAGE_QUALITY_REASONS,
    MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE,
    MULTIPLE_INDIVIDUALS_TYPES,
    VERDICT_TYPES,
    SamplingConfig,
    ValidationSession,
    Verdict,
    allowed_qualifier,
    anchor_of,
    close_session_log,
    delete_session,
    derive_verdict,
    has_qualifier,
    join_links,
    open_session_log,
    parse_links,
    qualifier_mark,
    session_files,
    sort_tracks_chronologically,
    source_csv_changed,
    with_link,
    with_qualifier,
)
from insectdetect_post.validation_stats import (
    agreement_by_rank,
    column_description,
    compare_sessions,
    confusion_pairs,
    effort_summary,
    export_stats,
    is_stratified,
    linked_track_conflicts,
    n_different_taxa_in_crops,
    n_different_taxa_in_track,
    n_individuals,
    n_low_quality,
    n_uncertain,
    per_taxon_table,
    progress_by_stratum,
    reliability_by_prob_bin,
    unvalidated_strata,
    validated_in_queue_order,
    verdict_distribution,
    verdict_table,
)
from insectdetect_post.validation_taxonomy import (
    INDET_LABEL,
    VALIDATION_RANKS,
    Lineage,
    TaxonomyIndex,
    rank_index,
    ranks_below,
    ranks_up_to,
)
from insectdetect_post.validation_widgets import (
    FIT,
    GRID_ZOOM_STEPS,
    LINK_GLYPH_NEXT,
    MULTIPLE_INDIVIDUALS_LABELS,
    SINGLE_ZOOM_STEPS,
    VERDICT_LABELS,
    CropGridModel,
    CropGridView,
    GridZoom,
    RowStatus,
    SingleImageView,
    TaxonCombo,
    ThumbnailCache,
    TrackListModel,
)

# Create module-level logger
logger = logging.getLogger(__name__)

# QSettings keys
_SETTINGS_KEY_VALIDATOR = "validation/validator"
_SETTINGS_KEY_LAST_RUN = "validation/last_run_root"
_SETTINGS_KEY_ZOOM = "validation/grid_zoom"  # 'px:<cell>' (reduced to a cell size) or 'x:<k>' (k x pixels)
_SETTINGS_KEY_SINGLE_ZOOM = "validation/single_zoom"  # 'fit' or a factor of SINGLE_ZOOM_STEPS
_SETTINGS_KEY_AUTO_ADVANCE = "validation/auto_advance"
_SETTINGS_KEY_PLAYBACK_MS = "validation/playback_ms"
_SETTINGS_KEY_COMPARE = "validation/compare_strip"
_SETTINGS_KEY_SORT = "validation/list_sort"

# Window title on the setup page (an open session shows its run, validator and ID instead)
_WINDOW_TITLE = "Insect Detect Post-Processing - Validation"

# Labels of the ranks (capitalized)
_RANK_LABELS = {rank: rank.capitalize() for rank in VALIDATION_RANKS}

# Setup page: delay of the subsample preview after the last parameter change
_PREVIEW_DELAY_MS = 150

# Verdict panel: gap between its three sections
_SECTION_GAP = 12

# Verdict panel: styles of the verdict buttons and of the flag buttons next to their dropdowns
_ACTION_BTN_STYLE = "QPushButton { font-weight: bold; padding: 6px; }"
_FLAG_BTN_STYLE = "QPushButton { font-weight: bold; padding: 3px 6px; }"

# Verdict panel: style of the recent-identification chips
_CHIP_STYLE = "QToolButton { border: 1px solid palette(mid); border-radius: 8px; padding: 3px 8px; }"

# Verdict panel: labels of the flag reasons and of the sex buttons
_IMAGE_QUALITY_REASON_LABELS = {
    "blurry": "Blurry", "too_small": "Too small", "partially_visible": "Partially visible",
    "occluded": "Occluded", "other": "Other",
}
_SEX_LABELS = {"female": "♀ female", "male": "♂ male"}

# Verdict panel: button tooltips
_AGREE_TOOLTIP = "The model prediction is correct down to its classification rank"
_BLIND_AGREE_TOOLTIP = "Not available in blind sessions: it would adopt the model prediction instead of your determination"
_LINK_BTN_TOOLTIP = ("This track continues an earlier track of the same individual (lost track, new ID); "
                     "choose the track next to it")
# 'Same individual' button with several tracks selected (see ValidationPage._batch_link_state())
_LINK_BTN_BATCH_TOOLTIPS = {
    "link": (f"Link the selected tracks in time order as one individual; tracks more than "
             f"{VALIDATION_SAME_INDIVIDUAL_WINDOW_S:g} s apart are not linked"),
    "unlink": "Remove the links of the selected tracks",
    "none": (f"No two selected tracks of the same device/session follow each other within "
             f"{VALIDATION_SAME_INDIVIDUAL_WINDOW_S:g} s"),
}

# Verdict panel: more children of a parent than this are offered by type-ahead only, not as a dropdown list
_MAX_LISTED_CHILDREN = 3000

# Track list: labels of the filters (TrackListModel.FILTERS)
_FILTER_LABELS = {
    "all": "All", "todo": "To do", "done": "Validated", "agree": "Agreed", "corrected": "Corrected",
    "indeterminate": "indet. / partial", "uncertain": "Uncertain (inc.)", "false_positive": "False positive",
    "linked": "Same individual", "multiple": "Multiple individuals", "low_quality": "Low quality",
}

# Track list: labels of the sort modes (TrackListModel.SORTS)
_SORT_LABELS = {
    "chrono": "Chronological per device", "setting": "Camera setting, then chronological",
    **{mode: f"Predicted {rank} (alphabetically)" for mode, rank in TrackListModel.TAXON_SORTS.items()},
    "prob_desc": "Weighted probability ↓", "prob_asc": "Weighted probability ↑",
    "det_conf_desc": "Detection confidence ↓", "det_conf_asc": "Detection confidence ↑",
    "duration_desc": "Track duration ↓", "duration_asc": "Track duration ↑",
    "queue": "Sampling order (shuffled)",
}

# Track list: row fields of the numeric sorts that come from a column of the final CSV
_SORT_COLUMNS = {"det_conf": "det_conf_mean", "duration": "duration_s"}

# Crops of the next track decoded in advance
_PREFETCH_NEXT_CROPS = 60

# Comparison strip: cell side (px)
_COMPARE_CELL = 96

# Comparison strip: tracks of the same predicted taxon (most confident first)
_COMPARE_SAME_TAXON_MAX = 10

# Delay before a changed verdict is written to disk
_AUTOSAVE_DELAY_MS = 500

# Delay of the auto-advance after a one-click verdict, so the verdict is seen
_AUTO_ADVANCE_DELAY_MS = 120

# Delay of the guidelines on a session's first start, so the page is shown first
_GUIDELINES_DELAY_MS = 300

# Batches larger than this ask for confirmation
_BATCH_CONFIRM_THRESHOLD = 25

# Help: block margins (QTextDocument supports only a CSS subset), guidelines and shortcuts
_HELP_CSS = """
h2 { font-size: 13pt; margin-top: 22px; margin-bottom: 2px; }
h3 { font-size: 11pt; margin-top: 18px; margin-bottom: 2px; }
p { margin-top: 2px; margin-bottom: 10px; line-height: 130%; }
li { margin-bottom: 7px; line-height: 130%; }
td { padding-bottom: 6px; }
"""

_GUIDELINES_HTML = f"""
<h2>How to validate</h2>
<p>A <b>track</b> is one individual followed over several images. You judge the whole track at once: look at its
crops, then confirm or correct the model&rsquo;s prediction. The prediction is never overwritten - your verdict
and determination are stored next to it. Everything is saved automatically: close the window at any time and
resume the session from the setup page.</p>

<h3>The window</h3>
<ul>
<li><b>Left: the track list.</b> Each row shows the status, device (D1 &hellip;), session (S &hellip;), track ID
and the prediction with its weighted probability. Status: &#9675; to do, &#10003; agree, &#9680; partial,
&#9998; corrected, &#10005; not an arthropod, ? not determinable. After it, &#8258; marks multiple individuals and
&#9662; low image quality. After the ID, &#8625;&times;n marks the first track of an individual with n tracks
and &#8627; a track continuing it.</li>
<li><b>Center: the track.</b> Track info and the <i>Model prediction</i> shown above the crops viewer. The name
of the predicted taxon (and candidates) is the raw BioCLIP 2 output, the lineage below it and the determination
dropdowns use the accepted GBIF species name (Catalogue of Life), so a synonym predicted by the model counts as
its accepted species. Your determination is compared with this name.
The weighted probability is the mean probability, lowered when only some of the crops were predicted as this taxon.
<i>1st choice in n / m crops</i> says how often it was the model&rsquo;s best guess (top-1). The candidates below
list the alternatives. Click a crop to show its own prediction at the bottom, double-click it for single view.
<i>Compare with similar tracks</i> shows one crop of each related track, including the same linked individual,
earlier tracks that could be linked, and tracks with the same prediction.</li>
<li><b>Right: your verdict</b> - quick verdicts, the determination per rank, qualifiers, flags and notes.</li>
<li><b>Top:</b> progress, <i>Statistics</i>, <i>Export merged results</i>, this help and <i>Back to setup</i>.</li>
</ul>

<h3>For every track</h3>
<ol>
<li><b>Look at the crops</b> - double-click one for the large view, <b>Space</b> plays the track.</li>
<li><b>Give a verdict:</b> agree (down to a rank), correct the taxon, or mark the track as not an arthropod
or not determinable.</li>
<li><b>Add qualifiers or flags</b> where they apply (details below).</li>
<li><b>Move on</b> with <b>Enter</b> (next track without a verdict) or <b>PgDn</b>.</li>
</ol>

<h3>Verdicts</h3>
<p>Your <b>determination</b> is the taxon you enter per rank, your <b>verdict</b> follows from what you enter:
<i>agree</i>, <i>partial</i> (confirmed, but not down to the lowest rank), <i>corrected</i>
(a taxon differs from the prediction), <i>not an arthropod</i> or <i>not determinable</i> (<i>not assessable</i> if
flagged as low image quality or different taxa). A track with a verdict is validated.</p>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td width="26%"><b>Agree (A)</b></td>
<td>The prediction is correct down to the rank this run was classified to. <i>Agree</i> and the rank buttons
below it are disabled in blind sessions.</td></tr>
<tr><td><b>Agree down to a rank (1 - 4)</b></td>
<td>The small buttons below <i>Agree</i>: correct down to this rank, the ranks below become <i>indet.</i> (not
determinable from the images) - e.g. &ldquo;certainly Syrphidae, genus unclear&rdquo;.</td></tr>
<tr><td><b>Correct a rank</b></td>
<td>Pick or type the taxon: higher ranks are filled in, lower ranks are reset. With the genus set, the species
field accepts the epithet alone (&ldquo;balteatus&rdquo;). Taxa missing from the list can be typed in. The
<b>?</b> next to a rank sets it and all lower ranks to <i>indet.</i>, click it again to restore the model&rsquo;s
name. <i>Recent</i> applies a determination you used before with one click.</td></tr>
<tr><td><b>Not an arthropod (F)</b></td>
<td>A false detection: background, plant part, shadow. Press again to take it back.</td></tr>
<tr><td><b>Not determinable (U)</b></td>
<td>An arthropod, but no rank can be determined. Press again to take it back.</td></tr>
<tr><td><b>Reset (R)</b></td>
<td>Discard everything recorded for the track - verdict, determination, qualifiers, flags, links and notes - and
start again from the prediction.</td></tr>
</table>

<h3>Qualifiers and flags</h3>
<p>These describe the determination or the track. On their own they do not validate a track:
it stays <i>To do</i> until you give a verdict.</p>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td><b>&#9792; female / &#9794; male</b></td>
<td>Optionally record the sex when it is clearly visible.</td></tr>
<tr><td width="26%"><b>inc. - uncertain (I)</b></td>
<td>The characteristic features are visible, but you are not certain of the name (written &ldquo;<i>Eristalis
tenax</i> sp. inc.&rdquo;). The statistics can leave uncertain determinations out. If the features of a rank are
not visible at all, set it to <i>indet.</i> with its <b>?</b> instead.</td></tr>
<tr><td><b>agg. - aggregate (G)</b></td>
<td>A group of closely related species that images cannot separate: enter the species the aggregate is named
after and set <i>agg.</i> (written &ldquo;<i>Bombus lucorum</i> agg.&rdquo;). Only with a species entered; the
species is then not compared with the model, the ranks above it are.</td></tr>
<tr><td><b>Same individual (L)</b></td>
<td>The track continues an earlier track of the same individual that the tracker lost and gave a new ID. Tick the
earlier tracks in the <i>Same individual as</i> dropdown (<b>L</b> links the nearest one). Linked tracks share
an <i>individual_id</i> in the export, their determinations are not copied. Tracks marked as <i>Not an
arthropod</i> (false positive) cannot be linked.</td></tr>
<tr><td><b>Multiple individuals (M)</b></td>
<td>More than one arthropod. <i>In crops</i>: several individuals inside the single crops; <i>in track</i>:
one individual per crop, but the crops of the track show different taxa (a tracking error). <i>Same taxon in
crops</i>: determine as usual. <i>Different taxa in crops</i> or <i>in track</i>: determine the majority or set
<i>indet.</i>. The model had no single target, so these tracks are left out of the accuracy statistics.</td></tr>
<tr><td><b>Low image quality (Q)</b></td>
<td>Blurry, too small, partially visible or occluded (choose the reason next to it). Determine what you can see,
but such tracks are left out of the accuracy statistics.</td></tr>
<tr><td><b>Notes</b></td>
<td>One short line (at most {VALIDATION_NOTES_MAX_CHARS} characters) for what the fields do not cover,
e.g. &ldquo;mating pair&rdquo; or &ldquo;pollen on the legs&rdquo;.</td></tr>
</table>

<h3>Working through many tracks</h3>
<ul>
<li><b>Filter and sort</b> the list, e.g. only <i>To do</i> or <i>Low quality</i> tracks, sorted by predicted
taxon, probability or camera setting. The default chronological order keeps neighboring tracks of a device
together, which helps to spot the same individual.</li>
<li><b>Search</b> for what a row shows: <i>ID 140</i>, <i>S3</i>, <i>D1</i> or part of a taxon name. IDs match
as a whole (<i>S1</i> does not find S12), and all terms must match (<i>D1 S3</i>).</li>
<li><b>Batch:</b> select several tracks in the list with Ctrl / Shift + click or all with Ctrl+A.
The verdict and flag buttons then apply to each of them - <i>Agree</i> confirms each track&rsquo;s own prediction.
<i>Copy determination to selection</i> copies the shown track&rsquo;s determination to the others instead.
Select tracks and press <b>L</b> to link them in time order as one individual, tracks more than 10 s
apart are not linked.</li>
<li><b>Auto-advance</b>: jump to the next track right after a one-click verdict.</li>
</ul>

<h3>Results</h3>
<p><i>Statistics</i> compares the model&rsquo;s predictions with your determinations - accuracy per rank, per
predicted taxon and per probability range - and exports the tables as CSV files. <i>Export merged results</i>
writes one CSV with every track of the run, your verdicts and determinations, and one combined taxon per track for
further analysis: your determination for a validated track, otherwise the model prediction.</p>
"""

_SHORTCUTS_HTML = """
<h2>Keyboard shortcuts</h2>
<p><b>PgDown</b> and <b>PgUp</b> move between tracks, <b>Enter</b> jumps to the next unvalidated one. The arrow
keys stay inside the current track: they move between its crops.</p>

<h3>Tracks</h3>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td width="26%"><b>PgDown / PgUp</b></td>
<td>Next / previous track (also <b>Ctrl+Right / Ctrl+Left</b>, except while typing in a text field)</td></tr>
<tr><td><b>Enter</b></td>
<td>Next track without a verdict (in a taxon field it first confirms a changed name)</td></tr>
<tr><td><b>Ctrl+A</b> in the track list</td>
<td>Select all shown tracks (batch); Ctrl / Shift + click select single tracks or a range</td></tr>
<tr><td><b>Ctrl+L</b></td><td>Collapse / restore the track list</td></tr>
</table>

<h3>Verdict and flags</h3>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td width="26%"><b>A</b></td><td>Agree with the model prediction</td></tr>
<tr><td><b>1 - 4</b></td><td>Agree down to class / order / family / genus</td></tr>
<tr><td><b>F</b> / <b>U</b></td><td>Toggle not an arthropod / not determinable</td></tr>
<tr><td><b>I</b> / <b>G</b></td><td>Toggle uncertain (inc.) / aggregate (agg.)</td></tr>
<tr><td><b>L</b></td><td>Link to the nearest earlier track (again: clear the links)</td></tr>
<tr><td><b>M</b> / <b>Q</b></td><td>Toggle multiple individuals / low image quality</td></tr>
<tr><td><b>R</b></td><td>Reset the track to the model prediction</td></tr>
<tr><td><b>Esc</b> in a taxon field</td><td>Undo the typing</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save now (saving is automatic as well)</td></tr>
</table>

<h3>Crops of the current track</h3>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td width="26%"><b>Double-click</b> a crop</td>
<td>Open the large single view; <b>Esc</b> returns to the grid (or use the <i>Grid</i> / <i>Single</i>
buttons)</td></tr>
<tr><td><b>Left / Right</b>, wheel</td>
<td>Previous / next crop in the single view; drag to move a magnified crop</td></tr>
<tr><td><b>+ / -</b>, <b>Ctrl+wheel</b></td>
<td>Zoom the grid or the single view (crops are enlarged in whole pixel steps)</td></tr>
<tr><td><b>Ctrl+0</b> / <b>Ctrl+9</b></td>
<td>Single view at 1:1 (one image pixel per screen pixel) / as large as fits</td></tr>
<tr><td><b>Space</b></td><td>Play / pause the crops of the track (switches to the single view)</td></tr>
</table>

<h3>Window</h3>
<table cellpadding="4" cellspacing="0" width="100%">
<tr><td width="26%"><b>F1</b></td><td>This help</td></tr>
</table>
"""

# Statistics dialog: labels of the verdict distribution's categories (verdict types and flags, see verdict_table())
_CATEGORY_LABELS = {
    **VERDICT_LABELS, "uncertain": "Uncertain (inc.)", "aggregate": "Aggregate (agg.)",
    "linked": "Linked as the same individual", **MULTIPLE_INDIVIDUALS_LABELS,
    DIFFERENT_TAXA_IN_TRACK: f"{MULTIPLE_INDIVIDUALS_LABELS[DIFFERENT_TAXA_IN_TRACK]} (tracking error)",
    "low_quality": "Low image quality",
}

# Statistics dialog: the exported table each table shows (its header tooltips, see column_description())
_STATS_TABLE_EXPORTS = {
    "strata": "progress_by_stratum", "verdicts": "verdict_distribution", "agreement": "agreement_by_rank",
    "reliability": "reliability_by_prob_bin", "taxa": "per_taxon", "confusions": "confusion_pairs",
    "linked": "linked_track_conflicts", "validators": "other_sessions",
}

# Statistics dialog: header tooltips of the dialog's own columns, by table
_STATS_TABLE_TIPS = {
    "strata": {"n_agree": "Validated tracks of this stratum the validator confirmed",
               "n_partial": "Tracks confirmed only down to a rank above the one the run was classified to",
               "n_corrected": "Tracks the validator determined as another taxon",
               "share_agree": "Share of the validated tracks of this stratum the validator confirmed"},
    "verdicts": {"category": "Verdict type or flag; a track has one verdict type and any number of flags",
                 "share": "Share of the validated tracks"},
}


class RunLoadWorker(QObject):
    """Loads a run (and optionally the taxonomy index) in a background thread.

    Signals:
        status(str): Step without measurable progress.
        download_progress(int, str): Taxonomy file download, percent and status text.
        finished(object): '(ValidationRun, TaxonomyIndex | None)'.
        error(str): Error message.
    """
    status = Signal(str)
    download_progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, run_root: Path, load_taxonomy: bool, allow_download: bool) -> None:
        """Initialize the worker.

        Args:
            run_root: Run directory (or a directory above a single run).
            load_taxonomy: Also build the taxonomy index.
            allow_download: Download the taxonomy file if it is missing.
        """
        super().__init__()
        self._run_root = run_root
        self._load_taxonomy = load_taxonomy
        self._allow_download = allow_download
        self._cancelled = False

    def cancel(self) -> None:
        """Stop the search for the run."""
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            run = ValidationRun.open(self._run_root, should_stop=lambda: self._cancelled)
            taxonomy: TaxonomyIndex | None = None
            if self._load_taxonomy:
                if TaxonomyIndex.mapping_file_available() or self._allow_download:
                    TaxonomyIndex.ensure_mapping_file(
                        lambda current, total, message: self.download_progress.emit(
                            int(100 * current / max(total, 1)), message)
                    )
                    self.status.emit("Loading taxonomy list...")
                    taxonomy = TaxonomyIndex.from_species_mapping()
                else:
                    taxonomy = TaxonomyIndex()
            self.finished.emit((run, taxonomy))
        except RunSearchCancelled:
            logger.debug("Search for a run in '%s' cancelled", self._run_root)
        except ValidationRunError as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("Loading run failed")
            self.error.emit(f"Loading failed: {e}")


class SetupPage(QWidget):
    """First page: pick the run and validator, select and subsample the tracks, then start or resume a session.

    Signals:
        session_ready(object): New or resumed 'ValidationSession'.
        session_deleted(object): 'Path' of the deleted session's CSV.
        theme_changed(str): Selected theme.
    """
    session_ready = Signal(object)
    session_deleted = Signal(object)
    theme_changed = Signal(str)

    def __init__(self, standalone: bool = False, parent: QWidget | None = None) -> None:
        """Initialize the setup page.

        Args:
            standalone: Started on its own, not from the processing GUI (shows the update button).
            parent: Parent widget.
        """
        super().__init__(parent)
        self.standalone = standalone
        self.update_btn: UpdateButton | None = None
        self.run: ValidationRun | None = None
        self.taxonomy: TaxonomyIndex | None = None
        self._setting_counts = setting_counts(pl.DataFrame())
        self._sessions: dict[int, dict[str, Any]] = {}
        self._summary_error: str | None = None
        self._thread: QThread | None = None
        self._worker: RunLoadWorker | None = None
        self._updating = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._update_preview)
        self._create_ui()
        self._restore_settings()

    def _create_ui(self) -> None:
        """Build the boxes of the setup page and connect their signals."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Data
        self.data_form = ParameterForm("data")
        self.data_form.set_flat(True)
        self.run_param = PathParameter("run_root")
        self.run_param.set_label("Run Directory:")
        self.run_param.set_method(PathParameter.Method.EXISTING_DIR)
        self.run_param.setToolTip("Output directory of a post-processing run "
                                  "(data_processed/<timestamp>_<source>_processed)")
        self.data_form.add_parameter(self.run_param)
        self.validator_param = StringParameter("validator")
        self.validator_param.set_label("Validator:")
        self.validator_param.set_placeholder("Your name or initials (used for the result files)")
        self.data_form.add_parameter(self.validator_param)
        self.blind_param = BoolParameter("blind")
        self.blind_param.set_label("Blind determination:")
        self.blind_param.setToolTip("Hide the model prediction until you have entered a determination for a track; "
                                    "Agree and the rank buttons stay disabled")
        self.data_form.add_parameter(self.blind_param)
        self.summary_label = QLabel("Select a run directory.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.data_form.add_widget(self.summary_label, column=1, column_span=2)
        taxonomy_row = QHBoxLayout()
        self.download_btn = StyledButton("Download taxonomy list", COLOR_BTN_UPDATE)
        self.download_btn.setToolTip("All arthropod species of the BioCLIP Tree of Life, mapped to GBIF names, for "
                                     "the taxon dropdowns. Without it, only taxa predicted in this run are listed.")
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self._download_taxonomy)
        taxonomy_row.addWidget(self.download_btn, 0)
        self.taxonomy_label = QLabel("")
        self.taxonomy_label.setWordWrap(True)
        taxonomy_row.addWidget(self.taxonomy_label, 1)
        self.data_form.add_layout(taxonomy_row, column=1, column_span=2)
        self.load_progress = QProgressBar()
        self.load_progress.setVisible(False)
        self.data_form.add_widget(self.load_progress, column=1, column_span=2)
        layout.addWidget(create_groupbox("Processed data", COLOR_GROUP_PROCESSING, self.data_form))

        # Sessions
        sessions_widget = QWidget()
        sessions_layout = QVBoxLayout(sessions_widget)
        sessions_layout.setContentsMargins(6, 6, 6, 6)
        self.sessions_table = QTableWidget(0, 6)
        self.sessions_table.setHorizontalHeaderLabels(
            ["Validator", "Started", "Last update", "Validated / sampled", "Selection", "Subsample"]
        )
        for col in (0, 1, 2, 3):
            self.sessions_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col in (4, 5):  # the Selection and Subsample columns share the remaining width
            self.sessions_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
            _header_item(self.sessions_table, col).setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.sessions_table.verticalHeader().setVisible(False)
        self.sessions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sessions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sessions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sessions_table.setMaximumHeight(140)
        self.sessions_table.itemDoubleClicked.connect(lambda _: self._resume_selected())
        sessions_layout.addWidget(self.sessions_table)
        sessions_buttons = QHBoxLayout()
        self.resume_btn = StyledButton("Resume selected session", COLOR_BTN_RUN)
        self.resume_btn.clicked.connect(self._resume_selected)
        sessions_buttons.addWidget(self.resume_btn)
        self.use_settings_btn = StyledButton("Use these settings", COLOR_BTN_UPDATE)
        self.use_settings_btn.setToolTip("Copy the selection, subsample and blind mode of the selected session "
                                         "into the boxes below, to start a new session with the same settings")
        self.use_settings_btn.clicked.connect(self._use_settings)
        sessions_buttons.addWidget(self.use_settings_btn)
        self.delete_btn = StyledButton("Delete selected session", COLOR_BTN_CANCEL)
        self.delete_btn.setToolTip("Remove the session with its verdicts, history and exported statistics "
                                   "(the merged export in the metadata directory is kept)")
        self.delete_btn.clicked.connect(self._delete_selected)
        sessions_buttons.addWidget(self.delete_btn)
        sessions_buttons.addStretch()
        self.sessions_hint = QLabel("No sessions yet for this run.")
        sessions_buttons.addWidget(self.sessions_hint)
        sessions_layout.addLayout(sessions_buttons)
        self.session_buttons = (self.resume_btn, self.use_settings_btn, self.delete_btn)
        for button in self.session_buttons:
            button.setEnabled(False)
        self.sessions_table.itemSelectionChanged.connect(self._update_session_buttons)
        self.sessions_box = create_groupbox(
            "Existing sessions", COLOR_GROUP_METADATA, sessions_widget  # type: ignore[arg-type]
        )
        layout.addWidget(self.sessions_box)

        # Track selection
        self.filter_form = ParameterForm("filter")
        self.filter_form.set_flat(True)
        self.taxa_rank_param = ComboParameter("taxa_rank")
        self.taxa_rank_param.set_label("Filter taxa at rank")
        self.taxa_rank_param.setToolTip("Only validate tracks whose prediction at this rank is selected below")
        self.filter_form.add_parameter(self.taxa_rank_param, checkable=True)
        self.taxa_values_param = MultiComboParameter("taxa_values")
        self.taxa_values_param.set_label("Taxa")
        self.taxa_values_param.set_placeholder("Select taxa...")
        self.filter_form.add_parameter(self.taxa_values_param)
        taxa_check = self.filter_form.widgets()["taxa_rank_enabled"]
        self.taxa_values_param.setEnabled(taxa_check.value())
        taxa_check.value_changed.connect(self.taxa_values_param.setEnabled)
        self.settings_values_param = MultiComboParameter("setting_values")
        self.settings_values_param.set_label("Filter camera settings")
        self.settings_values_param.set_placeholder("Select camera settings...")
        self.settings_values_param.setToolTip("Only validate tracks of sessions with one of these camera settings; "
                                              f"{NONE_LABEL} = sessions without a recorded setting")
        self.filter_form.add_parameter(self.settings_values_param, checkable=True)
        self.settings_filter_check = self.filter_form.widgets()["setting_values_enabled"]
        self.settings_filter_check.setToolTip(self.settings_values_param.toolTip())
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        self.filter_form.add_widget(self.selection_label, column=1, column_span=2)
        self.selection_box = create_groupbox("Track selection", COLOR_GROUP_CLASSIFICATION, self.filter_form)
        layout.addWidget(self.selection_box)

        # Stratified subsample
        self.sampling_form = ParameterForm("sampling")
        self.sampling_form.set_flat(True)
        self.stratum_rank_param = ComboParameter("stratum_rank")
        self.stratum_rank_param.set_label("Strata by taxon rank")
        self.stratum_rank_param.setToolTip("Sample separately per predicted taxon at this rank")
        self.sampling_form.add_parameter(self.stratum_rank_param, checkable=True)
        self.prob_edges_param = StringParameter("prob_edges")
        self.prob_edges_param.set_label("Strata by probability bins")
        self.prob_edges_param.set_default(", ".join(f"{e:g}" for e in VALIDATION_PROB_BIN_EDGES_DEFAULT))
        self.prob_edges_param.set_value(self.prob_edges_param.default())
        self.prob_edges_param.setToolTip("Bin edges of the weighted prediction probability from 0 to 1, "
                                         "e.g. 0, 0.2, 0.4, 0.6, 0.8, 1 (a bin includes its lower edge)")
        self.sampling_form.add_parameter(self.prob_edges_param, checkable=True)
        self.setting_param = StringParameter("stratum_setting")
        self.setting_param.set_label("Strata by camera setting")
        self.setting_param.text.setReadOnly(True)  # lists the run's settings
        self.setting_param.text.setFrame(False)  # type: ignore[union-attr]
        self.sampling_form.add_parameter(self.setting_param, checkable=True)
        self.setting_check = self.sampling_form.widgets()["stratum_setting_enabled"]
        setting_tip = ("Sample separately per camera setting (the platform or flower species recorded per "
                       "session); needs at least two settings in the run")
        self.setting_param.setToolTip(setting_tip)
        self.setting_check.setToolTip(setting_tip)
        self.min_param = create_num_param(
            "min_per_stratum", "Min. tracks per stratum", 1, 5000, VALIDATION_MIN_PER_STRATUM_DEFAULT,
            tooltip="Strata with fewer tracks are excluded from the validation sample (shown grayed out in "
                    "the table). Weighted results then refer to the remaining strata."
        )
        self.sampling_form.add_parameter(self.min_param, checkable=True)
        self.n_param = create_num_param(
            "max_per_stratum", "Max. tracks per stratum", 1, 5000, VALIDATION_MAX_PER_STRATUM_DEFAULT,
            tooltip="Random sample size per stratum (all tracks of a stratum if it is smaller). "
                    "Unchecked: validate every track."
        )
        self.sampling_form.add_parameter(self.n_param, checkable=True)
        self.seed_param = create_num_param(
            "seed", "Random seed", 0, 99999, VALIDATION_SEED_DEFAULT,
            tooltip="Seed of the random sample: the same seed reproduces the same subsample"
        )
        self.seed_param.set_slider_visible(False)
        self.sampling_form.add_parameter(self.seed_param)
        # Same width for all number fields
        field_width = max(p.line.sizeHint().width() for p in (self.min_param, self.n_param, self.seed_param))
        for number_param in (self.min_param, self.n_param, self.seed_param):
            number_param.line.setFixedWidth(field_width)
        # Keep the seed field left-aligned
        self.seed_param.layout().addStretch(1)  # type: ignore[union-attr]
        self.preview_table = QTableWidget(0, 5)
        self.preview_table.setHorizontalHeaderLabels(["Camera setting", "Stratum taxon", "Probability bin",
                                                      "Tracks in run", "Sampled"])
        # Content-width columns
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        for col in range(5):
            self.preview_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.preview_table.setColumnHidden(0, True)  # shown for runs with several settings
        _header_item(self.preview_table, 3).setToolTip(
            "Selected tracks in this stratum (its population for the weighted statistics)")
        _header_item(self.preview_table, 4).setToolTip("Tracks drawn from this stratum for validation")
        for col in (0, 1, 2):  # left-aligned over the data
            _header_item(self.preview_table, col).setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for col in (3, 4):
            _header_item(self.preview_table, col).setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.preview_table.setMinimumHeight(160)
        self.preview_table.setMaximumHeight(260)
        preview_row = QHBoxLayout()  # spare width outside the table
        preview_row.addWidget(self.preview_table)
        preview_row.addStretch(1)
        self.sampling_form.add_layout(preview_row, column=1, column_span=2)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.sampling_form.add_widget(self.preview_label, column=1, column_span=2)
        self.sampling_box = create_groupbox("Stratified subsample", COLOR_GROUP_VALIDATION, self.sampling_form)
        layout.addWidget(self.sampling_box)
        layout.addStretch(1)

        # Start
        buttons = QHBoxLayout()
        buttons.setContentsMargins(6, 0, 6, 0)
        self.start_btn = StyledButton("Start new validation session", COLOR_BTN_VALIDATE, font_size=10)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_new)
        buttons.addWidget(self.start_btn)
        buttons.addStretch()
        if self.standalone:
            self.update_btn = UpdateButton(self._restart_args)
            self.update_btn.install_started.connect(self._on_install_start)
            self.update_btn.install_finished.connect(self._on_install_finish)
            buttons.addWidget(self.update_btn)
            self.update_btn.start_check()

        # Theme selector (right-aligned)
        self.theme_select = QComboBox()
        self.theme_select.addItems(sorted(qt_themes.get_themes()))
        self.theme_select.setCurrentText(load_theme())
        self.theme_select.setToolTip("Select the color theme of the application")
        self.theme_select.setFixedWidth(150)
        self.theme_select.currentTextChanged.connect(self.theme_changed.emit)
        buttons.addWidget(QLabel("Theme:"))
        buttons.addWidget(self.theme_select)
        outer.addLayout(buttons)

        self.run_param.value_changed.connect(self._on_run_path_change)
        # React per keystroke (value_changed fires only when editing is finished)
        self.validator_param.text.textChanged.connect(lambda _: self._update_start_state())
        self.filter_form.parameter_changed.connect(lambda _: self._preview_timer.start())
        self.sampling_form.parameter_changed.connect(lambda _: self._preview_timer.start())
        # The counts of each filter list follow the other filter
        for widget in (self.taxa_rank_param, self.taxa_values_param, self.filter_form.widgets()["taxa_rank_enabled"],
                       self.settings_values_param, self.settings_filter_check):
            widget.value_changed.connect(lambda _: self._refresh_filter_lists())
        self._set_options_enabled(False)

    def _restore_settings(self) -> None:
        """Restore the last validator name."""
        settings = _settings()
        validator = settings.value(_SETTINGS_KEY_VALIDATOR, "")
        if isinstance(validator, str):
            self.validator_param.set_value(validator)

    def _set_options_enabled(self, enabled: bool) -> None:
        """Enable the selection and subsample boxes (only with a loaded run)."""
        self.selection_box.setEnabled(enabled)
        self.sampling_box.setEnabled(enabled)
        if not enabled:
            # Drop the previous run's counts
            self.selection_label.clear()
            self.preview_label.clear()
            self.preview_table.setRowCount(0)

    def set_run_root(self, run_root: Path | str) -> None:
        """Select a run directory and load it."""
        if self.run_param.value() == str(run_root):
            self._on_run_path_change(str(run_root))
        else:
            self.run_param.set_value(str(run_root))  # emits value_changed -> _on_run_path_change

    def _on_run_path_change(self, text: str) -> None:
        """Load the run of a changed directory, or show why it cannot be used."""
        path = resolve_and_validate_path(text)
        self.run = None
        self._setting_counts = setting_counts(pl.DataFrame())
        self.sessions_table.setRowCount(0)
        self.start_btn.setEnabled(False)
        self._set_options_enabled(False)
        if path is None:
            self._cleanup_thread()  # stop a search for the previous directory
            self.load_progress.setVisible(False)
            if not text.strip():
                self._set_summary("Select a run directory.")
            else:
                self._set_summary("Directory not found.", error=True)
            return
        self._start_loading(path, allow_download=False)

    def _start_loading(self, path: Path, allow_download: bool) -> None:
        """Start loading a run (and the taxonomy index) in the loader thread."""
        self._cleanup_thread()
        self.load_progress.setRange(0, 0)  # busy until a download reports progress
        self.load_progress.setVisible(True)
        self.summary_label.setText(f"Loading <i>{path.name}</i> ...")
        self.download_btn.setEnabled(False)
        if self.update_btn is not None:  # no update while loading
            self.update_btn.set_busy(True)
        self._thread = QThread()
        self._worker = RunLoadWorker(path, load_taxonomy=self.taxonomy is None, allow_download=allow_download)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_load_status)
        self._worker.download_progress.connect(self._on_download_progress)
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_load_error)
        self._thread.start()

    def _cleanup_thread(self) -> None:
        """Stop the loader thread and release it and its worker."""
        if self._worker is not None:
            self._worker.cancel()  # ends the walk at the next directory
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()  # never delete a running thread
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self.update_btn is not None:
            self.update_btn.set_busy(False)

    @Slot(str)
    def _on_load_status(self, message: str) -> None:
        """Show a loading step without measurable progress (busy bar)."""
        self.load_progress.setRange(0, 0)
        self.summary_label.setText(message)

    @Slot(int, str)
    def _on_download_progress(self, percent: int, message: str) -> None:
        """Show the progress of the taxonomy download."""
        self.load_progress.setRange(0, 100)
        self.load_progress.setValue(percent)
        self.summary_label.setText(message)

    @Slot(object)
    def _on_loaded(self, payload: tuple[ValidationRun, TaxonomyIndex | None]) -> None:
        """Take over a loaded run: summary, options, sessions and preview."""
        if self.sender() is not self._worker:
            return  # a load superseded by another directory: its queued result is not wanted
        self._cleanup_thread()
        run, taxonomy = payload
        self.run = run
        self._setting_counts = setting_counts(run.tracks)
        if taxonomy is not None:
            self.taxonomy = taxonomy
        if self.taxonomy is None:
            self.taxonomy = TaxonomyIndex()
        # Make every predicted taxon selectable
        lineages = [
            self.taxonomy.canonical_lineage(run.model_lineage(row), row.get("gbif_species"))
            for row in run.tracks.unique(subset=[f"bioclip_{r}" for r in run.active_ranks], maintain_order=True)
            .to_dicts()
        ]
        self.taxonomy.add_lineages(lineages, "model")
        self.load_progress.setVisible(False)
        self.download_btn.setEnabled(True)
        _settings().setValue(_SETTINGS_KEY_LAST_RUN, str(run.root))
        self._show_summary()
        self._populate_sampling_options()
        self._refresh_sessions()
        self._set_options_enabled(True)
        self._update_preview()
        self._update_start_state()

    @Slot(str)
    def _on_load_error(self, message: str) -> None:
        """Show why the run could not be loaded."""
        if self.sender() is not self._worker:
            return  # a load superseded by another directory: its queued result is not wanted
        self._cleanup_thread()
        self.load_progress.setVisible(False)
        self.download_btn.setEnabled(True)
        self._set_summary(message, error=True)
        logger.error(message)

    def _download_taxonomy(self) -> None:
        """Reload the run with the taxonomy file downloaded."""
        if self.run is not None:
            self.taxonomy = None
            self._start_loading(self.run.root, allow_download=True)

    def _set_summary(self, html: str, error: bool = False) -> None:
        """Show 'html' in the summary line (errors are re-colored on a theme change)."""
        self._summary_error = html if error else None
        self.summary_label.setText(f"<span style='color:{_error_color()}'>{html}</span>" if error else html)

    def _show_summary(self) -> None:
        """Show the run summary and the status of the taxonomy list."""
        assert self.run is not None and self.taxonomy is not None
        info = self.run.summary()
        plural = {"class": "classes", "order": "orders", "family": "families", "genus": "genera", "species": "species"}
        taxa = " · ".join(f"{n} {plural[r] if n != 1 else r}" for r, n in info["taxa_per_rank"].items())
        quantiles = info["prob_quantiles"]
        prob_text = (f"median {quantiles.get(0.5)} (10 % of the tracks below {quantiles.get(0.1)}, "
                     f"90 % below {quantiles.get(0.9)})") if quantiles else "-"
        tracks_text, filter_text = self._filter_texts(info)
        self._set_summary("<br>".join([
            f"<b>{info['run_name']}</b>",
            f"{tracks_text} · {info['n_crops']} crop{'' if info['n_crops'] == 1 else 's'} · classified to "
            f"<b>{info['model_rank']}</b> level · "
            + ("size estimates available" if info["has_size_estimate"] else "no size estimates"),
            f"Predicted taxa: {taxa or '-'}",
            f"Weighted probability: {prob_text}",
            filter_text,
            f"Camera settings: {self._settings_text() or 'none recorded'}",
        ]))
        if self.taxonomy.uses_species_mapping:
            self.taxonomy_label.setText(
                f"Taxonomy list: BioCLIP Tree of Life matched against GBIF, {self.taxonomy.n_species:,} species "
                "(dropdowns + type-ahead, free text allowed)."
            )
            self.download_btn.setVisible(False)
        else:
            self.taxonomy_label.setText(
                f"<span style='color:{_error_color()}'>Taxonomy list not available</span> - dropdowns only list "
                "taxa predicted in this run (free text is always possible)."
            )
            self.download_btn.setVisible(True)

    @classmethod
    def _filter_texts(cls, info: dict[str, Any]) -> tuple[str, str]:
        """Return '(track count, run filters line)' of the run summary."""
        kept, classified = info["n_tracks"], info["n_tracks_classified"]
        missing = max(classified - kept, 0)
        tracks_text = (f"<b>{kept}</b> of {classified} tracks" if missing
                       else f"<b>{kept}</b> track{'' if kept == 1 else 's'}")
        filters = info["filters"]
        gap = (f"{missing} of {classified} classified tracks are not in the results" if missing
               else "all classified tracks are in the results")
        if filters is None:
            return tracks_text, f"Run filters: unknown (the run has no config snapshot) - {gap}"
        conditions = cls._filter_conditions(filters)
        if not conditions:
            return tracks_text, f"Run filters: none - {gap}"
        outcome = f"{missing} of {classified} classified tracks removed" if missing else "no track removed"
        return tracks_text, f"Run filters: kept tracks with {' · '.join(conditions)} - {outcome}"

    @staticmethod
    def _filter_conditions(filters: dict[str, float]) -> list[str]:
        """Return the run's track filters as conditions (see ValidationRun.track_filters())."""
        conditions = []
        if "min_det_conf" in filters:
            conditions.append(f"mean detection confidence ≥ {filters['min_det_conf']:g}")
        low, high = filters.get("min_dur_s"), filters.get("max_dur_s")
        if low is not None and high is not None:
            conditions.append(f"duration {low:g}-{high:g} s")
        elif low is not None:
            conditions.append(f"duration ≥ {low:g} s")
        elif high is not None:
            conditions.append(f"duration ≤ {high:g} s")
        if "min_prob_weighted" in filters:
            conditions.append(f"weighted probability ≥ {filters['min_prob_weighted']:g}")
        return conditions

    def _settings_text(self, limit: int = 8) -> str:
        """Return the camera settings with their track counts (empty if none)."""
        rows = self._setting_counts.to_dicts()
        listed = " · ".join(f"{row[SETTING_COLUMN]} ({row['n_tracks']})" for row in rows[:limit])
        return listed + (f" · +{len(rows) - limit} more" if len(rows) > limit else "")

    def refresh_sessions(self) -> None:
        """Re-read the session list."""
        if self.run is not None:
            self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        """Fill the sessions table with the stored sessions of the run."""
        assert self.run is not None
        sessions = ValidationSession.list_sessions(self.run.root)
        self.sessions_table.setRowCount(len(sessions))
        self._sessions = dict(enumerate(sessions))
        for i, info in enumerate(sessions):
            sampling = info["sampling"]
            stale = source_csv_changed(info, self.run.final_provenance)
            cells = [
                str(info["validator"]) + ("   (blind)" if info["blind"] else ""),
                ("⚠ " if stale else "") + _fmt_time(info["created"]),
                _fmt_time(info["updated"]),
                f"{info['n_validated']} / {info['n_sampled']}",
                _fmt_selection(sampling),
                _fmt_subsample(sampling),
            ]
            tooltip = _session_tooltip(info)
            if stale:
                tooltip += (f"\n\nWarning: '{self.run.final_path.name}' has changed since this session was "
                            "started - the run was processed again, so its tracks may no longer match.")
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                if stale:
                    item.setForeground(QColor(COLOR_BTN_UPDATE))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(info["csv_path"]))
                self.sessions_table.setItem(i, col, item)
        self.sessions_hint.setText(
            "No sessions yet for this run." if not sessions else
            "A resumed session keeps its own selection and subsample."
        )
        self._update_session_buttons()

    def _update_session_buttons(self) -> None:
        """Enable the session buttons while a session is selected."""
        selected = bool(self.sessions_table.selectedItems())
        for button in self.session_buttons:
            button.setEnabled(selected)

    def _selected_session(self) -> dict[str, Any] | None:
        """Return the listing entry of the selected row, or None."""
        items = self.sessions_table.selectedItems()
        return self._sessions.get(items[0].row()) if items else None

    def _use_settings(self) -> None:
        """Put the selection, subsample and blind mode of the selected session into the form."""
        info = self._selected_session()
        if info is None:
            return
        sampling = SamplingConfig.from_dict(info["sampling"])
        self._updating = True
        self.settings_filter_check.set_value(bool(sampling.setting_filter_values)
                                             and self.settings_filter_check.isEnabled())
        self.settings_values_param.set_value(list(sampling.setting_filter_values))
        self.taxa_rank_param.set_value(sampling.taxon_filter_rank or self.taxa_rank_param.value())
        self._updating = False
        self._refresh_filter_lists()  # list the session's taxa before checking them
        self._updating = True
        filter_params, sampling_params = self.filter_form.widgets(), self.sampling_form.widgets()
        filter_params["taxa_rank_enabled"].set_value(bool(sampling.taxon_filter_rank))
        self.taxa_values_param.set_value(list(sampling.taxon_filter_values))
        sampling_params["stratum_rank_enabled"].set_value(bool(sampling.stratum_rank))
        if sampling.stratum_rank:
            self.stratum_rank_param.set_value(sampling.stratum_rank)
        sampling_params["prob_edges_enabled"].set_value(bool(sampling.prob_edges))
        if sampling.prob_edges:
            self.prob_edges_param.set_value(", ".join(f"{edge:g}" for edge in sampling.prob_edges))
        self.setting_check.set_value(bool(sampling.stratum_setting) and self.setting_check.isEnabled())
        sampling_params["max_per_stratum_enabled"].set_value(sampling.max_per_stratum is not None)
        if sampling.max_per_stratum is not None:
            sampling_params["max_per_stratum"].set_value(int(sampling.max_per_stratum))
        sampling_params["min_per_stratum_enabled"].set_value(sampling.min_per_stratum is not None)
        if sampling.min_per_stratum is not None:
            sampling_params["min_per_stratum"].set_value(int(sampling.min_per_stratum))
        sampling_params["seed"].set_value(int(sampling.seed))
        self.blind_param.set_value(bool(info["blind"]))
        self.validator_param.set_value(str(info["validator"]))
        self._updating = False
        self._refresh_filter_lists()  # recount for the session's taxa
        self._update_start_state()
        self.sessions_hint.setText("Settings copied below - 'Start validation' creates a new session with them.")

    def _delete_selected(self) -> None:
        """Delete the selected session after confirmation."""
        info = self._selected_session()
        if info is None:
            return
        csv_path = Path(info["csv_path"])
        try:
            paths = session_files(csv_path)
        except ValueError as e:
            QMessageBox.critical(self, "Delete session", str(e))
            return
        listed = "\n".join(f"  {path.name}" for path in paths) or "  (no files left)"
        reply = QMessageBox.question(
            self, "Delete session",
            f"Delete the session of '{info['validator']}' started {_fmt_time(info['created'])}?\n\n"
            f"{info['n_validated']} of {info['n_sampled']} tracks are validated in it. "
            f"These files are removed permanently:\n{listed}\n\n"
            "The merged export in the metadata directory is kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Close the session log file first
        self.session_deleted.emit(csv_path)
        try:
            removed = delete_session(csv_path)
        except OSError as e:
            QMessageBox.critical(self, "Delete session", f"Could not delete the session:\n\n{e}")
            return
        self._refresh_sessions()
        self.sessions_hint.setText(f"Session deleted ({len(removed)} files).")

    def _resume_selected(self) -> None:
        """Resume the selected session (after a warning if the run has changed)."""
        info = self._selected_session()
        if self.run is None or info is None:
            return
        csv_path = info["csv_path"]
        if source_csv_changed(info, self.run.final_provenance) and QMessageBox.warning(
            self, "Run was processed again",
            f"'{self.run.final_path.name}' has changed since this session was started "
            f"({_fmt_time(info['created'])}).\n\n"
            "The sampled tracks may no longer be the ones the session refers to, and its statistics can be "
            "misleading. Resume anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            session = ValidationSession.load(Path(csv_path))
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Load error", f"Could not load the session:\n\n{e}")
            return
        if self.taxonomy is not None:
            for entry in session.meta["custom_taxa"]:
                self.taxonomy.add_lineages([{r: entry.get(r) for r in VALIDATION_RANKS}], "free_text")
        _settings().setValue(_SETTINGS_KEY_VALIDATOR, session.validator)
        self.validator_param.set_value(session.validator)
        self.session_ready.emit(session)

    def _populate_sampling_options(self) -> None:
        """Fill the selection and subsample options for the loaded run."""
        assert self.run is not None
        self._updating = True
        rank_items = [(_RANK_LABELS[r], r) for r in self.run.active_ranks if r != "class"] or [("Class", "class")]
        self.taxa_rank_param.set_items(rank_items)
        default_rank = "family" if "family" in self.run.active_ranks else self.run.active_ranks[-1]
        self.taxa_rank_param.set_value(default_rank)
        strata_ranks = available_strata_ranks(self.run.active_ranks, self.run.model_rank)
        self.stratum_rank_param.set_items([(_RANK_LABELS[r], r) for r in strata_ranks] or [("Order", "order")])
        if "family" in strata_ranks:
            self.stratum_rank_param.set_value("family")
        # Setting filter and strata need at least two setting groups
        settings_available = setting_strata_available(self.run.tracks)
        self.settings_filter_check.set_value(False)
        self.settings_filter_check.setEnabled(settings_available)
        self.settings_values_param.set_items({f"{row[SETTING_COLUMN]} ({row['n_tracks']})": row[SETTING_COLUMN]
                                              for row in self._setting_counts.to_dicts()})
        self.settings_values_param.set_value([])
        n_settings = self._setting_counts.height
        self.settings_values_param.set_placeholder(
            "Select camera settings..." if settings_available else
            "only one camera setting in this run" if n_settings else "no camera setting recorded in this run")
        self.setting_check.set_value(False)
        self._sync_setting_strata()
        self.preview_table.setColumnHidden(0, not settings_available)
        self._updating = False
        self._refresh_filter_lists()

    def _refresh_filter_lists(self) -> None:
        """Recount the taxa and setting lists, each for the tracks the other filter selects.

        Every entry stays listed (possibly with 0), so checked entries survive.
        """
        if self.run is None or self._updating:
            return
        self._updating = True  # no re-entry from the lists' own signals
        try:
            self._sync_setting_strata()
            n_setting = dict(setting_counts(filter_taxa(self.run.tracks, *self._taxa_filter())).iter_rows())
            settings = sorted(self._setting_counts[SETTING_COLUMN].to_list(),
                              key=lambda setting: (-n_setting.get(setting, 0), setting))
            self.settings_values_param.set_items({f"{s} ({n_setting.get(s, 0)})": s for s in settings})
            rank = self.taxa_rank_param.value()
            counts = self.run.taxon_counts(rank) if rank else None
            if counts is None or counts.is_empty():
                self.taxa_values_param.set_items([])
            else:
                selected = self.run.taxon_counts(rank, filter_settings(self.run.tracks, self._setting_filter()))
                n_taxon = dict(selected.iter_rows())
                taxa = sorted(counts["taxon"].to_list(), key=lambda taxon: (-n_taxon.get(taxon, 0), taxon))
                self.taxa_values_param.set_items({f"{taxon} ({n_taxon.get(taxon, 0)})": taxon for taxon in taxa})
        finally:
            self._updating = False
        self._update_preview()

    def _taxa_filter(self) -> tuple[str | None, list[str] | None]:
        """Return '(rank, taxa)' of the taxa filter, '(None, None)' if off."""
        if not self.filter_form.widgets()["taxa_rank_enabled"].value():
            return None, None
        return self.taxa_rank_param.value(), list(self.taxa_values_param.value()) or None

    def _setting_filter(self) -> list[str]:
        """Return the selected camera settings (empty if the filter is off)."""
        return list(self.settings_values_param.value()) if self.settings_filter_check.value() else []

    def _sync_setting_strata(self) -> None:
        """Offer setting strata only while the selected settings form at least two groups."""
        assert self.run is not None
        available = setting_strata_available(filter_settings(self.run.tracks, self._setting_filter()))
        self.setting_check.setEnabled(available)
        if not available and self.setting_check.value():
            self.setting_check.set_value(False)
        n_settings = self._setting_counts.height
        self.setting_param.set_value(
            self._settings_text() if available and not self._setting_filter() else
            " · ".join(self._setting_filter()) if available else
            "only one camera setting selected" if n_settings >= 2 else
            "only one camera setting in this run" if n_settings else "no camera setting recorded in this run"
        )

    def _sampling_config(self) -> SamplingConfig | None:
        """Return the sampling parameters from the form. None (with a hint) if invalid."""
        values = {**self.filter_form.values(), **self.sampling_form.values()}
        cfg = SamplingConfig(seed=int(values.get("seed", VALIDATION_SEED_DEFAULT)))
        if values.get("taxa_rank_enabled"):
            cfg.taxon_filter_rank = values.get("taxa_rank")
            cfg.taxon_filter_values = list(values.get("taxa_values") or ())
        if values.get("setting_values_enabled"):
            cfg.setting_filter_values = list(values.get("setting_values") or ())
        if values.get("stratum_rank_enabled"):
            cfg.stratum_rank = values.get("stratum_rank")
        cfg.stratum_setting = bool(values.get("stratum_setting_enabled"))
        if values.get("prob_edges_enabled"):
            try:
                cfg.prob_edges = parse_prob_edges(str(values.get("prob_edges", "")))
            except ValueError as e:
                self.preview_label.setText(f"<span style='color:{_error_color()}'>{e}</span>")
                return None
        if values.get("max_per_stratum_enabled"):
            cfg.max_per_stratum = int(values.get("max_per_stratum", VALIDATION_MAX_PER_STRATUM_DEFAULT))
        if values.get("min_per_stratum_enabled"):
            cfg.min_per_stratum = int(values.get("min_per_stratum", VALIDATION_MIN_PER_STRATUM_DEFAULT))
        return cfg

    def _sample(self, cfg: SamplingConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return '(sampled tracks, stratum summary)' of a sampling config."""
        assert self.run is not None
        tracks = filter_taxa(self.run.tracks, cfg.taxon_filter_rank, cfg.taxon_filter_values or None)
        tracks = filter_settings(tracks, cfg.setting_filter_values or None)
        with_strata = assign_strata(tracks, cfg.stratum_rank, cfg.prob_edges, by_setting=cfg.stratum_setting)
        summary = stratum_summary(with_strata)
        sampled = sample_strata(with_strata, cfg.max_per_stratum, cfg.seed, cfg.min_per_stratum)
        return sampled, summary

    def _update_preview(self) -> None:
        """Recompute the subsample preview table and its summary line."""
        if self.run is None or self._updating:
            return
        cfg = self._sampling_config()
        if cfg is None:
            self.preview_table.setRowCount(0)
            self.start_btn.setEnabled(False)
            return
        sampled, summary = self._sample(cfg)
        sampled_counts = dict(sampled.group_by("stratum_id").len().iter_rows()) if not sampled.is_empty() else {}
        excluded = set(small_strata(summary, cfg.min_per_stratum))
        grayed = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
        self.preview_table.setRowCount(summary.height + 1)
        for i, row in enumerate(summary.to_dicts()):
            is_excluded = row["stratum_id"] in excluded
            n_sampled = sampled_counts.get(row["stratum_id"], 0)
            cells = [row["stratum_setting"] or "(all)", row["stratum_taxon"] or "(all)",
                     row["stratum_prob_bin"] or "(all)", str(row["n_population"]),
                     "excluded" if is_excluded else str(n_sampled)]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j >= 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if is_excluded:
                    item.setForeground(grayed)
                    item.setToolTip(f"Fewer than {cfg.min_per_stratum} tracks: not validated")
                self.preview_table.setItem(i, j, item)
        total_row = summary.height
        bold = QFont()
        bold.setBold(True)
        for j, text in enumerate(["Total", "", "", str(int(summary["n_population"].sum()) if summary.height else 0),
                                  str(sampled.height)]):
            item = QTableWidgetItem(text)
            item.setFont(bold)
            if j >= 3:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.preview_table.setItem(total_row, j, item)
        n_run = self.run.tracks.height
        n_selected = int(summary["n_population"].sum()) if summary.height else 0
        self.selection_label.setText(
            f"All <b>{n_run}</b> tracks of the run are selected." if n_selected == n_run else
            f"<b>{n_selected}</b> of the {n_run} tracks of the run are selected."
        )
        n_used = summary.height - len(excluded)
        strata_note = (f", drawn from {n_used} of {summary.height} strata" if excluded else
                       f", drawn from {summary.height} strata" if summary.height > 1 else "")
        if excluded:
            n_dropped = int(summary.filter(pl.col("stratum_id").is_in(list(excluded)))["n_population"].sum())
            excluded_note = (f" {len(excluded)} strata with fewer than {cfg.min_per_stratum} tracks "
                             f"({n_dropped} tracks) are excluded.")
        else:
            excluded_note = ""
        if not sampled.height:
            self.preview_label.setText(
                f"<span style='color:{_error_color()}'>No stratum reaches the minimum of "
                f"{cfg.min_per_stratum} tracks - nothing to validate.</span>" if excluded else
                f"<span style='color:{_error_color()}'>The current selection contains no tracks.</span>"
            )
        else:
            self.preview_label.setText(
                (f"No subsample: all <b>{n_selected}</b> selected tracks will be validated."
                 if sampled.height == n_selected else
                 f"<b>{sampled.height}</b> of the {n_selected} selected tracks will be validated{strata_note}.")
                + excluded_note
            )
        self._update_start_state(sampled.height)

    def _validator_name(self) -> str:
        """Return the validator name as typed (value() only updates when editing finishes)."""
        return self.validator_param.text.text().strip()  # type: ignore[union-attr]

    def _update_start_state(self, n_sampled: int | None = None) -> None:
        """Enable the start button with a run, a validator and a non-empty sample."""
        ready = self.run is not None and bool(self._validator_name())
        if ready and n_sampled is not None:
            ready = n_sampled > 0
        self.start_btn.setEnabled(ready)

    def _start_new(self) -> None:
        """Create, save and open a new session with the current settings."""
        if self.run is None or self.taxonomy is None:
            return
        validator = self._validator_name()
        if not validator:
            QMessageBox.warning(self, "Validator missing", "Please enter a validator name.")
            return
        cfg = self._sampling_config()
        if cfg is None:
            return
        sampled, _ = self._sample(cfg)
        if sampled.is_empty():
            QMessageBox.warning(self, "No tracks", "The current selection contains no tracks.")
            return
        # Every session gets its own id and files
        session = ValidationSession.create(self.run, validator, sampled, cfg, self.taxonomy,
                                           blind=bool(self.blind_param.value()))
        _settings().setValue(_SETTINGS_KEY_VALIDATOR, validator)
        try:
            session.save()
        except OSError as e:
            QMessageBox.critical(self, "Save error", f"Could not create the session files:\n\n{e}")
            return
        self.session_ready.emit(session)

    def set_theme_name(self, theme_name: str) -> None:
        """Show 'theme_name' in the selector without emitting theme_changed."""
        if self.theme_select.currentText() != theme_name:
            self.theme_select.blockSignals(True)
            self.theme_select.setCurrentText(theme_name)
            self.theme_select.blockSignals(False)

    def refresh_theme(self) -> None:
        """Re-render texts and items with theme colors."""
        if self._summary_error is not None:
            self._set_summary(self._summary_error, error=True)
        elif self.run is not None and self.taxonomy is not None:
            self._show_summary()
        if self.run is not None:
            self._refresh_sessions()  # the stale-run marker is an item color
            self._update_preview()  # the excluded strata use the disabled palette color

    @property
    def update_installing(self) -> bool:
        """Whether an update is being installed."""
        return self.update_btn is not None and self.update_btn.is_installing

    def _restart_args(self) -> list[str]:
        """Return the arguments that relaunch the validation window on the current run."""
        args = ["-m", "insectdetect_post.validation_gui"]
        run_root = self.run_param.value()
        return [*args, run_root] if run_root and Path(run_root).is_dir() else args

    @Slot()
    def _on_install_start(self) -> None:
        """Freeze the setup while the update is installed."""
        self.load_progress.setRange(0, 0)  # no progress reported
        self.load_progress.setVisible(True)
        self.summary_label.setText("Installing update ...")
        self._set_install_lock(True)

    @Slot(str)
    def _on_install_finish(self, status: str) -> None:
        """Release the setup after the update finished or failed."""
        self.load_progress.setVisible(False)
        self._set_install_lock(False)
        if self.run is not None:
            self._show_summary()
        else:
            self.summary_label.setText(status)

    def _set_install_lock(self, locked: bool) -> None:
        """Enable or disable everything that reads the installation."""
        self.data_form.setEnabled(not locked)
        self.sessions_box.setEnabled(not locked)
        self.selection_box.setEnabled(not locked)
        self.sampling_box.setEnabled(not locked)
        if locked:
            self.start_btn.setEnabled(False)
        else:
            self._update_start_state()

    def shutdown(self) -> None:
        """Stop the loader thread and the update button (when the window closes)."""
        self._cleanup_thread()
        if self.update_btn is not None:
            self.update_btn.shutdown()


class TrackInfoPanel(QWidget):
    """Header above the crops: the current track's metadata, the model's prediction and its candidate taxa."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the track info panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._shown: tuple[ValidationRun, TrackKey, dict[str, Any], Lineage] | None = None  # for refresh_theme()
        self._create_ui()

    def _create_ui(self) -> None:
        """Build the title, the metadata line, the blind-mode note and the model prediction box."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.title_label = QLabel("No track")
        font = self.title_label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        pred_inner = QWidget()
        pred_layout = QVBoxLayout(pred_inner)
        pred_layout.setContentsMargins(6, 2, 6, 6)
        # Only the predicted name is selectable, so it can be copied (e.g. for a web search)
        self.pred_name_label = _TaxonNameLabel("-")
        name_font = self.pred_name_label.font()
        self._base_pt = name_font.pointSize() if name_font.pointSize() > 0 else 9
        name_font.setPointSize(self._base_pt + 3)
        name_font.setBold(True)
        self.pred_name_label.setFont(name_font)
        self.pred_details_label = QLabel("")
        self.pred_details_label.setWordWrap(True)
        self.pred_details_label.setTextFormat(Qt.TextFormat.RichText)
        details_font = self.pred_details_label.font()
        details_font.setPointSize(self._base_pt + 1)
        self.pred_details_label.setFont(details_font)
        pred_layout.addWidget(self.pred_name_label, 0, Qt.AlignmentFlag.AlignLeft)
        pred_layout.addWidget(self.pred_details_label)
        self.pred_lineage_label = QLabel("")
        self.pred_lineage_label.setWordWrap(True)
        pred_layout.addWidget(self.pred_lineage_label)
        self.candidates_table = QTableWidget(0, 4)
        self.candidates_table.setHorizontalHeaderLabels(["Candidate", "Weighted probability", "Mean probability",
                                                         "Crops (1st/2nd)"])
        for col, tip in enumerate((
            "Taxon that was among the model's two best guesses in at least one crop of this track",
            ("Mean probability × crops (1st/2nd) / all crops of the track: what the candidates are ranked by, "
             "so a taxon seen in few crops of the track ends up low"),
            "Mean of the model's probability for this taxon over those crops only, not over all crops of the track",
            "Crops of the track in which this taxon was the model's first or second choice",
        )):
            _header_item(self.candidates_table, col).setToolTip(tip)
        self.candidates_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            self.candidates_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.candidates_table.verticalHeader().setVisible(False)
        self.candidates_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidates_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        # Show only two rows, the rest needs scrolling
        self.candidates_table.horizontalScrollBar().rangeChanged.connect(lambda *_: self._cap_candidates())
        self._cap_candidates()
        self.candidates_table.setToolTip("Alternative predictions for this track, ranked by weighted probability"
                                         "; scroll for further candidates")
        pred_layout.addWidget(self.candidates_table)
        self.pred_box = create_groupbox(
            "Model prediction", COLOR_GROUP_CLASSIFICATION, pred_inner)  # type: ignore[arg-type]
        self.blind_label = QLabel("Blind determination: the model prediction is shown after your determination.")
        self.blind_label.setWordWrap(True)
        self.blind_label.setVisible(False)
        layout.addWidget(self.blind_label)
        layout.addWidget(self.pred_box)

    def set_prediction_visible(self, visible: bool) -> None:
        """Show the model prediction, or the blind-mode note instead."""
        self.pred_box.setVisible(visible)
        self.blind_label.setVisible(not visible)

    def show_track(self, run: ValidationRun, key: TrackKey, row: dict[str, Any], model: Lineage) -> None:
        """Show the metadata, prediction and candidates of a track."""
        self._shown = (run, key, row, model)
        device_id, session_id, track_id = key
        device = _device_name(device_id, run.device_numbers)
        self.title_label.setText(f"Track ID {track_id}   ·   {device}   ·   session {session_id}")
        parts = [
            f"{_fmt_time(row.get('start_time'))} - {_fmt_time(row.get('end_time'))[11:] or '-'}",
            f"{_fmt(row.get('duration_s'), 1)} s",
            f"{row.get('track_imgs', '-')} crops",
            f"mean detection confidence {_fmt(row.get('det_conf_mean'))}",
        ]
        if row.get("bbox_length_mm_mean") is not None:
            parts.append(f"mean size ≈ {_fmt(row.get('bbox_length_mm_mean'), 1)} × "
                         f"{_fmt(row.get('bbox_width_mm_mean'), 1)} mm")
        if row.get(SETTING_COLUMN):
            parts.append(f"camera setting: {row[SETTING_COLUMN]}")
        self.meta_label.setText("   ·   ".join(parts))
        self.meta_label.setToolTip(
            "Sizes are estimated from the detection box and the frame size in mm of the run: the longer "
            "× the shorter side, averaged over the crops of the track" if row.get("bbox_length_mm_mean")
            is not None else ""
        )

        lineage = " › ".join(str(model[r]) for r in VALIDATION_RANKS if model.get(r))
        candidates = run.candidates_for(key)
        self.pred_name_label.setText(row.get("pred") or "-")
        self.pred_details_label.setText(
            f"weighted probability <b>{_fmt(row.get('pred_prob_weighted'))}</b>"
            f" · mean probability {_fmt(row.get('pred_prob_mean'))}"
            f" · {self._crops_text(candidates, row)}"
        )
        self.pred_lineage_label.setText(lineage)
        self.pred_lineage_label.setStyleSheet(f"color: {_theme_color('subtext0', '#9E9E9E')}")
        self.candidates_table.setRowCount(candidates.height)
        for i, cand in enumerate(candidates.to_dicts()):
            values = [str(cand.get("candidate", "-")), _fmt(cand.get("candidate_prob_weighted"), 3),
                      _fmt(cand.get("candidate_prob_mean"), 3), str(cand.get("candidate_imgs", "-"))]
            for j, text in enumerate(values):
                item = QTableWidgetItem(text)
                if j > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.candidates_table.setItem(i, j, item)

    def refresh_theme(self) -> None:
        """Re-render the shown track with the theme colors."""
        if self._shown is not None:
            self.show_track(*self._shown)

    @staticmethod
    def _crops_text(candidates: pl.DataFrame, row: dict[str, Any]) -> str:
        """Return the crops with the prediction as 1st choice ('top1_imgs'), else 'pred_imgs'."""
        track_imgs = row.get("track_imgs", "-")
        if not candidates.is_empty() and "top1_imgs" in candidates.columns:
            match = candidates.filter(pl.col("candidate") == row.get("pred"))
            best = match if not match.is_empty() else candidates.head(1)
            return f"1st choice in {best['top1_imgs'][0]} / {track_imgs} crops"
        return f"{row.get('pred_imgs', '-')} / {track_imgs} crops (1st/2nd)"

    def _cap_candidates(self) -> None:
        """Cap the candidates table at two rows, plus the scrollbar if shown."""
        table = self.candidates_table
        scrollbar = table.horizontalScrollBar()
        scrolls = scrollbar.maximum() > scrollbar.minimum()
        table.setMaximumHeight(
            table.horizontalHeader().sizeHint().height()
            + 2 * table.verticalHeader().defaultSectionSize()
            + (scrollbar.sizeHint().height() if scrolls else 0)
            + 2 * table.frameWidth()
        )


class _TaxonNameLabel(QLabel):
    """Selectable taxon name: double-click selects the whole name, not one word."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setToolTip("Double-click to select the name, then copy it with Ctrl+C")

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.setSelection(0, len(self.text()))
        event.accept()


class _LinkCombo(MultiComboParameter):
    """Multi-select combo of the earlier tracks a track continues, with a custom summary text."""

    def _init_ui(self) -> None:
        """Build the combo with an empty summary."""
        super()._init_ui()
        self._summary: Callable[[tuple[Any, ...]], str] = lambda _values: ""

    def set_summary(self, summary: Callable[[tuple[Any, ...]], str]) -> None:
        """Set the function that turns the checked track IDs into the combo text."""
        self._summary = summary
        self._refresh_text()

    def _refresh_text(self) -> None:
        """Show the summary of the ticked track IDs as the combo text."""
        self.combo.set_display_text(self._summary(self.value()))


class _SingleRow(QScrollArea):
    """Fixed-height row that scrolls horizontally instead of wrapping.

    Holds chips ('add_widget' / 'clear') or one rich-text label. The height includes the scrollbar.
    """

    # Padding, so chips do not touch the row edges
    _PAD = 3

    def __init__(self, content_height: int, parent: QWidget | None = None) -> None:
        """Initialize the row.

        Args:
            content_height: Height (px) of the row content.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.viewport().setAutoFillBackground(False)
        inner = QWidget()
        inner.setAutoFillBackground(False)
        self._layout = QHBoxLayout(inner)
        self._layout.setContentsMargins(0, self._PAD, 0, self._PAD)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self.setWidget(inner)
        extent = self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        self.setFixedHeight(content_height + 2 * self._PAD + extent + 2)

    def add_widget(self, widget: QWidget) -> None:
        """Append a widget before the trailing stretch."""
        self._layout.insertWidget(self._layout.count() - 1, widget)

    def clear(self) -> None:
        """Remove every widget and scroll back to the start."""
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.horizontalScrollBar().setValue(0)

    def count(self) -> int:
        """Return the number of widgets in the row."""
        return self._layout.count() - 1

    def widget_at(self, index: int) -> QWidget | None:
        """Return the widget at 'index', or None."""
        item = self._layout.itemAt(index)
        return item.widget() if item is not None else None


class VerdictPanel(QWidget):
    """Form for the validator's verdict on the current track: one-click actions, flags, taxon per rank, notes.

    Signals:
        verdict_changed(object): The edited 'Verdict' after a user action.
        reset_requested(): The 'Reset' button was pressed.
        action_done(): A one-click verdict was applied.
        custom_taxon_added(str, object): '(rank, lineage)' typed as free text.
        batch_action(str, object): A one-click action for the selected tracks.
        apply_to_selection_requested(): 'Copy determination to selection' was pressed.
        notes_changed(str): The notes text after a keystroke.
        jump_requested(str): Track ID of the same device and session to show.
    """
    verdict_changed = Signal(object)
    notes_changed = Signal(str)
    reset_requested = Signal()
    action_done = Signal()
    custom_taxon_added = Signal(str, object)
    batch_action = Signal(str, object)
    apply_to_selection_requested = Signal()
    jump_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the verdict panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._taxonomy = TaxonomyIndex()
        self._model: Lineage = dict.fromkeys(VALIDATION_RANKS)
        self._model_rank = "species"
        self._verdict = Verdict()
        self._loading = False
        self._blind = False
        self._full_models: dict[str, QStringListModel] = {}
        self._link_candidates: list[dict[str, Any]] = []  # earlier tracks offered for 'same individual'
        self._link_group: list[dict[str, Any]] = []  # all tracks of the individual (incl. this one)
        self._syncing_links = False  # filling the link combo, not an edit
        self._link_order_hint = False  # list not in chronological order
        self._recent: list[Lineage] = []
        self._batch_counts = (0, 0)
        self._batch_link: tuple[bool, bool, str] | None = None  # link button for a selection, see set_batch_link()
        self._create_ui()

    def _create_ui(self) -> None:
        """Build the panel top to bottom: quick verdicts, determination, flags, status, notes and batch banner."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        small_font = self.font()
        small_font.setPointSize(max(self.font().pointSize() - 1, 7))

        def toggle_button(text: str, tooltip: str, color: str, small: bool = True) -> QPushButton:
            """Return a checkable button filled with 'color' while checked (small: a compact secondary control)."""
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setStyleSheet((_FLAG_BTN_STYLE if small else _ACTION_BTN_STYLE)
                              + f"QPushButton:checked {{ background-color: {color}; color: #F5F5F5; }}")
            if small:
                btn.setFont(small_font)
                btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            return btn

        # Quick verdicts: Agree, agree down to a rank, not an arthropod, not determinable
        actions = QGridLayout()
        actions.setSpacing(6)
        self.agree_btn = StyledButton("Agree  (A)", COLOR_VERDICT["agree"], font_size=10)
        self.agree_btn.setToolTip(_AGREE_TOOLTIP)
        self.agree_btn.clicked.connect(self._on_agree)
        actions.addWidget(self.agree_btn, 0, 0, 1, 2)
        chips = QHBoxLayout()
        chips.setSpacing(4)
        self.chip_buttons: dict[str, StyledButton] = {}
        for i, rank in enumerate(VALIDATION_RANKS, start=1):
            chip = StyledButton(f"{i} {_RANK_LABELS[rank]}", COLOR_VERDICT["partial"], font_size=8)
            chip.clicked.connect(lambda _=False, r=rank: self.agree_down_to(r))
            chip.setMinimumWidth(40)
            chip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            chips.addWidget(chip)
            self.chip_buttons[rank] = chip
        actions.addLayout(chips, 1, 0, 1, 2)
        self.fp_btn = toggle_button("Not an arthropod  (F)",
                                    "False positive detection: background, plant, shadow, etc.",
                                    COLOR_VERDICT["false_positive"], small=False)
        self.fp_btn.toggled.connect(self._on_fp_toggled)
        self.all_indet_btn = toggle_button("Not determinable  (U)",
                                           "An arthropod, but no rank can be determined (all ranks indet.)",
                                           COLOR_VERDICT["indeterminate"], small=False)
        self.all_indet_btn.clicked.connect(self._on_all_indet_clicked)
        # Buttons may shrink below their text width (elided)
        for btn in (self.agree_btn, self.fp_btn, self.all_indet_btn):
            btn.setMinimumWidth(60)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        actions.addWidget(self.fp_btn, 2, 0)
        actions.addWidget(self.all_indet_btn, 2, 1)
        layout.addLayout(actions)
        layout.addSpacing(_SECTION_GAP)

        # Recent identifications
        self.recent_label = QLabel("Recent:")
        layout.addWidget(self.recent_label)
        probe = QToolButton()  # reserves the row height when no chips are shown
        probe.setText("ID 0")
        probe.setStyleSheet(_CHIP_STYLE)
        self.recent_row = _SingleRow(probe.sizeHint().height())
        layout.addWidget(self.recent_row)
        layout.addSpacing(_SECTION_GAP)

        # Determination: one taxon combo per rank, with its indet. button and the model's name
        combos = QFormLayout()
        combos.setContentsMargins(0, 4, 0, 0)
        combos.setSpacing(4)
        combos.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.combos: dict[str, TaxonCombo] = {}
        self.indet_buttons: dict[str, QToolButton] = {}
        self.model_labels: dict[str, QLabel] = {}
        for rank in VALIDATION_RANKS:
            combo = TaxonCombo(rank)
            combo.committed.connect(lambda text, r=rank: self._on_taxon_committed(r, text))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(combo, 1)
            indet_btn = QToolButton()
            indet_btn.setText("?")
            indet_btn.setCheckable(True)
            indet_btn.setFixedWidth(26)
            indet_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            indet_btn.setStyleSheet(
                f"QToolButton {{ font-weight: bold; }}"
                f"QToolButton:checked {{ background-color: {COLOR_VERDICT['indeterminate']}; color: #F5F5F5; }}"
            )
            indet_btn.clicked.connect(lambda checked=False, r=rank: self._on_indet_clicked(r, checked))
            row_layout.addWidget(indet_btn, 0)
            model_label = QLabel("")
            model_label.setMinimumWidth(16)
            model_label.setToolTip("Model prediction at this rank")
            row_layout.addWidget(model_label, 0)
            combos.addRow(f"{_RANK_LABELS[rank]}:", row_widget)
            self.combos[rank] = combo
            self.indet_buttons[rank] = indet_btn
            self.model_labels[rank] = model_label
        layout.addLayout(combos)
        # Sex and qualifiers of the determination in one row
        self.sex_buttons: dict[str, QPushButton] = {}
        for sex, label in _SEX_LABELS.items():
            btn = toggle_button(label, f"Sex of the individual: {sex}, if clearly visible (optional)",
                                COLOR_VERDICT["partial"])
            btn.clicked.connect(lambda checked=False, s=sex: self._on_sex_clicked(s, checked))
            self.sex_buttons[sex] = btn
        self.inc_btn = toggle_button(
            "inc.  uncertain", "Uncertain determination (inc.): the characteristic features are visible, but you are "
            "not certain of the taxon - written after its rank, e.g. Syrphidae fam. inc.  (I)",
            COLOR_VERDICT["partial"])
        self.inc_btn.toggled.connect(self._on_inc_toggled)
        self.agg_btn = toggle_button(
            "agg.  aggregate", "Aggregate determination (agg.): a group of closely related species that cannot be "
            "separated on images, named after the entered species, e.g. Bombus lucorum agg.  (G)",
            COLOR_VERDICT["partial"])
        self.agg_btn.toggled.connect(self._on_agg_toggled)
        qualifiers = QHBoxLayout()
        qualifiers.setContentsMargins(0, 4, 0, 0)
        qualifiers.setSpacing(6)
        for btn in (*self.sex_buttons.values(), self.inc_btn, self.agg_btn):
            qualifiers.addWidget(btn)
        qualifiers.addStretch(1)
        layout.addLayout(qualifiers)
        layout.addSpacing(_SECTION_GAP)


        # Flags: each button labels the dropdown next to it
        flags = QGridLayout()
        flags.setHorizontalSpacing(6)
        flags.setVerticalSpacing(4)
        flags.setColumnStretch(1, 1)
        self.link_btn = toggle_button("Same individual  (L)", _LINK_BTN_TOOLTIP, COLOR_FLAG_SAME_INDIVIDUAL)
        self.link_btn.toggled.connect(self._on_link_toggled)
        self._apply_link_style()  # adds the tint of a first track
        self.link_combo = _LinkCombo()
        self.link_combo.setToolTip(f"Earlier tracks of this device/session that ended at most "
                                   f"{VALIDATION_SAME_INDIVIDUAL_WINDOW_S:g} s before this one started (or overlap it "
                                   "slightly); tick every one that shows the same individual")
        self.link_combo.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.link_combo.combo.setMinimumContentsLength(10)
        self.link_combo.value_changed.connect(self._on_link_values)
        flags.addWidget(self.link_btn, 0, 0)
        flags.addWidget(self.link_combo, 0, 1)
        self.individual_row = _SingleRow(self.fontMetrics().lineSpacing())
        self.individual_label = QLabel("")
        self.individual_label.setTextFormat(Qt.TextFormat.RichText)
        self.individual_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.individual_label.linkActivated.connect(lambda href: self.jump_requested.emit(str(href)))
        self.individual_row.add_widget(self.individual_label)
        flags.addWidget(QLabel("Individual:"), 1, 0)
        flags.addWidget(self.individual_row, 1, 1)
        self.multiple_btn = toggle_button("Multiple individuals  (M)",
                                          "More than one individual in the crop or track; choose the type next to it",
                                          COLOR_FLAG_MULTIPLE_INDIVIDUALS)
        self.multiple_btn.toggled.connect(self._on_multiple_toggled)
        self.multiple_type = QComboBox()
        for kind in MULTIPLE_INDIVIDUALS_TYPES:
            self.multiple_type.addItem(MULTIPLE_INDIVIDUALS_LABELS.get(kind, kind), kind)
        self.multiple_type.currentIndexChanged.connect(self._on_multiple_type_changed)
        self.multiple_type.setEnabled(False)
        flags.addWidget(self.multiple_btn, 2, 0)
        flags.addWidget(self.multiple_type, 2, 1)
        self.quality_btn = toggle_button(
            "Low image quality  (Q)", "Blurry, too small, partially visible or occluded (choose the reason next to "
            "it); left out of the accuracy statistics by default",
            COLOR_FLAG_LOW_QUALITY)
        self.quality_btn.toggled.connect(self._on_quality_toggled)
        self.quality_reason = QComboBox()
        for reason in IMAGE_QUALITY_REASONS:
            self.quality_reason.addItem(_IMAGE_QUALITY_REASON_LABELS.get(reason, reason), reason)
        self.quality_reason.currentIndexChanged.connect(self._on_quality_reason_changed)
        self.quality_reason.setEnabled(False)
        flags.addWidget(self.quality_btn, 3, 0)
        flags.addWidget(self.quality_reason, 3, 1)
        for combo in (self.quality_reason, self.multiple_type):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.setMinimumContentsLength(8)
        layout.addLayout(flags)
        layout.addSpacing(_SECTION_GAP)

        # Notes: one line with a length limit
        layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional short note, e.g. on unclear taxonomy ...")
        self.notes_edit.setMaxLength(VALIDATION_NOTES_MAX_CHARS)
        self.notes_edit.textChanged.connect(self._on_notes_edited)
        layout.addWidget(self.notes_edit)

        # Status label that shows the current verdict
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        self.status_label.setFixedHeight(self.fontMetrics().lineSpacing() * 2 + 2)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.status_label)

        # Reset button
        self.reset_btn = QPushButton("Reset  (R)")
        self.reset_btn.setToolTip("Discard everything recorded for this track (with a selection: for all selected "
                                  "tracks) and start again")
        self.reset_btn.setStyleSheet(_FLAG_BTN_STYLE)
        self.reset_btn.setFont(small_font)
        self.reset_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(self.reset_btn)
        layout.addLayout(reset_row)

        # Batch banner (enabled when several tracks are selected)
        self.batch_frame = QFrame()
        self.batch_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.batch_frame.setStyleSheet(f"QFrame {{ border: 1px solid {COLOR_BTN_VALIDATE}; border-radius: 4px; }}")
        batch_layout = QVBoxLayout(self.batch_frame)
        batch_layout.setContentsMargins(6, 4, 6, 4)
        self.batch_label = QLabel("")
        self.batch_label.setWordWrap(True)
        self.batch_label.setMinimumHeight(self.fontMetrics().lineSpacing() * 2 + 2)
        batch_layout.addWidget(self.batch_label)
        self.apply_selection_btn = StyledButton("Copy determination to selection", COLOR_BTN_VALIDATE)
        self.apply_selection_btn.setToolTip("Writes taxon, qualifier and sex of the shown track to the other "
                                            "selected tracks; the shown track is confirmed too if it is still to do")
        self.apply_selection_btn.clicked.connect(self.apply_to_selection_requested.emit)
        batch_layout.addWidget(self.apply_selection_btn)
        layout.addWidget(self.batch_frame)
        self.set_batch_count(0)

    def set_taxonomy(self, taxonomy: TaxonomyIndex) -> None:
        """Use a taxonomy index for the combos (the type-ahead models are rebuilt)."""
        self._taxonomy = taxonomy
        self._full_models = {}

    def set_blind(self, blind: bool) -> None:
        """Set blind mode: prefill indet. and hide the model until a determination is entered."""
        self._blind = blind

    @property
    def blind_hidden(self) -> bool:
        """Whether the model prediction is hidden (blind, not yet determined)."""
        return self._blind and not self._verdict.validated

    def set_batch_count(self, count: int, copy_targets: int = 0) -> None:
        """Update the batch banner.

        Args:
            count: Selected tracks.
            copy_targets: Selected tracks other than the shown one (0 disables the copy button).
        """
        batch = count > 1
        self._batch_counts = (count, copy_targets)  # for refresh_theme()
        self.apply_selection_btn.setEnabled(copy_targets > 0)
        self.apply_selection_btn.setText(
            f"Copy determination to {copy_targets} track{'s' if copy_targets > 1 else ''}" if copy_targets
            else "Copy determination to selection"
        )
        self.batch_frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLOR_BTN_VALIDATE if batch else 'palette(mid)'}; border-radius: 4px; }}"
        )
        if batch:
            self.batch_label.setText(
                f"<b>{count} tracks selected.</b> Verdicts and flags apply to all of them. Use the button "
                "below to also copy your determination:"
            )
            self.batch_label.setToolTip(("" if self._blind else "Agree confirms each track's own prediction; ")
                                        + ("Entries" if self._blind else "entries")
                                        + " in 'Same individual as' stay with the shown track")
        else:
            self.batch_label.setText(f"<span style='color:{_theme_color('subtext0', '#9E9E9E')}'>Single track - "
                                     "Ctrl/Shift-click in the list to select several.</span>")
            self.batch_label.setToolTip("")

    def load(self, model: Lineage, verdict: Verdict | None, model_rank: str, recent: list[Lineage]) -> None:
        """Show a track, prefilled from the stored verdict or the model."""
        self._loading = True
        self._model = dict(model)
        self._model_rank = model_rank
        pending = verdict if verdict is not None and not verdict.validated else None
        if verdict is not None and verdict.validated:
            self._verdict = verdict.copy()
        elif self._blind:
            self._verdict = Verdict(val=dict.fromkeys(VALIDATION_RANKS, INDET_LABEL), val_source="taxonomy_list")
        else:
            self._verdict = Verdict(val=self._model_prefill(), val_source="model")
        if pending is not None:
            # Keep pending notes, flags and attributes
            self._verdict.notes = pending.notes
            self._verdict.image_quality, self._verdict.image_quality_reason = (
                pending.image_quality, pending.image_quality_reason)
            self._verdict.multiple_individuals = pending.multiple_individuals
            self._verdict.same_individual_as = pending.same_individual_as
            self._verdict.qualifier, self._verdict.sex = pending.qualifier, pending.sex
        for rank in VALIDATION_RANKS:
            chip = self.chip_buttons[rank]
            if chip.isVisibleTo(self) != (rank_index(rank) < rank_index(model_rank)):
                chip.setVisible(rank_index(rank) < rank_index(model_rank))
            chip.setEnabled(rank_index(rank) <= rank_index(model_rank) and bool(model.get(rank)))
            self.model_labels[rank].setText("")
        self._recent = list(recent)
        self._sync_widgets()
        self._loading = False

    def current_verdict(self) -> Verdict:
        """Return a copy of the verdict shown in the form."""
        return self._verdict.copy()

    @staticmethod
    def prefill_for(model: Lineage, model_rank: str) -> dict[str, str | None]:
        """Return the model lineage down to 'model_rank', indet. below."""
        val: dict[str, str | None] = {}
        for rank in VALIDATION_RANKS:
            if rank_index(rank) <= rank_index(model_rank) and model.get(rank):
                val[rank] = model[rank]
            else:
                val[rank] = INDET_LABEL
        return val

    def apply_lineage(self, lineage: Lineage) -> None:
        """Apply a full/partial lineage (recent identification chip)."""
        new_val: Lineage = {r: (lineage.get(r) or INDET_LABEL) for r in VALIDATION_RANKS}
        self._store_val(new_val)

    def focus_rank(self, rank: str) -> None:
        """Focus the combo of a rank with its text selected."""
        self.combos[rank].setFocus()
        if (edit := self.combos[rank].lineEdit()) is not None:
            edit.selectAll()

    def confirm_current(self) -> Verdict:
        """Validate the shown track with the form's determination and return it (batch copy)."""
        self._commit()
        return self._verdict.copy()

    def batch_verdict(self, action: str, payload: Any, model: Lineage, existing: Verdict | None) -> Verdict | None:
        """Return the verdict a one-click action produces for another selected track, None to leave it unchanged.

        A flag on a track without a determination stays pending. Withdrawing 'Not an arthropod' or 'Not determinable'
        only affects tracks with that verdict.
        """
        fresh = existing is None or not existing.validated
        if action in ("false_positive", "indet") and not payload["checked"]:
            if existing is None or fresh:
                return None
            matches = (existing.false_positive if action == "false_positive"
                       else not existing.false_positive and not existing.determinate_ranks())
            if not matches:
                return None  # another verdict: nothing to withdraw
            withdrawn = existing.copy()
            withdrawn.false_positive, withdrawn.val, withdrawn.val_source = False, self._fresh_val(model), (
                "taxonomy_list" if self._blind else "model")
            withdrawn.validated, withdrawn.validated_at = False, None
            return withdrawn
        if existing is not None and not fresh:
            base = existing.copy()
        else:
            base = Verdict(val=self._fresh_val(model))
            if existing is not None:
                # Keep pending notes, flags and attributes
                base.notes = existing.notes
                base.image_quality, base.image_quality_reason = existing.image_quality, existing.image_quality_reason
                base.multiple_individuals = existing.multiple_individuals
                base.same_individual_as = existing.same_individual_as
                base.qualifier, base.sex = existing.qualifier, existing.sex
        base.validated = True
        if action == "agree_down_to":
            base.val = {r: ((model.get(r) or INDET_LABEL) if rank_index(r) <= rank_index(payload) else INDET_LABEL)
                        for r in VALIDATION_RANKS}
            base.false_positive, base.val_source = False, "model"
        elif action == "indet":
            base.val = dict.fromkeys(VALIDATION_RANKS, INDET_LABEL)
            base.false_positive, base.val_source = False, "taxonomy_list"
        elif action == "false_positive":
            base.false_positive = True
        elif action in ("low_quality", "multiple_individuals"):
            if action == "low_quality":
                base.image_quality = "low" if payload["checked"] else "ok"
                base.image_quality_reason = payload["reason"] if payload["checked"] else None
            else:
                base.multiple_individuals = payload["kind"] if payload["checked"] else None
            if fresh:
                base.validated = False  # a flag alone stays pending
        elif action == "apply_determination":
            # Only the determination is copied
            current: Verdict = payload
            base.val = dict(current.val)
            base.false_positive = current.false_positive
            base.qualifier, base.sex = current.qualifier, current.sex
            base.val_source = "free_text" if current.val_source == "free_text" else "taxonomy_list"
        # Qualifier rule of the form (see _commit())
        determined = base.determinate_ranks()
        base.qualifier = allowed_qualifier(base.qualifier, determined[-1] if determined else None)
        return base

    def set_batch_link(self, state: tuple[bool, bool, str] | None) -> None:
        """Let the 'Same individual' button act on the selection.

        Args:
            state: '(enabled, checked, tooltip)' of the button for the selected tracks, None for the shown track only.
        """
        self._batch_link = state
        self._render_links()

    def set_link_context(self, candidates: list[dict[str, Any]], group: list[dict[str, Any]],
                         order_hint: bool = False) -> None:
        """Set the 'same individual' context of the shown track (before load() and after changes).

        Args:
            candidates: Earlier tracks to link (ValidationRun.previous_tracks() rows plus 'human').
            group: Tracks of the individual in time order, dicts with 'track_id', 'is_current',
                'in_queue', 'human', 'start_time', 'end_time', 'warning' (false positive), 'note'.
            order_hint: Show the hint that the list is not chronological.
        """
        self._link_candidates = list(candidates)
        self._link_group = list(group)
        self._link_order_hint = order_hint
        if not self._loading:
            self._render_links()

    def refresh_theme(self) -> None:
        """Recompute the mixed colors and re-render colored rich texts."""
        self._apply_link_style()
        self._render_links()
        self._render_status()
        self._render_recent(self._recent)
        self.set_batch_count(*self._batch_counts)

    def agree_down_to(self, rank: str) -> None:
        """Agree with the model lineage down to 'rank' (deeper ranks indet.), keeping qualifiers and sex."""
        if not self.chip_buttons[rank].isEnabled():
            return
        new_val = dict.fromkeys(VALIDATION_RANKS)
        for r in ranks_up_to(rank):
            new_val[r] = self._model.get(r) or INDET_LABEL
        for r in ranks_below(rank):
            new_val[r] = INDET_LABEL
        self._verdict.val = new_val
        self._verdict.false_positive = False
        self._verdict.val_source = "model"
        self._commit(one_click=True)
        self.batch_action.emit("agree_down_to", rank)

    def toggle_false_positive(self) -> None:
        """Toggle 'Not an arthropod' (shortcut)."""
        self.fp_btn.toggle()

    def toggle_indeterminate(self) -> None:
        """Toggle 'Not determinable' (shortcut)."""
        self._on_all_indet_clicked(not self._is_all_indet())

    def toggle_uncertain(self) -> None:
        """Toggle 'inc.' if a rank is determined (shortcut)."""
        if self.inc_btn.isEnabled():
            self.inc_btn.toggle()

    def toggle_aggregate(self) -> None:
        """Toggle 'agg.' if a species is determined (shortcut)."""
        if self.agg_btn.isEnabled():
            self.agg_btn.toggle()

    def toggle_low_quality(self) -> None:
        """Toggle 'Low image quality' (shortcut)."""
        self.quality_btn.toggle()

    def toggle_link(self) -> None:
        """Toggle 'Same individual' if the track can be linked (shortcut)."""
        if self.link_btn.isEnabled():
            self.link_btn.toggle()

    def toggle_multiple(self) -> None:
        """Toggle 'Multiple individuals' unless the track is a false positive (shortcut)."""
        if self.multiple_btn.isEnabled():
            self.multiple_btn.toggle()

    def set_all_indet(self) -> None:
        """Mark every rank as indet. (nothing determinable)."""
        self._verdict.val = dict.fromkeys(VALIDATION_RANKS, INDET_LABEL)
        self._verdict.false_positive = False
        self._verdict.val_source = "taxonomy_list"
        self._commit(one_click=True)  # _commit() drops the qualifier
        self.batch_action.emit("indet", {"checked": True})

    def set_indet(self, rank: str) -> None:
        """Mark a rank and the ranks below as indet."""
        if self.combos[rank].isEnabled():
            self._on_taxon_committed(rank, INDET_LABEL)

    def _model_prefill(self) -> dict[str, str | None]:
        """Return the model lineage down to the model rank, indet. below."""
        return self.prefill_for(self._model, self._model_rank)

    def _full_model(self, rank: str) -> QStringListModel:
        """Return the type-ahead model over all names of a rank (built once)."""
        model = self._full_models.get(rank)
        if model is None:
            model = QStringListModel(self._taxonomy.names(rank), self)
            self._full_models[rank] = model
        return model

    def _fresh_val(self, model: Lineage) -> dict[str, str | None]:
        """Return the initial determination: the model prefill, or indet. when blind."""
        if self._blind:
            return dict.fromkeys(VALIDATION_RANKS, INDET_LABEL)
        return self.prefill_for(model, self._model_rank)

    def _store_val(self, val: dict[str, str | None]) -> None:
        """Record an edited determination (clears false positive, recomputes the source)."""
        self._verdict.val = val
        self._verdict.false_positive = False
        self._verdict.val_source = self._compute_source()
        self._commit()

    def _compute_source(self) -> str:
        """Return 'model' if the lineage equals the model prefill, else 'free_text' / 'taxonomy_list'."""
        if all((self._verdict.val.get(r) or INDET_LABEL) == (self._model_prefill().get(r) or INDET_LABEL)
               for r in VALIDATION_RANKS):
            return "model"
        for rank in VALIDATION_RANKS:
            name = self._verdict.val.get(rank)
            if name and name != INDET_LABEL and self._taxonomy.source_of(rank, name) == "free_text":
                return "free_text"
        return "taxonomy_list"

    def _commit(self, one_click: bool = False, validate: bool = True) -> None:
        """Apply the qualifier rule, update the widgets and emit the edited verdict.

        Args:
            one_click: Also emit action_done (auto-advance after a one-click verdict).
            validate: Mark the track validated, False keeps a pending entry pending.
        """
        if self._loading:
            return
        determined = self._verdict.determinate_ranks()
        self._verdict.qualifier = allowed_qualifier(self._verdict.qualifier, determined[-1] if determined else None)
        if validate:
            self._verdict.validated = True
        self._sync_widgets()
        self.verdict_changed.emit(self._verdict.copy())
        if one_click:
            self.action_done.emit()

    def _is_all_indet(self) -> bool:
        """Return whether the verdict is 'not determinable'."""
        return self._verdict.validated and not self._verdict.false_positive and not self._verdict.determinate_ranks()

    def _has_pending_content(self) -> bool:
        """Return whether pending content was recorded (see ValidationSession.set_pending())."""
        verdict = self._verdict
        return bool(verdict.notes.strip() or verdict.image_quality == "low" or verdict.multiple_individuals
                    or verdict.same_individual_as or verdict.qualifier or verdict.sex)

    def _withdraw_verdict(self) -> None:
        """Withdraw the verdict of the shown track (to do again). Pending notes, flags and attributes are kept."""
        self._verdict.false_positive = False
        self._verdict.val = self._fresh_val(self._model)
        self._verdict.val_source = "taxonomy_list" if self._blind else "model"
        self._verdict.validated = False
        self._verdict.validated_at = None
        self._commit(validate=False)

    def _restorable_model_name(self, rank: str) -> str | None:
        """Return the model name to restore at an indet. rank if the ranks above still match."""
        model_name = self._model.get(rank)
        if not model_name or self._blind or rank_index(rank) > rank_index(self._model_rank):
            return None
        if any(self._verdict.val.get(r) != self._model.get(r) for r in ranks_up_to(rank)[:-1]):
            return None
        return model_name

    @staticmethod
    def _display_name(rank: str, name: str, parent_name: str | None) -> str:
        """Return a species as its epithet when the genus is known, other ranks unchanged."""
        if rank == "species" and parent_name and name.startswith(parent_name + " "):
            return name[len(parent_name) + 1:]
        return name

    def _full_name(self, rank: str, text: str) -> str:
        """Prefix a typed epithet with the current genus."""
        if rank != "species" or not text or text == INDET_LABEL or " " in text:
            return text
        genus = self._verdict.val.get("genus")
        if genus and genus != INDET_LABEL:
            return f"{genus} {text}"
        return text

    def _sync_widgets(self) -> None:
        """Reflect the verdict in all widgets without triggering handlers."""
        was_loading = self._loading
        self._loading = True
        verdict = self._verdict
        self.fp_btn.setChecked(verdict.false_positive)
        low = verdict.image_quality == "low"
        self.quality_btn.setChecked(low)
        self.quality_reason.setEnabled(low)
        idx = self.quality_reason.findData(verdict.image_quality_reason or IMAGE_QUALITY_REASONS[0])
        self.quality_reason.setCurrentIndex(max(idx, 0))
        multiple = verdict.multiple_individuals is not None and not verdict.false_positive
        self.multiple_btn.setEnabled(not verdict.false_positive)
        self.multiple_btn.setChecked(multiple)
        self.multiple_type.setEnabled(multiple)
        idx = self.multiple_type.findData(verdict.multiple_individuals or MULTIPLE_INDIVIDUALS_TYPES[0])
        self.multiple_type.setCurrentIndex(max(idx, 0))
        self._render_links()
        self._render_recent(self._recent)

        taxa_enabled = not verdict.false_positive
        has_taxon = bool(verdict.determinate_ranks())
        hidden = self.blind_hidden
        self.all_indet_btn.setChecked(self._is_all_indet())
        self.inc_btn.setEnabled(taxa_enabled and has_taxon)
        self.inc_btn.setChecked(has_qualifier(verdict.qualifier, ID_QUALIFIER_INC) and taxa_enabled and has_taxon)
        has_species = "species" in verdict.determinate_ranks()  # agg. stands after a species name
        self.agg_btn.setEnabled(taxa_enabled and has_species)
        self.agg_btn.setChecked(has_qualifier(verdict.qualifier, ID_QUALIFIER_AGG) and taxa_enabled and has_species)
        for sex, btn in self.sex_buttons.items():
            btn.setEnabled(taxa_enabled)
            btn.setChecked(taxa_enabled and verdict.sex == sex)
        self.agree_btn.setEnabled(taxa_enabled and not self._blind and bool(self._model.get(self._model_rank)))
        self.agree_btn.setToolTip(_BLIND_AGREE_TOOLTIP if self._blind else _AGREE_TOOLTIP)
        for rank in VALIDATION_RANKS:
            combo = self.combos[rank]
            combo.setEnabled(taxa_enabled)
            is_indet = verdict.val.get(rank) in (None, "", INDET_LABEL)
            restorable = self._restorable_model_name(rank) if is_indet else None
            indet_btn = self.indet_buttons[rank]
            indet_btn.setChecked(is_indet)
            indet_btn.setEnabled(taxa_enabled and (not is_indet or restorable is not None))
            indet_btn.setToolTip(f"Back to the model prediction ({restorable})" if restorable
                                 else f"Set {rank} (and the ranks below) to indet.")
            parent_rank = VALIDATION_RANKS[rank_index(rank) - 1] if rank_index(rank) > 0 else None
            parent_name = verdict.val.get(parent_rank) if parent_rank else None
            if parent_name in (None, "", INDET_LABEL):
                parent_name = None
            children = self._taxonomy.children(rank, parent_name) if (parent_name or parent_rank is None) else []
            if parent_rank is not None and parent_name is None:
                # Parent unknown: type-ahead over all names of this rank
                combo.set_choices([], self._full_model(rank))
            elif len(children) > _MAX_LISTED_CHILDREN:
                combo.set_choices([], self._full_model(rank))
            else:
                # Species as epithets once the genus is known
                combo.set_choices([self._display_name(rank, c, parent_name) for c in children], None)
            combo.set_text(self._display_name(rank, verdict.val.get(rank) or INDET_LABEL, parent_name))
            model_name = self._model.get(rank)
            human_name = verdict.val.get(rank)
            if verdict.false_positive or not model_name or hidden:
                mark, color = "", None
            elif human_name == model_name:
                mark, color = "✓", COLOR_VERDICT["agree"]
            elif human_name in (None, "", INDET_LABEL):
                mark, color = "?", COLOR_VERDICT["indeterminate"]
            else:
                mark, color = "✎", COLOR_VERDICT["corrected"]
            self.model_labels[rank].setText(mark)
            self.model_labels[rank].setStyleSheet(f"color: {color}; font-weight: bold;" if color else "")
            self.model_labels[rank].setToolTip("Model: hidden (blind)" if hidden else
                                               f"Model: {model_name}" if model_name else "Model: not predicted")
        for rank, btn in self.chip_buttons.items():
            btn.setEnabled(taxa_enabled and not self._blind and rank_index(rank) <= rank_index(self._model_rank)
                           and bool(self._model.get(rank)))
            btn.setToolTip(_BLIND_AGREE_TOOLTIP if self._blind else f"Model prediction correct down to {rank}; "
                           f"deeper ranks become indet. ({rank_index(rank) + 1})")
        self.notes_edit.blockSignals(True)
        if self.notes_edit.text() != verdict.notes:
            self.notes_edit.setText(verdict.notes)
        self.notes_edit.blockSignals(False)
        self.notes_edit.setToolTip(verdict.notes)
        self._render_status()
        self._loading = was_loading

    def _render_status(self) -> None:
        """Show what the buttons do not: verdict type, agreement, qualifier mark, link conflict, free text."""
        verdict_type, agree_rank, lowest = derive_verdict(self._model, self._verdict, self._model_rank)
        if not self._verdict.validated:
            self.status_label.setText(f"<span style='color:{_theme_color('subtext0', '#9E9E9E')}'>Not validated</span>")
            self.status_label.setToolTip("")
            return
        color = COLOR_VERDICT.get(verdict_type or "", "#9E9E9E")
        text = f"<b style='color:{color}'>{VERDICT_LABELS.get(verdict_type or '', verdict_type)}</b>"
        if agree_rank and verdict_type in ("partial", "corrected"):
            text += f" · agrees down to {agree_rank}"
        if lowest and verdict_type in ("agree", "partial", "corrected"):
            text += f" · determined to {lowest}"
            # Qualifier mark of the rank it applies to
            if mark := qualifier_mark(self._verdict.qualifier, lowest):
                text += f" (<b>{mark}</b>)"
        conflict = self._link_conflict() if self._link_allowed() else None
        if conflict:
            text += (f" · <b style='color:{COLOR_VERDICT['corrected']}'>⚠ ID {conflict[0]} determined as "
                     f"{conflict[3]} ({conflict[1]})</b>")
        if self._verdict.val_source == "free_text":
            text += " · <i>contains free-text taxon</i>"
        self.status_label.setText(text)
        self.status_label.setToolTip(QTextDocumentFragment.fromHtml(text).toPlainText())

    def _render_recent(self, recent: list[Lineage]) -> None:
        """Show chips of the individual's other determinations and of recent lineages."""
        self.recent_row.clear()
        entries: list[tuple[str, Lineage]] = []
        if self._link_allowed():
            # Lineages of the individual's other tracks
            seen_lineages: list[Lineage] = []
            for member in self._group_others():
                human = member.get("human")
                if human and human not in seen_lineages:
                    seen_lineages.append(human)
                    entries.append((f"← ID {member['track_id']}: ", human))
        entries.extend(("", lineage) for lineage in recent)
        if not entries:
            self.recent_row.add_widget(QLabel(f"<span style='color:{_theme_color('subtext0', '#9E9E9E')}'>-</span>"))
        for prefix, lineage in entries:
            deepest = next((lineage[r] for r in reversed(VALIDATION_RANKS)
                            if lineage.get(r) and lineage[r] != INDET_LABEL), None)
            if not deepest:
                continue
            btn = QToolButton()
            btn.setText(prefix + deepest)
            btn.setToolTip(" › ".join(str(lineage.get(r)) for r in VALIDATION_RANKS if lineage.get(r)))
            btn.setAutoRaise(True)
            btn.setStyleSheet(_CHIP_STYLE)
            btn.clicked.connect(lambda _=False, lin=dict(lineage): self.apply_lineage(lin))
            self.recent_row.add_widget(btn)

    def _link_allowed(self) -> bool:
        """Return whether the track can be linked (not a false positive)."""
        return not self._verdict.false_positive

    def _group_others(self) -> list[dict[str, Any]]:
        """Return the other tracks of the individual, else the linked candidates."""
        others = [m for m in self._link_group if not m.get("is_current")]
        if others:
            return others
        links = parse_links(self._verdict.same_individual_as)
        return [c for c in self._link_candidates if str(c["track_id"]) in links]

    def _link_conflict(self) -> tuple[str, str, str, str] | None:
        """Return '(track_id, rank, own name, other name)' of the first contradicting group member."""
        for member in self._group_others():
            other = member.get("human")
            if not other or member.get("note"):  # different taxa in track: may show another taxon
                continue
            for rank in VALIDATION_RANKS:
                a, b = self._verdict.val.get(rank), other.get(rank)
                if a and b and a != INDET_LABEL and b != INDET_LABEL and a != b:
                    return str(member["track_id"]), rank, a, b
        return None

    def _render_links(self) -> None:
        """Rebuild the 'same individual' dropdown, summary line and button."""
        links, link_ok = parse_links(self._verdict.same_individual_as), self._link_allowed()
        entries: list[tuple[str, str]] = []  # (label, track ID) of the linkable earlier tracks
        for cand in self._link_candidates:
            tid = str(cand["track_id"])
            gap = cand["gap_s"]
            label = f"ID {tid} · {abs(gap):.1f} s {'overlap' if gap < 0 else 'before'}"
            if not self._blind and cand.get("pred"):
                label += f" · {cand['pred']}"
            entries.append((label, tid))
        shown = [tid for _label, tid in entries]
        for tid in links:
            if tid not in shown:  # stored link outside the candidate window
                entries.append((f"ID {tid} · linked", tid))
                shown.append(tid)
        later = [str(m["track_id"]) for m in self._link_group
                 if not m.get("is_current") and str(m["track_id"]) not in links]
        is_first = link_ok and not links and bool(later)

        def summary(values: tuple[Any, ...]) -> str:
            if values:
                return f"{LINK_GLYPH_NEXT} " + ", ".join(f"ID {value}" for value in values)
            if is_first:
                return "starts the individual · continued by " + ", ".join(f"ID {tid}" for tid in later)
            if entries:
                return f"{len(entries)} earlier track{'s' if len(entries) > 1 else ''} · none linked"
            return f"no earlier track within {VALIDATION_SAME_INDIVIDUAL_WINDOW_S:g} s"

        self._syncing_links = True  # not an edit
        try:
            self.link_combo.set_items(entries)
            self.link_combo.set_summary(summary)
            self.link_combo.set_value(tuple(tid for tid in shown if tid in links))
        finally:
            self._syncing_links = False
        self.link_combo.setEnabled(link_ok and bool(entries))
        self.link_btn.blockSignals(True)  # a shown state, not an edit
        if self._batch_link is not None:
            enabled, checked, tooltip = self._batch_link
            self.link_btn.setChecked(checked)
            self.link_btn.setEnabled(enabled)
            self.link_btn.setToolTip(tooltip)
        else:
            self.link_btn.setChecked(bool(links) and link_ok)
            self.link_btn.setEnabled(link_ok and bool(entries))
            self.link_btn.setToolTip(
                "This track starts an individual continued by " + ", ".join(f"ID {tid}" for tid in later)
                + (" · tick an earlier track to link it further back" if entries else "")
                if is_first else _LINK_BTN_TOOLTIP)
        self.link_btn.blockSignals(False)
        tinted = is_first and self._batch_link is None
        if self.link_btn.property("chainFirst") != tinted:
            self.link_btn.setProperty("chainFirst", tinted)
            self.link_btn.style().unpolish(self.link_btn)
            self.link_btn.style().polish(self.link_btn)
        others = [m for m in self._link_group if not m.get("is_current")]
        if not others or not link_ok:
            self.individual_label.setText(f"<span style='color:{_theme_color('subtext0', '#9E9E9E')}'>-</span>")
            self.individual_label.setToolTip("")
        else:
            parts = []
            for m in self._link_group:
                tid = str(m["track_id"])
                if m.get("is_current"):
                    parts.append(f"<b>[ID {tid}]</b>")
                elif m.get("in_queue"):
                    parts.append(f"<a href='{tid}' style='color:{COLOR_FLAG_SAME_INDIVIDUAL}'>ID {tid}</a>")
                else:
                    parts.append(f"<span style='color:{_theme_color('subtext0', '#9E9E9E')}'>ID {tid}</span>")
                if m.get("warning"):
                    parts[-1] += f" <span style='color:{COLOR_VERDICT['corrected']}'>⚠ {m['warning']}</span>"
                elif m.get("note"):
                    parts[-1] += f" <span style='color:{_theme_color('subtext0', '#9E9E9E')}'>({m['note']})</span>"
            first = next((m for m in self._link_group if m.get("start_time")), None)
            last = next((m for m in reversed(self._link_group) if m.get("end_time")), None)
            span = (f" · {_fmt_time(first['start_time'])[11:]}–{_fmt_time(last['end_time'])[11:]}"
                    if first and last else "")
            text = " → ".join(parts) + f" · {len(self._link_group)} tracks{span}"
            notes = []
            if any(not m.get("in_queue") and not m.get("is_current") for m in self._link_group):
                notes.append("grayed IDs are not in your queue - their crops are shown in the comparison strip")
            if self._link_order_hint:
                notes.append("sort chronologically to step through neighboring tracks")
            if notes:
                text += f" · <span style='color:{_theme_color('subtext0', '#9E9E9E')}'>{'; '.join(notes)}</span>"
            self.individual_label.setText(text)
            self.individual_label.setToolTip("; ".join(notes) if notes else "")

    def _apply_link_style(self) -> None:
        """Style the 'Same individual' button; a first track gets a tint (computed, no rgba blending)."""
        link = QColor(COLOR_FLAG_SAME_INDIVIDUAL)
        back = QApplication.palette().color(QPalette.ColorRole.Button)
        tint = QColor(*(round((a + b) / 2) for a, b in
                        ((link.red(), back.red()), (link.green(), back.green()), (link.blue(), back.blue()))))
        self.link_btn.setStyleSheet(
            _FLAG_BTN_STYLE
            + f"QPushButton:checked {{ background-color: {COLOR_FLAG_SAME_INDIVIDUAL}; color: #F5F5F5; }}"
            f'QPushButton[chainFirst="true"]:!checked {{ background-color: {tint.name()}; }}'
        )

    def _apply_link_values(self) -> None:
        """Store the ticked entries as links."""
        if self._loading or self._syncing_links:
            return
        links = join_links([str(value) for value in self.link_combo.value()])
        if links == self._verdict.same_individual_as:
            return  # already stored
        self._verdict.same_individual_as = links
        self._commit(validate=self._verdict.validated)

    def _on_agree(self) -> None:
        """Agree down to the model rank."""
        self.agree_down_to(self._model_rank)

    def _on_fp_toggled(self, checked: bool) -> None:
        """Mark the track as not an arthropod, or withdraw that verdict (also for the batch)."""
        if self._loading:
            return
        if checked:
            self._verdict.false_positive = True
            self._commit(one_click=True)
        else:
            self._withdraw_verdict()
        self.batch_action.emit("false_positive", {"checked": checked})

    def _on_all_indet_clicked(self, checked: bool) -> None:
        """Set every rank to indet., or withdraw that verdict (also for the batch)."""
        if checked:
            self.set_all_indet()
        else:
            self._withdraw_verdict()
            self.batch_action.emit("indet", {"checked": False})

    def _on_indet_clicked(self, rank: str, checked: bool) -> None:
        """Set a rank (and below) to indet. with its '?' button, or restore the model name."""
        if checked:
            self.set_indet(rank)
            return
        model_name = self._restorable_model_name(rank)
        if model_name is None:
            self._sync_widgets()
            return
        val = dict(self._verdict.val)
        val[rank] = model_name
        self._store_val(val)

    def _on_taxon_committed(self, rank: str, text: str) -> None:
        """Store a name entered at a rank (listed or free text) and move to the next rank."""
        if self._loading:
            return
        text = self._full_name(rank, text.strip())
        val = dict(self._verdict.val)
        if not text or text == INDET_LABEL:
            for r in [rank, *ranks_below(rank)]:
                val[r] = INDET_LABEL
        else:
            lineage = self._taxonomy.lineage_for(rank, text, context=val)
            if lineage is None:
                # Free text: keep the higher ranks, remember the lineage
                new_lineage = {r: (val.get(r) if val.get(r) not in (None, INDET_LABEL) else None)
                               for r in ranks_up_to(rank)}
                new_lineage[rank] = text
                self._taxonomy.add_lineages([new_lineage], "free_text")
                self._full_models.pop(rank, None)
                self.custom_taxon_added.emit(rank, new_lineage)
                lineage = new_lineage
            text = lineage.get(rank) or text  # the listed spelling of a name typed in another case
            changed = val.get(rank) != text
            for r in ranks_up_to(rank):
                if lineage.get(r):
                    val[r] = lineage[r]
            val[rank] = text
            if changed:
                for r in ranks_below(rank):
                    val[r] = INDET_LABEL
        self._store_val(val)
        # Move on to the next deeper rank for fast typing
        deeper = ranks_below(rank)
        if deeper and text and text != INDET_LABEL:
            QTimer.singleShot(0, lambda: self.focus_rank(deeper[0]))

    def _on_sex_clicked(self, sex: str, checked: bool) -> None:
        """Set the sex, or clear it when the set button is clicked again."""
        if self._loading:
            return
        self._verdict.sex = sex if checked else None  # sex is mutually exclusive
        self._commit(validate=self._verdict.validated)

    def _on_inc_toggled(self, checked: bool) -> None:
        """Add or remove 'inc.' qualifier."""
        if self._loading:
            return
        self._verdict.qualifier = with_qualifier(self._verdict.qualifier, ID_QUALIFIER_INC, checked)
        self._commit(validate=self._verdict.validated)

    def _on_agg_toggled(self, checked: bool) -> None:
        """Add or remove 'agg.' qualifier."""
        if self._loading:
            return
        self._verdict.qualifier = with_qualifier(self._verdict.qualifier, ID_QUALIFIER_AGG, checked)
        self._commit(validate=self._verdict.validated)

    def _on_quality_toggled(self, checked: bool) -> None:
        """Set or clear the low image quality flag (also for the batch)."""
        if self._loading:
            return
        self._verdict.image_quality = "low" if checked else "ok"
        reason = self.quality_reason.currentData() or IMAGE_QUALITY_REASONS[0]
        self._verdict.image_quality_reason = reason if checked else None
        self._commit(validate=self._verdict.validated)
        self.batch_action.emit("low_quality", {"checked": checked, "reason": self._verdict.image_quality_reason})

    def _on_link_toggled(self, checked: bool) -> None:
        """Link the nearest earlier track, or remove all links; with a selection, link or unlink the selection."""
        if self._loading:
            return
        if self._batch_link is not None:
            self.batch_action.emit("same_individual", {"checked": checked})  # the page links the selected tracks
            return
        if checked:
            if not self._link_candidates:
                self._sync_widgets()
                return
            self._verdict.same_individual_as = with_link(None, str(self._link_candidates[0]["track_id"]), True)
        else:
            self._verdict.same_individual_as = None
        self._commit(validate=self._verdict.validated)

    def _on_link_values(self, _values: tuple[Any, ...]) -> None:
        """Apply the ticked dropdown entries after the combo's own signal."""
        if self._loading or self._syncing_links:
            return
        QTimer.singleShot(0, self._apply_link_values)

    def _on_multiple_toggled(self, checked: bool) -> None:
        """Set or clear the multiple individuals flag (also for the batch)."""
        if self._loading:
            return
        kind = (self.multiple_type.currentData() or MULTIPLE_INDIVIDUALS_TYPES[0]) if checked else None
        self._verdict.multiple_individuals = kind
        self._commit(validate=self._verdict.validated)
        self.batch_action.emit("multiple_individuals", {"checked": checked, "kind": kind})

    def _on_quality_reason_changed(self, _index: int) -> None:
        """Store a changed reason of the low image quality flag (also for the batch)."""
        if self._loading or self._verdict.image_quality != "low":
            return
        self._verdict.image_quality_reason = self.quality_reason.currentData() or IMAGE_QUALITY_REASONS[0]
        self._commit(validate=self._verdict.validated)
        self.batch_action.emit("low_quality", {"checked": True, "reason": self._verdict.image_quality_reason})

    def _on_multiple_type_changed(self, _index: int) -> None:
        """Store a changed type of the multiple individuals flag (also for the batch)."""
        if self._loading or self._verdict.multiple_individuals is None:
            return
        kind = self.multiple_type.currentData() or MULTIPLE_INDIVIDUALS_TYPES[0]
        self._verdict.multiple_individuals = kind
        self._commit(validate=self._verdict.validated)
        self.batch_action.emit("multiple_individuals", {"checked": True, "kind": kind})

    def _on_notes_edited(self, _text: str = "") -> None:
        """Store the typed note and show it in the tooltip."""
        self.notes_edit.setToolTip(self.notes_edit.text())  # full note in the tooltip
        if self._loading:
            return
        self._verdict.notes = self.notes_edit.text()
        self.notes_changed.emit(self._verdict.notes)
        self._render_status()


class _ZoomSlider(QSlider):
    """Slider that moves one step per wheel notch."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.setValue(self.value() + (1 if delta > 0 else -1))
        event.accept()


class ValidationPage(QWidget):
    """Second page, where tracks are validated: track list, crops, info and verdict panel; saves automatically.

    Signals:
        back_requested(): Return to the setup page.
    """
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the validation page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.run: ValidationRun | None = None
        self.session: ValidationSession | None = None
        self._session_log: logging.Handler | None = None
        self.taxonomy: TaxonomyIndex = TaxonomyIndex()
        self._keys: list[TrackKey] = []
        self._key_set: set[TrackKey] = set()
        self._key_pos: dict[TrackKey, int] = {}
        self._current: TrackKey | None = None
        self._image_rows: list[dict[str, Any]] = []
        self._single_index = 0
        self._view_mode = "grid"
        self._grid_zoom: GridZoom = ("px", GRID_ZOOM_STEPS[2])
        self._single_zoom: int | str = FIT
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_AUTOSAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.save_now)
        self._syncing_list = False
        self._shown_at: float | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_step)
        self._crops_hidden: bool | None = None
        self._cache = ThumbnailCache(parent=self)
        self._create_ui()
        self._create_shortcuts()

    def _create_ui(self) -> None:
        """Build the top bar, track list, crop views, comparison strip and verdict pane."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top bar
        top = QHBoxLayout()
        top.setSpacing(8)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m validated (%p%)")
        self.progress_bar.setMinimumWidth(220)
        top.addWidget(self.progress_bar, 2)
        self.summary_label = QLabel("")
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        top.addWidget(self.summary_label, 3)
        self.auto_advance_cb = QCheckBox("Auto-advance")
        self.auto_advance_cb.setToolTip("Jump to the next unvalidated track after a one-click verdict (Agree, a "
                                        "rank button, Not an arthropod, Not determinable)")
        self.auto_advance_cb.setChecked(bool(_settings().value(_SETTINGS_KEY_AUTO_ADVANCE, False, type=bool)))
        self.auto_advance_cb.toggled.connect(lambda v: _settings().setValue(_SETTINGS_KEY_AUTO_ADVANCE, v))
        top.addWidget(self.auto_advance_cb)
        self.stats_btn = StyledButton("Statistics", COLOR_BTN_SAVE)
        self.stats_btn.clicked.connect(self.show_stats)
        top.addWidget(self.stats_btn)
        self.export_btn = StyledButton("Export merged results", COLOR_BTN_RUN)
        self.export_btn.setToolTip("Save the final CSV with your verdicts and determinations merged in "
                                   "(<final CSV>_validated_<validator>_<started>.csv, next to the run's final CSV)")
        self.export_btn.clicked.connect(self.export_merged)
        top.addWidget(self.export_btn)
        self.help_btn = StyledButton("Help (F1)", COLOR_BTN_HELP)
        self.help_btn.clicked.connect(self.show_help)
        top.addWidget(self.help_btn)
        self.back_btn = StyledButton("Back to setup", COLOR_BTN_RESET)
        self.back_btn.clicked.connect(self.leave_session)
        top.addWidget(self.back_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # Left: track list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        filter_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        for mode in TrackListModel.FILTERS:
            self.filter_combo.addItem(_FILTER_LABELS[mode], mode)
        self.filter_combo.currentIndexChanged.connect(self._apply_list_filter)
        filter_row.addWidget(self.filter_combo, 1)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search ID / taxon ...")
        self.search_edit.setToolTip("Type what a row shows (ID 140, S3, D1) or part of a taxon name")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_list_filter)
        filter_row.addWidget(self.search_edit, 1)
        left_layout.addLayout(filter_row)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        for mode in TrackListModel.SORTS:
            self.sort_combo.addItem(_SORT_LABELS[mode], mode)
        self.sort_combo.setToolTip("Order of the track list. A taxon sort keeps the lower ranks together, highest "
                                   "probability first. Equal values are ordered alphabetically by taxon name.")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(self.sort_combo, 1)
        left_layout.addLayout(sort_row)
        self.track_model = TrackListModel(self)
        self.track_model.set_status_provider(self._status_of)
        self.track_view = QListView()
        self.track_view.setModel(self.track_model)
        self.track_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.track_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_view.setUniformItemSizes(True)
        self.track_view.setToolTip("Click: show track · Ctrl/Shift-click or Ctrl+A: select several tracks "
                                   "for batch verdicts")
        self.track_view.selectionModel().currentChanged.connect(self._on_list_current_changed)
        self.track_view.selectionModel().selectionChanged.connect(self._on_list_selection_changed)
        left_layout.addWidget(self.track_view, 1)
        self.list_count_label = QLabel("")
        left_layout.addWidget(self.list_count_label)
        splitter.addWidget(left)

        # Center: track info + crop viewer + comparison strip
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        self.info_panel = TrackInfoPanel()
        center_layout.addWidget(self.info_panel, 0)
        zoom_row = QHBoxLayout()
        self.grid_mode_btn = QPushButton("Grid")
        self.grid_mode_btn.setCheckable(True)
        self.grid_mode_btn.setToolTip("All crops of the track (Esc)")
        self.single_mode_btn = QPushButton("Single")
        self.single_mode_btn.setCheckable(True)
        self.single_mode_btn.setToolTip("One crop, magnified without smoothing (double-click or Enter on a crop)")
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.grid_mode_btn)
        mode_group.addButton(self.single_mode_btn)
        self.grid_mode_btn.clicked.connect(lambda: self._set_view_mode("grid"))
        self.single_mode_btn.clicked.connect(lambda: self._set_view_mode("single"))
        zoom_row.addWidget(self.grid_mode_btn)
        zoom_row.addWidget(self.single_mode_btn)
        zoom_row.addSpacing(8)
        zoom_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = _ZoomSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, len(GRID_ZOOM_STEPS) - 1)
        self.zoom_slider.setToolTip("Zoom (Ctrl+wheel or +/-): in the grid from smaller cells up to 1x, 2x, 3x ... "
                                    "screen pixels per image pixel; in the single view from 1:1 to 10x")
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        zoom_row.addWidget(self.zoom_slider, 1)
        self.zoom_label = QLabel("")
        self.zoom_label.setMinimumWidth(self.fontMetrics().horizontalAdvance("Crop · fit 10×  "))
        zoom_row.addWidget(self.zoom_label)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setCheckable(True)
        self.fit_btn.setToolTip("Single crop at the largest supported magnification that fits the view (Ctrl+9)")
        self.fit_btn.clicked.connect(lambda: self._open_single(self._single_index, zoom=FIT))
        zoom_row.addWidget(self.fit_btn)
        self.original_btn = QPushButton("Original (1:1)")
        self.original_btn.setCheckable(True)
        self.original_btn.setToolTip("Single crop with one image pixel per screen pixel (Ctrl+0)")
        self.original_btn.clicked.connect(lambda: self._open_single(self._single_index, zoom=1))
        zoom_row.addWidget(self.original_btn)
        self.play_btn = QPushButton("▶ Play  (Space)")
        self.play_btn.setCheckable(True)
        self.play_btn.setToolTip("Play the crops of the track once at the selected interval")
        self.play_btn.toggled.connect(self._on_play_toggled)
        zoom_row.addWidget(self.play_btn)
        self.play_interval = QComboBox()
        for ms in VALIDATION_PLAYBACK_INTERVALS_MS:
            self.play_interval.addItem(f"{ms / 1000:g} s", ms)
        stored_ms = _settings().value(_SETTINGS_KEY_PLAYBACK_MS, VALIDATION_PLAYBACK_INTERVAL_DEFAULT_MS, type=int)
        self.play_interval.setCurrentIndex(max(self.play_interval.findData(stored_ms), 0))
        self.play_interval.setToolTip("Time between crops during playback")
        self.play_interval.currentIndexChanged.connect(self._on_play_interval_changed)
        zoom_row.addWidget(self.play_interval)
        center_layout.addLayout(zoom_row)
        self.image_stack = QStackedWidget()
        self.grid_model = CropGridModel(self._cache, self)
        self.grid_model.max_side_changed.connect(self._on_grid_max_side)
        self.grid_view = CropGridView()
        self.grid_view.setModel(self.grid_model)
        self.grid_view.zoom_requested.connect(self._zoom_by)
        self.grid_view.open_requested.connect(self._open_single)
        self.grid_view.selectionModel().currentChanged.connect(self._on_grid_current_changed)
        self.image_stack.addWidget(self.grid_view)
        self.single_view = SingleImageView()
        self.single_view.zoom_requested.connect(self._zoom_by)
        self.single_view.zoom_resolved.connect(self._on_single_zoom_resolved)
        self.single_view.navigate.connect(self._single_navigate)
        self.single_view.closed.connect(self._close_single)
        self.single_view.play_toggled.connect(self.toggle_playback)
        self.image_stack.addWidget(self.single_view)
        center_layout.addWidget(self.image_stack, 1)
        self.image_info_label = QLabel("")
        self.image_info_label.setWordWrap(True)
        center_layout.addWidget(self.image_info_label, 0)
        # Comparison strip: one crop per similar track
        compare_header = QHBoxLayout()
        self.compare_toggle = QToolButton()
        self.compare_toggle.setCheckable(True)
        self.compare_toggle.setChecked(bool(_settings().value(_SETTINGS_KEY_COMPARE, True, type=bool)))
        self.compare_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.compare_toggle.setAutoRaise(True)
        self.compare_toggle.setText("Compare with similar tracks")
        self.compare_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.compare_toggle.toggled.connect(self._on_compare_toggled)
        compare_header.addWidget(self.compare_toggle)
        self.compare_label = QLabel("")
        # Clipped when the pane is narrow, never widens it (the tooltip holds the full line)
        self.compare_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        compare_header.addWidget(self.compare_label, 1)
        center_layout.addLayout(compare_header)
        self.compare_model = CropGridModel(self._cache, self)
        self.compare_view = CropGridView()
        self.compare_view.setModel(self.compare_model)
        self.compare_view.setWrapping(False)
        self.compare_view.set_icon_side(_COMPARE_CELL)
        self.compare_view.setFixedHeight(_COMPARE_CELL + 26 + 22)  # cell, label row and horizontal scrollbar
        self.compare_view.setToolTip(
            "One crop per similar track: the other tracks of this individual, the earlier tracks offered as a "
            "link, then the tracks of the run the model predicted as the same taxon, most confident first"
        )
        self.compare_view.open_requested.connect(self._on_compare_open)
        center_layout.addWidget(self.compare_view, 0)
        self._compare_keys: list[TrackKey | None] = []
        self._link_candidates: list[dict[str, Any]] = []
        self._link_group: list[dict[str, Any]] = []
        self.compare_toggle.setArrowType(Qt.ArrowType.DownArrow if self.compare_toggle.isChecked()
                                         else Qt.ArrowType.RightArrow)
        self.compare_view.setVisible(self.compare_toggle.isChecked())
        splitter.addWidget(center)

        # Right: verdict
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self.verdict_panel = VerdictPanel()
        self.verdict_panel.verdict_changed.connect(self._on_verdict_changed)
        self.verdict_panel.notes_changed.connect(self._on_notes_changed)
        self.verdict_panel.reset_requested.connect(self._reset_current)
        self.verdict_panel.action_done.connect(self._after_one_click)
        self.verdict_panel.custom_taxon_added.connect(self._on_custom_taxon)
        self.verdict_panel.batch_action.connect(self._on_batch_action)
        self.verdict_panel.apply_to_selection_requested.connect(self._apply_determination_to_selection)
        self.verdict_panel.jump_requested.connect(self._on_link_jump)
        right_layout.addWidget(
            create_groupbox("Your verdict", COLOR_GROUP_VALIDATION, self.verdict_panel)  # type: ignore[arg-type]
        )
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Previous  (PgUp)")
        self.prev_btn.setToolTip("Previous track (PgUp or Ctrl+Left); the arrow keys move between crops")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn = QPushButton("Next  (PgDn)")
        self.next_btn.setToolTip("Next track (PgDn or Ctrl+Right); the arrow keys move between crops")
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.next_todo_btn = StyledButton("Next unvalidated  (Enter)", COLOR_BTN_VALIDATE)
        self.next_todo_btn.setToolTip("Next track without a verdict (Enter)")
        self.next_todo_btn.clicked.connect(self.next_unvalidated)
        for btn in (self.prev_btn, self.next_btn, self.next_todo_btn):
            btn.setMinimumWidth(60)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.next_todo_btn)
        right_layout.addLayout(nav)
        self.save_label = QLabel("")
        self.save_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.save_label)
        right_layout.addStretch(1)
        right_scroll.setWidget(right)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(right_scroll)
        # Panes must not collapse (except the track list)
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, True)
        left.setMinimumWidth(200)
        center.setMinimumWidth(420)
        # The right pane is at least as wide as the verdict panel
        right.adjustSize()
        right_min = max(380, right.minimumSizeHint().width() + 12)
        right_scroll.setMinimumWidth(right_min)
        # Extra space goes to the center pane (crops viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 10)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 1200, right_min])
        self.splitter = splitter
        self._track_list_width = 260

    def _create_shortcuts(self) -> None:
        """Create the window-wide keyboard shortcuts."""
        def shortcut(keys: str, handler: Callable[[], None]) -> None:
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(handler)

        shortcut("Ctrl+0", lambda: self._open_single(self._single_index, zoom=1))
        shortcut("Ctrl+9", lambda: self._open_single(self._single_index, zoom=FIT))
        shortcut("A", lambda: self.verdict_panel.agree_down_to(self.session.model_rank) if self.session else None)
        shortcut("F", self.verdict_panel.toggle_false_positive)
        shortcut("U", self.verdict_panel.toggle_indeterminate)
        shortcut("I", self.verdict_panel.toggle_uncertain)
        shortcut("G", self.verdict_panel.toggle_aggregate)
        shortcut("Q", self.verdict_panel.toggle_low_quality)
        shortcut("M", self.verdict_panel.toggle_multiple)
        shortcut("L", self.verdict_panel.toggle_link)
        shortcut("R", self._reset_current)
        shortcut("Space", self.toggle_playback)
        for i, rank in enumerate(VALIDATION_RANKS[:-1], start=1):
            shortcut(str(i), lambda r=rank: self.verdict_panel.agree_down_to(r))
        shortcut("PgDown", lambda: self._step(1))
        shortcut("PgUp", lambda: self._step(-1))
        shortcut("Ctrl+Right", lambda: self._step(1))
        shortcut("Ctrl+Left", lambda: self._step(-1))
        shortcut("Return", self._on_return_key)
        shortcut("Enter", self._on_return_key)
        shortcut("Ctrl+S", self.save_now)
        shortcut("Ctrl+L", self.toggle_track_list)
        shortcut("F1", self.show_help)

    def load_session(self, run: ValidationRun, session: ValidationSession, taxonomy: TaxonomyIndex) -> None:
        """Open a session: log file, track list, sort options, zoom and the start track."""
        self.save_now()
        self._close_session_log()  # one log per open session
        self._session_log = open_session_log(session)
        logger.info("=" * 60)
        logger.info("Validation session %s of '%s' opened by '%s'", session.session_id, run.root.name,
                    session.validator)
        self.run = run
        self.session = session
        self.taxonomy = taxonomy
        self.verdict_panel.set_taxonomy(taxonomy)
        self.verdict_panel.set_blind(session.blind)
        self.track_model.set_blind(session.blind, apply=False)
        self.sort_combo.blockSignals(True)
        stored_sort = str(_settings().value(_SETTINGS_KEY_SORT, TrackListModel.DEFAULT_SORT))
        if stored_sort not in TrackListModel.SORTS:
            stored_sort = TrackListModel.DEFAULT_SORT
        # Taxon sorts: only the run's ranks, none when blind
        for mode, rank in TrackListModel.TAXON_SORTS.items():
            available = not session.blind and rank in run.active_ranks
            item = self.sort_combo.model().item(self.sort_combo.findData(mode))  # type: ignore[union-attr]
            item.setEnabled(available)
            if not available and stored_sort == mode:
                stored_sort = TrackListModel.DEFAULT_SORT
        # Setting sort: at least two setting groups among the session's tracks
        session_tracks = run.tracks.join(
            pl.DataFrame(session.keys(), schema=dict.fromkeys(TRACK_KEY_COLUMNS, pl.String), orient="row"),
            on=TRACK_KEY_COLUMNS, how="semi")
        setting_item = self.sort_combo.model().item(self.sort_combo.findData("setting"))  # type: ignore[union-attr]
        setting_item.setEnabled(setting_strata_available(session_tracks))
        if not setting_item.isEnabled() and stored_sort == "setting":
            stored_sort = TrackListModel.DEFAULT_SORT
        for mode, (field, _) in TrackListModel.NUMERIC_SORTS.items():
            available = field not in _SORT_COLUMNS or _SORT_COLUMNS[field] in run.tracks.columns
            item = self.sort_combo.model().item(self.sort_combo.findData(mode))  # type: ignore[union-attr]
            item.setEnabled(available)
            if not available and stored_sort == mode:
                stored_sort = TrackListModel.DEFAULT_SORT
        self.sort_combo.setCurrentIndex(max(self.sort_combo.findData(stored_sort), 0))
        self.sort_combo.blockSignals(False)
        self.track_model.set_sort(self.sort_combo.currentData(), apply=False)
        self._keys = session.keys()
        self._key_set = set(self._keys)
        self._key_pos = {key: i for i, key in enumerate(self._keys)}
        extra_cols = [c for c in ("start_time", *_SORT_COLUMNS.values(), SETTING_COLUMN)
                      if c in run.tracks.columns]
        extras: dict[TrackKey, dict[str, Any]] = {
            tuple(row[:3]): dict(zip(extra_cols, row[3:], strict=True))
            for row in run.tracks.select([*TRACK_KEY_COLUMNS, *extra_cols]).iter_rows()
        } if extra_cols else {}
        rows = []
        for key in self._keys:
            base = session.base_row(key) or {}
            extra = extras.get(key, {})
            rows.append({
                "key": key, "device_id": key[0], "session_id": key[1], "track_id": key[2],
                "device_no": run.device_numbers.get(key[0]), "setting": extra.get(SETTING_COLUMN),
                "pred": base.get("pred"), "prob": base.get("pred_prob_weighted"),
                **{f"pred_{rank}": base.get(f"pred_{rank}") for rank in VALIDATION_RANKS},
                "start_time": extra.get("start_time"),
                **{field: extra.get(column) for field, column in _SORT_COLUMNS.items()},
            })
        self.track_model.set_rows(rows)
        self.progress_bar.setMaximum(max(len(self._keys), 1))
        self._grid_zoom = _parse_grid_zoom(str(_settings().value(_SETTINGS_KEY_ZOOM, "")))
        self._single_zoom = _parse_single_zoom(str(_settings().value(_SETTINGS_KEY_SINGLE_ZOOM, FIT)))
        self._view_mode = "grid"
        self._apply_zoom()
        self._update_progress()
        last_key = session.meta["ui"].get("last_track")
        start_key = tuple(last_key) if isinstance(last_key, list) and len(last_key) == 3 else None
        if start_key not in self._keys:
            validated = set(session.validated_keys())
            start_key = next((k for k in self._keys if k not in validated), self._keys[0] if self._keys else None)
        if start_key is not None:
            self._show_track(start_key)
        if not session.meta["ui"].get("guidelines_shown"):
            session.set_ui("guidelines_shown", True)
            QTimer.singleShot(_GUIDELINES_DELAY_MS, self.show_help)

    def save_now(self) -> None:
        """Save the session if it has unsaved changes (autosave, Ctrl+S, before leaving)."""
        self._flush_time()
        if self.session is None or not self.session.unsaved:
            return
        try:
            self.session.save()
            self.save_label.setText(f"Saved ✓ {self.session.meta['updated'][11:19]}")
        except OSError as e:
            logger.exception("Saving the validation session failed")
            self.save_label.setText(f"<span style='color:{_error_color()}'>Save failed: {e}</span>")

    @Slot(object)
    def forget_session(self, csv_path: Path) -> None:
        """Close the session of 'csv_path' if it is loaded (before deletion)."""
        if self.session is not None and self.session.csv_path == Path(csv_path):
            self._save_timer.stop()
            self._close_session_log()
            self.session = None
            self.save_label.setText("Session deleted")

    def _update_progress(self) -> None:
        """Update the progress bar and the verdict summary of the top bar."""
        if self.session is None:
            return
        validated, total = self.session.progress()
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(validated)
        counts: dict[str, int] = {}
        low = linked = 0
        several_taxa = dict.fromkeys(MULTIPLE_INDIVIDUALS_NOT_ASSESSABLE, 0)
        for key in self.session.validated_keys():
            verdict_type, _, _ = self.session.derived(key)
            counts[verdict_type or ""] = counts.get(verdict_type or "", 0) + 1
            entry = self.session.verdicts[key]
            if entry.image_quality == "low":
                low += 1
            if entry.multiple_individuals in several_taxa and not entry.false_positive:
                several_taxa[entry.multiple_individuals] += 1
            if entry.same_individual_as and not entry.false_positive:
                linked += 1
        parts = []
        for verdict_type in VERDICT_TYPES:
            n = counts.get(verdict_type, 0)
            if n:
                pct = 100 * n / max(validated, 1)
                parts.append(f"<span style='color:{COLOR_VERDICT[verdict_type]}'>"
                             f"{VERDICT_LABELS[verdict_type]} {pct:.0f} %</span>")
        if linked:
            parts.append(f"<span style='color:{COLOR_FLAG_SAME_INDIVIDUAL}'>linked {linked}</span>")
        for kind, n in several_taxa.items():
            if n:
                parts.append(f"<span style='color:{COLOR_FLAG_MULTIPLE_INDIVIDUALS}'>"
                             f"{MULTIPLE_INDIVIDUALS_LABELS[kind].lower()} {n}</span>")
        if low:
            parts.append(f"<span style='color:{COLOR_FLAG_LOW_QUALITY}'>low quality {low}</span>")
        self.summary_label.setText(" · ".join(parts))

    def _flush_time(self) -> None:
        """Add the time spent on the current track to its effort."""
        if self.session is not None and self._current is not None and self._shown_at is not None:
            self.session.add_seconds(self._current, time.monotonic() - self._shown_at)
        self._shown_at = time.monotonic() if self._current is not None else None

    def refresh_theme(self) -> None:
        """Re-render texts with theme colors."""
        self.info_panel.refresh_theme()
        self.verdict_panel.refresh_theme()

    @Slot()
    def show_help(self) -> None:
        """Show the guidelines and keyboard shortcuts."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Validation help")
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.document().setDocumentMargin(14)
        browser.document().setIndentWidth(16)
        browser.document().setDefaultStyleSheet(_HELP_CSS)
        browser.setHtml(_GUIDELINES_HTML + _SHORTCUTS_HTML)
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        # Fit the content, at most the screen size
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        width = min(760, int(screen.width() * 0.9))
        browser.document().setTextWidth(width - 40)
        content_height = int(browser.document().size().height()) + buttons.sizeHint().height() + 60
        dialog.resize(width, min(content_height, int(screen.height() * 0.85)))
        dialog.exec()
        dialog.deleteLater()  # not kept as a child of the page

    @Slot()
    def show_stats(self) -> None:
        """Save and open the statistics window."""
        if self.session is None:
            return
        self.save_now()
        dialog = StatsDialog(self.session, self)
        dialog.exec()
        dialog.deleteLater()  # tables, charts and the other sessions are not kept

    @Slot()
    def export_merged(self) -> None:
        """Save and write the merged export of the session."""
        if self.session is None or self.run is None:
            return
        self.save_now()
        try:
            out_path = self.session.export_merged(self.run, self.taxonomy)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        validated, total = self.session.progress()
        QMessageBox.information(
            self, "Export written",
            f"Merged results written to:\n{out_path}\n\n{validated} of {total} sampled tracks are validated; "
            "unvalidated tracks keep the model prediction (combined_source = model).",
        )

    def leave_session(self) -> None:
        """Save, close the session log and return to the setup page ('Back' button)."""
        self.save_now()
        self._close_session_log()
        self.back_requested.emit()

    def _status_of(self, key: TrackKey) -> RowStatus:
        """Return the RowStatus of a track for the track list."""
        if self.session is None or self.run is None:
            return RowStatus()
        group = self.session.group_of(key)
        individual = anchor_of(group, self.run.track_start_times) if len(group) > 1 else None
        n_individual = len(group) if individual else 0
        verdict = self.session.verdict(key)
        if verdict is None:
            return RowStatus(individual=individual, n_individual=n_individual)
        linked = verdict.same_individual_as if not verdict.false_positive else None
        continued = join_links(self.session.successors_of(key))
        if not verdict.validated:
            return RowStatus(None, verdict.image_quality == "low",
                             has_qualifier(verdict.qualifier, ID_QUALIFIER_INC),
                             verdict.multiple_individuals, linked, continued, individual, n_individual)
        verdict_type, _, lowest = self.session.derived(key)
        return RowStatus(verdict_type, verdict.image_quality == "low",
                         bool(has_qualifier(verdict.qualifier, ID_QUALIFIER_INC) and lowest),
                         verdict.multiple_individuals if not verdict.false_positive else None,
                         linked, continued, individual, n_individual)

    def _apply_list_filter(self, *_args: Any) -> None:
        """Apply the list filter and search, keeping the shown track selected."""
        self.track_model.set_filter(self.filter_combo.currentData(), self.search_edit.text())
        self.list_count_label.setText(f"{self.track_model.rowCount()} of {len(self._keys)} tracks shown")
        if self._current is not None:
            list_row = self.track_model.row_of_key(self._current)
            if list_row >= 0:
                self._select_list_row(self.track_model.index(list_row))

    def _on_sort_changed(self, _index: int) -> None:
        """Re-sort the track list, remember the mode and scroll to the top."""
        mode = self.sort_combo.currentData()
        _settings().setValue(_SETTINGS_KEY_SORT, mode)
        self.track_model.set_sort(mode)
        self._apply_list_filter()
        self.track_view.scrollToTop()

    def _visible_keys(self) -> list[TrackKey]:
        """Return the keys of the tracks shown in the list, in list order."""
        return [k for k in (self.track_model.key_at(i) for i in range(self.track_model.rowCount())) if k is not None]

    def _select_list_row(self, index: QModelIndex) -> None:
        """Make 'index' the current and only selected row (also with Ctrl held)."""
        self._syncing_list = True
        self.track_view.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        self._syncing_list = False

    def _on_list_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        """Show the track that became current in the list."""
        if self._syncing_list:
            return
        key = self.track_model.key_at(current.row())
        if key is not None and key != self._current:
            self._show_track(key)

    def _on_list_selection_changed(self, *_args: Any) -> None:
        """Update the batch banner for the selected tracks."""
        self._sync_batch()

    def _sync_batch(self) -> None:
        """Update the batch banner and the 'Same individual' button for the selected tracks."""
        self.verdict_panel.set_batch_count(len(self._selected_keys()), len(self._batch_targets()))
        self.verdict_panel.set_batch_link(self._batch_link_state())

    def _selected_keys(self) -> list[TrackKey]:
        """Return the keys of the selected tracks."""
        keys = [self.track_model.key_at(idx.row()) for idx in self.track_view.selectionModel().selectedIndexes()]
        return [k for k in keys if k is not None]

    @Slot()
    def toggle_track_list(self) -> None:
        """Collapse or restore the track list pane."""
        sizes = self.splitter.sizes()
        if sizes[0] > 0:
            self._track_list_width = sizes[0]
            self.splitter.setSizes([0, sizes[1] + sizes[0], sizes[2]])
        else:
            width = max(self._track_list_width, 200)
            self.splitter.setSizes([width, max(sizes[1] - width, 420), sizes[2]])

    def _show_track(self, key: TrackKey) -> None:
        """Show a track (info, verdict form, comparison, crops) and prefetch the next one."""
        if self.run is None or self.session is None:
            return
        self._flush_time()
        self._stop_playback()
        self._cache.cancel_pending()
        self._current = key
        self._shown_at = time.monotonic()
        row = self.run.track_row(key) or {}
        model = self.session.model_lineage(key)
        self.info_panel.show_track(self.run, key, row, model)
        self._link_candidates, self._link_group = self._link_context(key)
        self.verdict_panel.set_link_context(self._link_candidates, self._link_group,
                                            order_hint=self.sort_combo.currentData() not in TrackListModel.CHRONO_SORTS)
        self.verdict_panel.load(model, self.session.verdict(key), self.session.model_rank,
                                self.session.recent_lineages())
        self._sync_blind_visibility()
        self._populate_comparison(key, row)
        images = self.run.images_for(key)
        self._image_rows = []
        for img in images.to_dicts():
            path = self.run.crop_abs_path(img.get("crop_path"))
            self._image_rows.append({
                "path": str(path) if path else "", "label": _fmt_time(img.get("timestamp"))[11:19],
                "tooltip": self._crop_tooltip(img), "data": img,
            })
        self.grid_model.set_images(self._image_rows)
        if self._view_mode == "grid":
            self._apply_zoom()
        self.image_info_label.setText(f"{len(self._image_rows)} crops - click one for details, double-click to enlarge")
        if self._image_rows:
            self.grid_view.setCurrentIndex(self.grid_model.index(0))
            self.grid_view.scrollToTop()
        if self.image_stack.currentWidget() is self.single_view:
            self._single_index = 0
            self._show_single()
        # Keep the list selection in sync
        list_row = self.track_model.row_of_key(key)
        if list_row >= 0:
            idx = self.track_model.index(list_row)
            if self.track_view.currentIndex() != idx:
                self._select_list_row(idx)
                self.track_view.scrollTo(idx)
        self.session.set_ui("last_track", list(key))
        # Prefetch the next track's thumbnails
        next_key = self.track_model.key_at(list_row + 1) if list_row >= 0 else None
        if next_key is None:
            pos = self._key_pos.get(key, -1)
            next_key = self._keys[pos + 1] if 0 <= pos < len(self._keys) - 1 else None
        if next_key is not None:
            next_images = self.run.images_for(next_key)
            paths = [str(p) for p in (self.run.crop_abs_path(c)
                                      for c in next_images["crop_path"].to_list()[:_PREFETCH_NEXT_CROPS]) if p]
            self._cache.prefetch(paths)

    def _crop_tooltip(self, img: dict[str, Any]) -> str:
        """Return the crop tooltip: file, time, the model's 1st choice (not when blind) and size."""
        parts = [Path(img.get("crop_path") or "").name, _fmt_time(img.get("timestamp"))]
        top1 = next((img[c] for c in img if c.startswith("bioclip_top1_") and not c.endswith("_prob") and img[c]),
                    None)
        if top1 and not self.verdict_panel.blind_hidden:
            parts.append(f"1st choice: {top1} ({_fmt(img.get('bioclip_top1_prob'))})")
        if img.get("bbox_length_mm") is not None:
            parts.append(f"size ≈ {_fmt(img.get('bbox_length_mm'), 1)} × {_fmt(img.get('bbox_width_mm'), 1)} mm")
        return "\n".join(parts)

    def _sync_blind_visibility(self) -> None:
        """Hide the model's information while a blind track is undetermined."""
        hidden = self.verdict_panel.blind_hidden
        self.info_panel.set_prediction_visible(not hidden)
        self.compare_view.setVisible(self.compare_toggle.isChecked() and not hidden)
        self.compare_label.setVisible(not hidden)
        if hidden:
            self.compare_model.set_images([])
            self._compare_keys = []
        if hidden != self._crops_hidden:
            # Crop texts name the model's choices: shown once determined
            self._crops_hidden = hidden
            for row in self._image_rows:
                row["tooltip"] = self._crop_tooltip(row["data"])
            self.grid_model.refresh_rows()
            if self._image_rows:
                if self.image_stack.currentWidget() is self.single_view:
                    self._show_single()
                else:
                    self._show_image_info(self._image_rows[min(self._single_index, len(self._image_rows) - 1)])

    def _link_context(self, key: TrackKey) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return '(link candidates, individual)' of a track."""
        if self.run is None or self.session is None:
            return [], []
        candidates = []
        for cand in self.run.previous_tracks(key):
            other = self.session.verdict(cand["key"])
            if other is not None and other.validated and other.false_positive:
                continue
            cand["human"] = self._human_lineage(cand["key"])
            candidates.append(cand)
        group = []
        members = self.session.group_of(key)
        if len(members) > 1:
            queue = self._key_set
            for member in sort_tracks_chronologically(members, self.run.track_start_times):
                row = self.run.track_row(member) or {}
                entry = self.session.verdict(member)
                false_positive = entry is not None and entry.validated and entry.false_positive
                # Different taxa in track: the individual may be one of them, so its determination can differ
                in_track = (entry is not None and not entry.false_positive
                            and entry.multiple_individuals == DIFFERENT_TAXA_IN_TRACK)
                group.append({
                    "key": member, "track_id": member[2], "is_current": member == key, "in_queue": member in queue,
                    "human": self._human_lineage(member), "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"), "warning": "false positive" if false_positive else None,
                    "note": MULTIPLE_INDIVIDUALS_LABELS[DIFFERENT_TAXA_IN_TRACK].lower() if in_track else None,
                })
        return candidates, group

    def _human_lineage(self, key: TrackKey) -> Lineage | None:
        """Return the validator's lineage of a track, None if not validated or a false positive."""
        other = self.session.verdict(key) if self.session else None
        if other is None or not other.validated or other.false_positive:
            return None
        return {r: v for r, v in other.val.items() if v and v != INDET_LABEL} or None

    @property
    def _single_mode(self) -> bool:
        """Whether the single-image view is active."""
        return self._view_mode == "single"

    def _set_view_mode(self, mode: str) -> None:
        """Switch between the grid and the single-image view."""
        if mode == "single":
            self._open_single(self._single_index)
        else:
            self._close_single()

    def _grid_ladder(self) -> list[GridZoom]:
        """Return the grid zoom steps of the track: cell sizes below the largest crop, then 1x, 2x, ..."""
        max_side = self.grid_model.max_side
        dpr = self.grid_view.devicePixelRatioF()  # cell sizes are logical, crop sizes physical pixels
        ladder: list[GridZoom] = [("px", s) for s in GRID_ZOOM_STEPS if not max_side or s * dpr < max_side]
        k = 1
        while len(ladder) < len(GRID_ZOOM_STEPS):
            ladder.append(("x", k))
            k += 1
        return ladder

    def _grid_step_for(self, zoom: GridZoom) -> int:
        """Return the slider position of a grid zoom (nearest step)."""
        ladder = self._grid_ladder()
        if zoom in ladder:
            return ladder.index(zoom)
        mode, value = zoom
        if mode == "px":
            smaller = [i for i, (m, v) in enumerate(ladder) if m == "px" and v <= value]
            return smaller[-1] if smaller else ladder.index(("x", 1))
        magnified = [i for i, (m, v) in enumerate(ladder) if m == "x" and v <= value]
        return magnified[-1] if magnified else len(ladder) - 1

    def _apply_zoom(self) -> None:
        """Show the active view with its remembered zoom."""
        self.zoom_slider.blockSignals(True)
        if self._single_mode:
            self.image_stack.setCurrentWidget(self.single_view)
            self.zoom_slider.setRange(0, len(SINGLE_ZOOM_STEPS) - 1)
            self.single_view.set_zoom(self._single_zoom)
            self._show_single()
            self._sync_single_readout(self.single_view.effective_zoom)
            self.single_view.setFocus()
            _settings().setValue(_SETTINGS_KEY_SINGLE_ZOOM, str(self._single_zoom))
        else:
            self.image_stack.setCurrentWidget(self.grid_view)
            self.zoom_slider.setRange(0, len(GRID_ZOOM_STEPS) - 1)
            ladder = self._grid_ladder()
            step = self._grid_step_for(self._grid_zoom)
            resolved = ladder[step]
            self.grid_view.set_zoom(resolved)
            mode, value = resolved
            self.zoom_slider.setValue(step)
            magnified = "1:1" if value == 1 else f"{value}×"
            self.zoom_label.setText(f"Grid · {value} px" if mode == "px" else f"Grid · {magnified}")
            # Keep the requested zoom, not the fallback
            mode, value = self._grid_zoom
            _settings().setValue(_SETTINGS_KEY_ZOOM, f"{mode}:{value}")
        self.zoom_slider.blockSignals(False)
        self.grid_mode_btn.setChecked(not self._single_mode)
        self.single_mode_btn.setChecked(self._single_mode)
        self.fit_btn.setChecked(self._single_mode and self._single_zoom == FIT)
        self.original_btn.setChecked(self._single_mode and self._single_zoom == 1)

    def _on_zoom_slider(self, value: int) -> None:
        """Apply the zoom step of the slider to the active view."""
        if self._single_mode:
            self._single_zoom = SINGLE_ZOOM_STEPS[max(0, min(value, len(SINGLE_ZOOM_STEPS) - 1))]
        else:
            self._grid_zoom = self._grid_ladder()[max(0, min(value, len(GRID_ZOOM_STEPS) - 1))]
        self._apply_zoom()

    def _zoom_by(self, delta: int) -> None:
        """Zoom the active view by one step."""
        if self._single_mode:
            current = self.single_view.effective_zoom
            pos = SINGLE_ZOOM_STEPS.index(current) if current in SINGLE_ZOOM_STEPS else 0
            self._single_zoom = SINGLE_ZOOM_STEPS[max(0, min(pos + delta, len(SINGLE_ZOOM_STEPS) - 1))]
        else:
            step = self._grid_step_for(self._grid_zoom)
            self._grid_zoom = self._grid_ladder()[max(0, min(step + delta, len(GRID_ZOOM_STEPS) - 1))]
        self._apply_zoom()

    def _on_grid_max_side(self, _side: int) -> None:
        """Resize magnified cells after a larger crop was decoded."""
        if not self._single_mode:
            self._apply_zoom()

    def _open_single(self, row: int, zoom: int | str | None = None) -> None:
        """Show a crop in the single view at 'zoom' or the remembered one."""
        self._single_index = max(0, min(row, max(len(self._image_rows) - 1, 0)))
        if zoom is not None:
            self._single_zoom = zoom
        self._view_mode = "single"
        self._apply_zoom()

    def _close_single(self) -> None:
        """Return to the grid at the crop shown last."""
        self._stop_playback()
        self._view_mode = "grid"
        self._apply_zoom()
        self.grid_view.setCurrentIndex(self.grid_model.index(self._single_index))
        self.grid_view.setFocus()

    def _show_single(self) -> None:
        """Show the current crop in the single-image view."""
        if not self._image_rows:
            self.single_view.set_image(None, "")
            return
        self._single_index = max(0, min(self._single_index, len(self._image_rows) - 1))
        row = self._image_rows[self._single_index]
        self.single_view.set_image(row["path"] or None,
                                   f"{self._single_index + 1} / {len(self._image_rows)}   ·   "
                                   + row["tooltip"].replace("\n", "   ·   "))
        self._show_image_info(row)

    def _single_navigate(self, delta: int) -> None:
        """Show the next or previous crop in the single-image view."""
        if not self._image_rows:
            return
        self._single_index = max(0, min(self._single_index + delta, len(self._image_rows) - 1))
        self._show_single()

    def _sync_single_readout(self, effective: int) -> None:
        """Show the effective single-image magnification on the slider and label."""
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(SINGLE_ZOOM_STEPS.index(effective) if effective in SINGLE_ZOOM_STEPS else 0)
        self.zoom_slider.blockSignals(False)
        text = "1:1" if effective == 1 else f"{effective}×"
        self.zoom_label.setText(f"Crop · fit {text}" if self._single_zoom == FIT else f"Crop · {text}")

    def _on_single_zoom_resolved(self, effective: int) -> None:
        """Update the readout after the single view resolved its magnification."""
        if self._single_mode:
            self._sync_single_readout(effective)

    def _on_grid_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        """Show the details of the crop that became current in the grid."""
        row = self.grid_model.row(current.row())
        if row is not None:
            self._single_index = current.row()
            self._show_image_info(row)

    def _show_image_info(self, row: dict[str, Any]) -> None:
        """Show the details of a crop (choices, confidence, size) below the views."""
        img = row["data"]
        parts = [f"crop {self._single_index + 1} / {len(self._image_rows)}", _fmt_time(img.get("timestamp"))]
        slots = () if self.verdict_panel.blind_hidden else (("top1", "1st choice"), ("top2", "2nd choice"))
        for slot, label in slots:
            name = next((img[c] for c in img
                         if c.startswith(f"bioclip_{slot}_") and not c.endswith("_prob") and img[c]), None)
            if name:
                parts.append(f"{label}: {name} ({_fmt(img.get(f'bioclip_{slot}_prob'))})")
        if img.get("confidence") is not None:
            parts.append(f"detection confidence {_fmt(img.get('confidence'))}")
        if img.get("bbox_length_mm") is not None:
            parts.append(f"size ≈ {_fmt(img.get('bbox_length_mm'), 1)} × {_fmt(img.get('bbox_width_mm'), 1)} mm")
        self.image_info_label.setText("   ·   ".join(parts))

    @Slot()
    def toggle_playback(self) -> None:
        """Start or pause the playback of the track's crops (shortcut)."""
        self.play_btn.toggle()

    def _on_play_toggled(self, checked: bool) -> None:
        """Start the playback in the single view, or stop it."""
        if checked:
            if len(self._image_rows) < 2:
                self.play_btn.setChecked(False)
                return
            if self.image_stack.currentWidget() is not self.single_view:
                self._open_single(self._single_index)
            if self._single_index >= len(self._image_rows) - 1:
                # On the last crop: play from the beginning
                self._single_index = 0
                self._show_single()
            self._play_timer.start(int(self.play_interval.currentData() or VALIDATION_PLAYBACK_INTERVAL_DEFAULT_MS))
            self.play_btn.setText("❚❚ Pause  (Space)")
        else:
            self._play_timer.stop()
            self.play_btn.setText("▶ Play  (Space)")

    def _stop_playback(self) -> None:
        """Stop the playback."""
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)

    def _play_step(self) -> None:
        """Show the next crop and stop at the end of the track."""
        if not self._image_rows or self.image_stack.currentWidget() is not self.single_view:
            self._stop_playback()
            return
        if self._single_index + 1 >= len(self._image_rows):
            # One pass only: rewind to the first crop and stop
            self._single_index = 0
            self._show_single()
            self._stop_playback()
            return
        self._single_index += 1
        self._show_single()

    def _on_play_interval_changed(self, _index: int) -> None:
        """Apply and remember a changed playback interval."""
        ms = int(self.play_interval.currentData() or VALIDATION_PLAYBACK_INTERVAL_DEFAULT_MS)
        _settings().setValue(_SETTINGS_KEY_PLAYBACK_MS, ms)
        if self._play_timer.isActive():
            self._play_timer.start(ms)

    def _on_compare_toggled(self, checked: bool) -> None:
        """Show or hide the comparison strip."""
        _settings().setValue(_SETTINGS_KEY_COMPARE, checked)
        self.compare_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.compare_view.setVisible(checked and not self.verdict_panel.blind_hidden)
        if checked and self._current is not None and self.run is not None:
            self._populate_comparison(self._current, self.run.track_row(self._current) or {})

    def _populate_comparison(self, key: TrackKey, row: dict[str, Any]) -> None:
        """Fill the comparison strip with one crop per similar track."""
        self._compare_keys = []
        if (
            self.run is None
            or self.session is None
            or not self.compare_toggle.isChecked()
            or self.verdict_panel.blind_hidden
        ):
            self.compare_model.set_images([])
            return
        run, session = self.run, self.session
        rank = run.model_rank
        col = f"bioclip_{rank}"
        queue = self._key_set
        items: list[dict[str, Any]] = []
        shown: set[TrackKey] = {key}

        def revealed(other_key: TrackKey) -> bool:
            """Blind: the prediction of a queue track stays hidden until it is determined."""
            if not session.blind or other_key not in queue:
                return True
            entry = session.verdict(other_key)
            return entry is not None and entry.validated

        def add_same_taxon(taxon: str | None) -> int:
            """Tracks predicted as 'taxon' with the highest probability."""
            if not taxon:
                return 0
            added = 0
            for other_key, prob in run.tracks_predicted_as(rank, taxon):
                if other_key in shown or not revealed(other_key):
                    continue
                path = run.best_crop(other_key, taxon)
                if path is None:
                    continue
                shown.add(other_key)
                in_queue = other_key in queue
                items.append({
                    "path": str(path), "label": f"ID {other_key[2]}",
                    "tooltip": f"{taxon}\nTrack {other_key[2]} · {_device_name(other_key[0], run.device_numbers)} · "
                               f"session {other_key[1]}\n"
                               f"weighted probability {_fmt(prob)}"
                               + ("" if in_queue else "\n(not in your validation queue)"),
                    "data": {},
                })
                self._compare_keys.append(other_key if in_queue else None)
                added += 1
                if added >= _COMPARE_SAME_TAXON_MAX:
                    break
            return added

        # Show the individual's other tracks first, then link candidates
        strip_rows: list[tuple[TrackKey, str, str, str]] = []
        for member in self._link_group:
            if member.get("is_current"):
                continue
            span = f"{_fmt_time(member.get('start_time'))[11:]}–{_fmt_time(member.get('end_time'))[11:]}"
            strip_rows.append((member["key"], f"{LINK_GLYPH_NEXT} ID {member['track_id']}",
                               f"Same individual (track {member['track_id']}, {span})", "individual"))
        group_ids = {str(m["track_id"]) for m in self._link_group}
        for cand in self._link_candidates:
            if str(cand["track_id"]) in group_ids:
                continue
            when = "overlap" if cand["gap_s"] < 0 else "before"
            strip_rows.append((cand["key"], f"{LINK_GLYPH_NEXT}? ID {cand['track_id']}",
                               f"Earlier track ({abs(cand['gap_s']):.1f} s {when}) - same individual candidate",
                               "candidate"))
        counts = {"individual": 0, "candidate": 0}
        for other_key, label, headline, kind in strip_rows:
            if other_key in shown:
                continue
            images = run.images_for(other_key)
            path = run.best_crop(other_key)
            if path is None:
                continue
            shown.add(other_key)
            in_queue = other_key in queue
            items.append({
                "path": str(path), "label": label,
                "tooltip": f"{headline}\nTrack {other_key[2]} · {_device_name(other_key[0], run.device_numbers)} · "
                           f"session {other_key[1]} · "
                           f"{images.height} crops" + ("" if in_queue else "\n(not in your validation queue)"),
                "data": {},
            })
            self._compare_keys.append(other_key if in_queue else None)
            counts[kind] += 1

        # Show other tracks with this prediction
        taxon = row.get(col)
        available = sum(1 for other_key, _ in run.tracks_predicted_as(rank, taxon)
                        if other_key not in shown and revealed(other_key)) if taxon else 0
        n_same = add_same_taxon(taxon)
        self.compare_model.set_images(items)
        parts = []
        if counts["individual"]:
            parts.append(f"{counts['individual']} of this individual")
        if counts["candidate"]:
            parts.append(f"{counts['candidate']} link candidate{'' if counts['candidate'] == 1 else 's'}")
        if n_same:
            parts.append(f"{n_same} most confident of {available} tracks predicted as <i>{taxon}</i>"
                         if n_same < available else
                         f"all {n_same} other track{'' if n_same == 1 else 's'} predicted as <i>{taxon}</i>")
        text = " · ".join(parts) if parts else "no similar tracks in this run"
        self.compare_label.setText(text)
        self.compare_label.setToolTip(text)

    def _on_compare_open(self, index: int) -> None:
        """Show the track of a double-clicked comparison crop (if in the queue)."""
        if 0 <= index < len(self._compare_keys):
            key = self._compare_keys[index]
            if key is not None:
                self._show_track(key)

    def _after_one_click(self) -> None:
        """Go to the next unvalidated track after a one-click verdict (auto-advance)."""
        if self.auto_advance_cb.isChecked():
            QTimer.singleShot(_AUTO_ADVANCE_DELAY_MS, self.next_unvalidated)

    def _on_verdict_changed(self, verdict: Verdict) -> None:
        """Store an edited verdict or pending entry. Refresh links, strip and progress."""
        if self.session is None or self._current is None:
            return
        self._flush_time()
        linked_before = self.session.group_of(self._current)
        if verdict.validated:
            self.session.set_verdict(self._current, verdict)
        else:
            existing = self.session.verdict(self._current)
            if existing is not None and existing.validated:
                self.session.reset(self._current)  # verdict withdrawn, only notes/flags remain
            self.session.set_pending(self._current, verdict)
        self._refresh_individual(self._current, linked_before)
        self._link_candidates, self._link_group = self._link_context(self._current)
        self.verdict_panel.set_link_context(self._link_candidates, self._link_group,
                                            order_hint=self.sort_combo.currentData() not in TrackListModel.CHRONO_SORTS)
        if self.run is not None:
            self._populate_comparison(self._current, self.run.track_row(self._current) or {})
        self._sync_blind_visibility()
        self._sync_batch()
        self._update_progress()
        self._save_timer.start()

    def _on_notes_changed(self, notes: str) -> None:
        """Store the note of the shown track."""
        if self.session is None or self._current is None:
            return
        self.session.set_note(self._current, notes)
        self._save_timer.start()

    def _on_custom_taxon(self, rank: str, lineage: Lineage) -> None:
        """Remember a free-text lineage in the session."""
        if self.session is not None:
            self.session.add_custom_taxon(rank, lineage)

    def _on_link_jump(self, track_id: str) -> None:
        """Show a track of the individual (link in the individual line)."""
        if self._current is None:
            return
        key: TrackKey = (self._current[0], self._current[1], track_id)
        if key in self._key_set:
            self._show_track(key)

    def _reset_current(self) -> None:
        """Discard the verdict of the shown track, or of all selected tracks."""
        if self.session is None or self._current is None:
            return
        keys = self._selected_keys()
        targets = keys if len(keys) > 1 and self._current in keys else [self._current]
        if len(targets) > 1 and not self._confirm_batch(len(targets), "reset"):
            return
        for key in targets:
            linked_before = self.session.group_of(key)
            self.session.reset(key)
            self._refresh_individual(key, linked_before)
        self._refresh_shown_track()
        self._sync_batch()
        self._update_progress()
        self._save_timer.start()

    def _refresh_shown_track(self) -> None:
        """Reload the form, links and comparison strip of the shown track after a change to the selection.

        Unlike _show_track(), crops, view and playback stay as they are.
        """
        if self.run is None or self.session is None or self._current is None:
            return
        key = self._current
        self._link_candidates, self._link_group = self._link_context(key)
        self.verdict_panel.set_link_context(self._link_candidates, self._link_group,
                                            order_hint=self.sort_combo.currentData() not in TrackListModel.CHRONO_SORTS)
        self.verdict_panel.load(self.session.model_lineage(key), self.session.verdict(key), self.session.model_rank,
                                self.session.recent_lineages())
        self._sync_blind_visibility()
        self._populate_comparison(key, self.run.track_row(key) or {})

    def _refresh_individual(self, key: TrackKey, before: list[TrackKey]) -> None:
        """Repaint the rows of a track and its individual."""
        for other in dict.fromkeys([key, *before, *self.session.group_of(key)] if self.session else [key]):
            self.track_model.refresh_key(other)

    def _batch_targets(self) -> list[TrackKey]:
        """Return the selected tracks other than the current one."""
        keys = self._selected_keys()
        if len(keys) < 2 or self._current not in keys:
            return []
        return [k for k in keys if k != self._current]

    def _confirm_batch(self, n: int, what: str, others: bool = False) -> bool:
        """Return whether 'what' may be applied to 'n' tracks (asks for large batches).

        Args:
            n: Tracks to change.
            what: Name of the action.
            others: The shown track already has it, 'n' counts the other selected tracks.
        """
        if n + others <= _BATCH_CONFIRM_THRESHOLD:
            return True
        question = (f"Also apply '{what}' to the other {n} selected tracks?" if others
                    else f"Apply '{what}' to all {n} selected tracks?")
        reply = QMessageBox.question(self, "Batch verdict", question,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    def _on_batch_action(self, action: str, payload: Any) -> None:
        """Apply a one-click action of the verdict form to the other selected tracks."""
        targets = self._batch_targets()
        if not targets or self.session is None:
            return
        if action == "same_individual":
            self._batch_link([*targets, self._current], bool(payload["checked"]))  # type: ignore[list-item]
            return
        if not self._confirm_batch(len(targets), action.replace("_", " "), others=True):
            return
        for key in targets:
            existing = self.session.verdict(key)
            verdict = self.verdict_panel.batch_verdict(action, payload, self.session.model_lineage(key), existing)
            if verdict is None:
                continue
            linked_before = self.session.group_of(key)  # a false positive leaves its individual
            if verdict.validated:
                self.session.set_verdict(key, verdict)
            else:
                if existing is not None and existing.validated:
                    self.session.reset(key)
                self.session.set_pending(key, verdict)
            self._refresh_individual(key, linked_before)
        self._sync_batch()
        self._update_progress()
        self._save_timer.start()
        self.save_label.setText(f"Applied to {len(targets) + 1} tracks")

    def _batch_link(self, keys: list[TrackKey], link: bool) -> None:
        """Link the selected tracks as one individual per device/session, or remove their links."""
        if self.session is None or self.run is None:
            return
        if not self._confirm_batch(len(keys), "same individual"):
            self._sync_batch()  # the button shows the selection's state again
            return
        session = self.session
        links: dict[TrackKey, str | None] = {}
        if link:
            for later, earlier in self._batch_link_pairs(keys):
                if later not in links:
                    entry = session.verdict(later)
                    links[later] = entry.same_individual_as if entry is not None else None
                links[later] = with_link(links[later], earlier[2], True)
        else:
            links = {key: None for key in keys if not self._link_excluded(key) and session.predecessors_of(key)}
        changed = 0
        for key, value in links.items():
            existing = session.verdict(key)
            if existing is not None and existing.same_individual_as == value:
                continue
            linked_before = session.group_of(key)
            base = existing.copy() if existing is not None else Verdict(validated=False)
            base.same_individual_as = value
            if base.validated:
                session.set_verdict(key, base)
            else:
                session.set_pending(key, base)
            changed += 1
            self._refresh_individual(key, linked_before)
        self._refresh_shown_track()
        self._sync_batch()
        self._update_progress()
        self._save_timer.start()
        self.save_label.setText(f"{'Linked' if link else 'Unlinked'} {changed} tracks")

    def _batch_link_pairs(self, keys: list[TrackKey]) -> list[tuple[TrackKey, TrackKey]]:
        """Return the '(later, earlier)' links of a batch: consecutive selected tracks of a device/session in time
        order that are link candidates of each other (see ValidationRun.is_link_candidate()).

        False positives are left out, so the chain bridges them.
        """
        if self.run is None:
            return []
        run = self.run
        groups: dict[tuple[str, str], list[TrackKey]] = {}
        for key in keys:
            if not self._link_excluded(key):
                groups.setdefault((key[0], key[1]), []).append(key)
        pairs = []
        for members in groups.values():
            ordered = sort_tracks_chronologically(members, run.track_start_times)
            pairs += [(later, earlier) for earlier, later in pairwise(ordered) if run.is_link_candidate(earlier, later)]
        return pairs

    def _batch_link_state(self) -> tuple[bool, bool, str] | None:
        """Return '(enabled, checked, tooltip)' of the 'Same individual' button for the selection, None without one.

        Checked (pressing unlinks) once all links of the selection are set, or if only stored links remain.
        """
        if self.session is None or not self._batch_targets():
            return None
        keys = self._selected_keys()
        pairs = self._batch_link_pairs(keys)
        if any(earlier[2] not in self.session.predecessors_of(later) for later, earlier in pairs):
            return True, False, _LINK_BTN_BATCH_TOOLTIPS["link"]
        if any(not self._link_excluded(key) and self.session.predecessors_of(key) for key in keys):
            return True, True, _LINK_BTN_BATCH_TOOLTIPS["unlink"]
        return False, False, _LINK_BTN_BATCH_TOOLTIPS["none"]

    def _link_excluded(self, key: TrackKey) -> bool:
        """Return whether a track cannot be linked (a false positive)."""
        entry = self.session.verdict(key) if self.session is not None else None
        return entry is not None and entry.false_positive

    def _apply_determination_to_selection(self) -> None:
        """Copy the shown track's determination to the other selected tracks."""
        targets = self._batch_targets()
        if not targets or self.session is None:
            return
        current = self.verdict_panel.current_verdict()
        if not current.validated and not current.determinate_ranks():
            QMessageBox.information(self, "Batch determination",
                                    "Determine the shown track first, then copy it to the selection.")
            return
        if not self._confirm_batch(len(targets), "copy determination"):
            return
        confirmed = not current.validated
        if confirmed:
            current = self.verdict_panel.confirm_current()
        for key in targets:
            verdict = self.verdict_panel.batch_verdict("apply_determination", current, self.session.model_lineage(key),
                                                       self.session.verdict(key))
            if verdict is None:
                continue
            linked_before = self.session.group_of(key)  # a copied false positive leaves its individual
            self.session.set_verdict(key, verdict)
            self._refresh_individual(key, linked_before)
        self._sync_batch()
        self._update_progress()
        self._save_timer.start()
        self.save_label.setText(f"Determination copied to {len(targets)} tracks"
                                + (" (shown track confirmed)" if confirmed else ""))

    def _step(self, delta: int) -> None:
        """Show the next or previous track of the list."""
        keys = self._visible_keys() or self._keys
        if not keys:
            return
        if self._current in keys:
            pos = keys.index(self._current) + delta
        else:
            pos = 0
        pos = max(0, min(pos, len(keys) - 1))
        self._show_track(keys[pos])

    @Slot()
    def next_unvalidated(self) -> None:
        """Show the next track without a verdict, from the shown one on."""
        if self.session is None:
            return
        keys = self._visible_keys() or self._keys
        if not keys:
            return
        start = keys.index(self._current) + 1 if self._current in keys else 0
        order = keys[start:] + keys[:start]
        for key in order:
            verdict = self.session.verdict(key)
            if verdict is None or not verdict.validated:
                self._show_track(key)
                return
        self.save_label.setText("All tracks validated ✓")

    def _on_return_key(self) -> None:
        """Commit a changed entry on Enter, else go to the next track."""
        focus = QApplication.focusWidget()
        # An editable combo box holds the focus for its line edit
        combo = focus if isinstance(focus, TaxonCombo) else focus.parent() if isinstance(focus, QLineEdit) else None
        if isinstance(combo, TaxonCombo):
            if not combo.commit_current():
                self.next_unvalidated()
            return
        if isinstance(focus, QLineEdit) and focus is not self.verdict_panel.notes_edit:
            focus.editingFinished.emit()
            return
        self.next_unvalidated()

    def shutdown(self) -> None:
        """Stop playback and thumbnails, save and close the session log (window closes)."""
        self._stop_playback()
        self._save_timer.stop()
        self.save_now()
        self._cache.cancel_pending()
        self._close_session_log()

    def _close_session_log(self) -> None:
        """Log the end of the sitting and detach the session's log file."""
        if self._session_log is None:
            return
        if self.session is not None:
            validated, total = self.session.progress()
            logger.info("Validation session %s closed: %d/%d tracks validated", self.session.session_id,
                        validated, total)
        close_session_log(self._session_log)
        self._session_log = None


class StatsDialog(QDialog):
    """Dialog with the session's verdicts, model agreement and accuracy as tables and charts, with CSV export."""

    def __init__(self, session: ValidationSession, parent: QWidget | None = None) -> None:
        """Initialize the statistics window.

        Args:
            session: Session to evaluate.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.session = session
        self.frame = session.to_frame()
        self.model_rank = session.model_rank
        self.tables: dict[str, QTableWidget] = {}
        self._tab_tables: dict[QWidget, list[str]] = {}
        self._column_widths: dict[str, list[int]] = {}
        self._caveat_text = ""
        self.stratified = is_stratified(self.frame)
        self.other_sessions = [info for info in ValidationSession.list_sessions(session.run_root)
                               if Path(info["csv_path"]) != session.csv_path]
        others = []
        for info in self.other_sessions:
            try:
                other = ValidationSession.load(Path(info["csv_path"]))
            except (OSError, ValueError):
                continue
            others.append((other.validator, other.session_id, other.to_frame()))
        self.session_comparison = compare_sessions(self.frame, session.validator, others, self.model_rank)
        sampling = session.meta["sampling"]
        self.prob_edges = list(sampling.get("prob_edges") or VALIDATION_PROB_BIN_EDGES_DEFAULT)
        self.setWindowTitle(f"Validation statistics - {session.validator} ({session.session_id})")
        self.resize(1000, 700)
        self._create_ui()
        self._refresh()

    def _create_ui(self) -> None:
        """Build the header controls, the tabs and the buttons."""
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.header_label = QLabel("")
        self.header_label.setWordWrap(True)
        header.addWidget(self.header_label, 1)
        self.low_quality_cb = QCheckBox("Include not-assessable tracks")
        self.low_quality_cb.setToolTip("Include tracks flagged as low image quality or as showing different taxa "
                                       "(in the crops or along the track) in the accuracy statistics")
        self.low_quality_cb.toggled.connect(lambda _: self._refresh())
        header.addWidget(self.low_quality_cb)
        self.uncertain_cb = QCheckBox("Include uncertain (inc.)")
        self.uncertain_cb.setChecked(True)
        self.uncertain_cb.setToolTip("Include determinations the validator marked as uncertain (inc.)")
        self.uncertain_cb.toggled.connect(lambda _: self._refresh())
        header.addWidget(self.uncertain_cb)
        header.addWidget(QLabel("Rank:"))
        self.rank_combo = QComboBox()
        for rank in ranks_up_to(self.model_rank):
            self.rank_combo.addItem(_RANK_LABELS[rank], rank)
        default_rank = "family" if "family" in ranks_up_to(self.model_rank) else self.model_rank
        self.rank_combo.setCurrentIndex(max(self.rank_combo.findData(default_rank), 0))
        self.rank_combo.setToolTip("Rank of the probability-bin, per-taxon and confusion tables "
                                   "(and of the exported effort table and confusion matrix)")
        self.rank_combo.currentIndexChanged.connect(lambda _: self._refresh())
        header.addWidget(self.rank_combo)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(lambda _: self._fit_current())
        layout.addWidget(self.tabs, 1)
        self.chart_container = QWidget()
        self._chart_layout = QVBoxLayout(self.chart_container)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        # The chart views stay, a refresh replaces their charts and the notes below them
        charts_row = QHBoxLayout()
        self._chart_views = [QChartView(), QChartView()]
        for view in self._chart_views:
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            charts_row.addWidget(view, 1)
        self._chart_layout.addLayout(charts_row)
        self.tabs.addTab(self.chart_container, "Overview")
        self._add_tab("Verdicts", ["verdicts"])
        if self.stratified:
            self._add_tab("Sample", ["strata"])
        self._add_tab("Agreement by rank", ["agreement"], caveats=True)
        self._add_tab("Accuracy by probability bin", ["reliability"])
        self._add_tab("Per predicted taxon", ["taxa"])
        self._add_tab("Confusions", ["confusions"])
        self._add_tab("Data checks", ["linked"])
        if self.other_sessions:
            self._add_tab("Other sessions", ["validators"])
        self.tabs.setCurrentIndex(0)

        buttons = QDialogButtonBox()
        self.export_btn = buttons.addButton("Export statistics (CSV)", QDialogButtonBox.ButtonRole.ActionRole)
        self.export_btn.clicked.connect(self._export)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_tab(self, title: str, names: list[str], headings: dict[str, str] | None = None,
                 caveats: bool = False) -> None:
        """Add a tab holding one table per name, each optionally under a heading."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        if caveats:
            self.caveats_label = QLabel("")
            self.caveats_label.setWordWrap(True)
            page_layout.addWidget(self.caveats_label)
        for name in names:
            if headings and name in headings:
                page_layout.addWidget(QLabel(f"<b>{headings[name]}</b>"))
            table = QTableWidget()
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            table.horizontalHeader().setStretchLastSection(False)
            self.tables[name] = table
            page_layout.addWidget(table, 1)
        self._tab_tables[page] = names
        self.tabs.addTab(page, title)

    def _refresh(self) -> None:
        """Recompute every table, the header, the caveats and the charts for the current settings."""
        include_low = self.low_quality_cb.isChecked()
        include_inc = self.uncertain_cb.isChecked()
        rank = self.rank_combo.currentData() or self.model_rank
        frame = self.frame
        validated, total = self.session.progress()
        low = n_low_quality(frame)
        in_track = n_different_taxa_in_track(frame)
        in_crops = n_different_taxa_in_crops(frame)
        n_inc = n_uncertain(frame)
        n_ind = n_individuals(frame)
        n_detected = self._n_detected()
        effort = effort_summary(frame).to_dicts()[0]
        median = effort["median_seconds"]
        self.header_label.setText(
            f"<b>{validated}</b> of {total} sampled tracks validated"
            + (f" ({n_detected} arthropod tracks of {n_ind} individuals)" if n_ind != n_detected else "")
            + f" · {n_inc} uncertain (inc.){' (excluded)' if not include_inc and n_inc else ''}"
            f" · {in_crops} different taxa in crops · {in_track} different taxa in track · {low} low image quality"
            f"{' (excluded from accuracies)' if not include_low and (low or in_crops or in_track) else ''}"
            f" · {self._population_text()}"
            f" · model rank <b>{self.model_rank}</b>"
            + (" · <b>blind</b> session" if self.session.blind else "")
            + f" · effort {effort['total_minutes']} min"
            + (f" (median {median:.0f} s per track)" if median is not None else "")
        )
        agreement = agreement_by_rank(frame, self.model_rank, include_low, include_inc)
        self._fill("agreement", agreement)
        self.tables["agreement"].setToolTip(
            "Accuracy: share of tracks where the validator confirmed the model's taxon at that rank"
        )
        self._caveat_text = " ".join(f"⚠ {note}" for note in self._caveats(validated, agreement))
        self.caveats_label.setText(self._caveat_text)
        self.caveats_label.setVisible(bool(self._caveat_text))
        self._fill("verdicts", self._verdict_rows())
        if self.stratified:
            self._fill("strata", self._sample_rows())
        conflicts = linked_track_conflicts(frame)
        linked_col = frame.filter(pl.col("validated"))["same_individual_as"]
        n_linked = int(linked_col.is_not_null().sum())
        self._fill("linked", conflicts, empty_text=(
            f"No contradictions: the determinations of all {n_linked} linked tracks are consistent." if n_linked
            else "No validated track is linked as the same individual as another one ('Same individual', L)."
        ))
        self.tables["linked"].setToolTip(
            f"{n_linked} validated tracks linked as the same individual as another track · "
            f"{conflicts.height} pairs with contradicting determinations"
        )
        self._fill("reliability", reliability_by_prob_bin(frame, rank, self.prob_edges, include_low, include_inc))
        taxa_cols = ["taxon", "n_validated", "n_correct", "n_wrong", "n_false_positive", "precision",
                     "precision_ci_low", "precision_ci_high"]
        taxa_cols += (["precision_weighted", "precision_weighted_ci_low", "precision_weighted_ci_high"]
                      if self.stratified else ["recall", "f1"])
        self._fill("taxa", per_taxon_table(frame, rank, include_low, include_inc).select(taxa_cols),
                   empty_text="No validated tracks yet.")
        n_comparable = agreement.filter(pl.col("rank") == rank)["n_comparable"].sum()
        self._fill("confusions", confusion_pairs(frame, rank, include_not_assessable=include_low,
                                                 include_uncertain=include_inc), empty_text=(
            f"No disagreements: the validator confirmed the model's {rank} on all {n_comparable} comparable tracks."
            if n_comparable else f"No comparable tracks at {rank} rank yet (see 'Agreement by rank')."
        ))
        if self.other_sessions:
            self._fill_validators()
        self._render_overview_charts(agreement, include_low, include_inc)

    def _population_text(self) -> str:
        """Return the population the weighted results refer to (the sampled strata of the selection)."""
        meta = self.session.meta
        total = meta["n_population_total"]
        notes = [] if (selection := _fmt_selection(meta["sampling"])) == "all tracks of the run" else [selection]
        if excluded := meta["strata_excluded"]:
            notes.append(f"without {len(excluded)} strata below the minimum")
        n_population = sum(meta["strata_population"].values())
        if not notes:
            return f"population {total} tracks"
        return f"population {n_population} of {total} tracks ({'; '.join(notes)})"

    def _caveats(self, validated: int, agreement: pl.DataFrame) -> list[str]:
        """Return the caveats of the session's accuracies."""
        notes = []
        individuals, detected = n_individuals(self.frame), self._n_detected()
        if individuals and individuals != detected:
            notes.append(f"{detected} validated arthropod tracks are {individuals} individuals: linked tracks are "
                         "not independent observations, so the confidence intervals are somewhat too narrow.")
        if self.stratified:
            notes.append("Stratified sample: the plain shares describe the sample, the '_weighted' columns "
                         "estimate all tracks of the sampled strata, and recall is not estimable.")
            unvalidated = unvalidated_strata(self.frame)
            if validated and unvalidated.height:
                n_strata = self.frame["stratum_id"].n_unique()
                notes.append(f"{unvalidated.height} of {n_strata} sampled strata ({int(unvalidated['n_population'].sum())} "
                             "tracks) have no validated track yet: the weighted estimates cover only the other strata.")
        if validated < self.frame.height and not validated_in_queue_order(self.frame):
            notes.append(f"Tracks were not validated in sampling order (sort '{_SORT_LABELS['queue']}'): until "
                         "every sampled track is validated, the estimates assume that the validated tracks are a "
                         "random part of the sample.")
        comparable = [n for n in agreement["n_comparable"].to_list() if n]
        if comparable and max(comparable) < VALIDATION_SMALL_SAMPLE_TRACKS:
            notes.append(f"Fewer than {VALIDATION_SMALL_SAMPLE_TRACKS} comparable tracks at every rank: "
                         "every share still moves a lot with each verdict.")
        return notes

    def _n_detected(self) -> int:
        """Return the number of validated tracks that are not false positives."""
        return self.frame.filter(pl.col("validated") & (pl.col("verdict") != "false_positive")).height

    def _verdict_rows(self) -> pl.DataFrame:
        """Return the verdict distribution with one labeled row per category, the validated tracks first."""
        table = verdict_table(self.frame)
        if self.stratified and "share_weighted" not in table.columns:  # no validated track yet
            table = table.with_columns(pl.lit(None, pl.Float64).alias("share_weighted"))
        validated = pl.DataFrame({"category": ["Validated tracks"], "n_tracks": [table["n_validated"][0]]},
                                 schema={"category": pl.String, "n_tracks": pl.Int64})
        rows = table.drop("n_validated").with_columns(pl.col("category").replace_strict(_CATEGORY_LABELS))
        return pl.concat([validated, rows], how="diagonal")

    def _sample_rows(self) -> pl.DataFrame:
        """Return the population, sample and verdicts per stratum."""
        progress = progress_by_stratum(self.frame)
        verdicts = verdict_distribution(self.frame, by_stratum=True)
        keep = [c for c in ("stratum_id", "n_agree", "n_partial", "n_corrected", "n_false_positive",
                            "share_agree") if c in verdicts.columns]
        return progress.join(verdicts.select(keep), on="stratum_id", how="left", maintain_order="left")

    def _fill(self, name: str, df: pl.DataFrame, digits: int = 3, empty_text: str = "") -> None:
        """Fill a table from a frame (header tooltips from _STATS_TABLE_TIPS, else column_description()).

        Args:
            name: Key of the table.
            df: Rows to show.
            digits: Decimals of the float values.
            empty_text: Shown across the columns when 'df' has no rows.
        """
        table = self.tables[name]
        self._column_widths.pop(name, None)
        table.clear()
        table.clearSpans()
        table.setRowCount(df.height)
        table.setColumnCount(df.width)
        table.setHorizontalHeaderLabels(df.columns)
        overrides = _STATS_TABLE_TIPS.get(name, {})
        for j, column in enumerate(df.columns):
            tip = overrides.get(column) or column_description(_STATS_TABLE_EXPORTS.get(name, name), column)
            if tip:
                _header_item(table, j).setToolTip(tip)
        for i, row in enumerate(df.iter_rows()):
            for j, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.{digits}f}"
                elif value is None:
                    text = "-"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(i, j, item)
        if df.is_empty() and empty_text:
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem(empty_text))
            table.setVerticalHeaderItem(0, QTableWidgetItem(""))  # a message, not row 1
            if df.width > 1:
                table.setSpan(0, 0, 1, df.width)
        self._fit_columns(name)

    def _fill_validators(self) -> None:
        """Fill the comparison with the other sessions."""
        df = self.session_comparison
        if df.is_empty():
            df = pl.DataFrame({"info": ["The other sessions of this run could not be read."]})
        self._fill("validators", df)
        self.tables["validators"].setToolTip(
            "Agreement with the other sessions of this run on the tracks both validated. A session of the same "
            "validator is a test-retest comparison (their own consistency), not inter-rater agreement."
        )

    def _render_overview_charts(self, agreement: pl.DataFrame, include_low: bool, include_inc: bool) -> None:
        """Draw the overview charts of accuracy per rank and verdicts (weighted when stratified)."""
        layout = self._chart_layout
        while layout.count() > 1:  # the notes below the charts row
            item = layout.takeAt(1)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()  # deleted on return to the event loop, not painted until then
                widget.deleteLater()
        text_color = QColor(_theme_color("text", "#E0E0E0"))
        weighted = self.stratified

        # Accuracy per rank: the stratum-weighted one for a stratified sample
        rank_rows = [r for r in agreement.to_dicts() if r["n_comparable"]]
        if weighted:
            rank_rows = [r for r in rank_rows if r["accuracy_weighted"] is not None]
        settings = [text for text, active in (("incl. not-assessable", include_low),
                                              ("excl. uncertain", not include_inc)) if active]
        accuracy_title = (("Stratum-weighted accuracy per rank" if weighted else "Accuracy per rank (human = model)")
                          + (f" ({', '.join(settings)})" if settings else ""))
        agree_bars = QBarSet("agree")
        agree_bars.setColor(QColor(COLOR_VERDICT["agree"]))
        disagree_bars = QBarSet("disagree")
        disagree_bars.setColor(QColor(COLOR_VERDICT["corrected"]))
        ranks = []
        details = []
        for r in rank_rows:
            if weighted:  # shares of 1000, the series shows percentages
                agree_bars.append(round(1000 * r["accuracy_weighted"]))
                disagree_bars.append(1000 - round(1000 * r["accuracy_weighted"]))
            else:
                agree_bars.append(r["n_agree"])
                disagree_bars.append(r["n_comparable"] - r["n_agree"])
            ci = (f" [{r['accuracy_ci_low']:.2f}-{r['accuracy_ci_high']:.2f}]"
                  if r["accuracy_ci_low"] is not None else "")
            ranks.append(r["rank"])
            details.append(f"<b>{r['rank']}</b> {r['accuracy']:.2f}{ci} n={r['n_comparable']}")

        # Verdict distribution: the stratum-weighted shares for a stratified sample
        distribution = verdict_distribution(self.frame).to_dicts()
        verdict_bars = []
        for verdict_type in VERDICT_TYPES:
            bars = QBarSet(VERDICT_LABELS[verdict_type])
            bars.setColor(QColor(COLOR_VERDICT[verdict_type]))
            if distribution and weighted:  # shares of 1000, the series shows percentages
                bars.append(round(1000 * (distribution[0].get(f"share_{verdict_type}_weighted") or 0)))
            else:
                bars.append(distribution[0][f"n_{verdict_type}"] if distribution else 0)
            verdict_bars.append(bars)
        n_validated = distribution[0]["n_validated"] if distribution else 0

        charts = (
            _percent_bar_chart(accuracy_title, [agree_bars, disagree_bars], ranks or ["-"], text_color),
            _percent_bar_chart("Verdicts (stratum-weighted)" if weighted else "Verdicts", verdict_bars,
                               [f"validated n={n_validated}"], text_color),
        )
        for view, chart in zip(self._chart_views, charts, strict=True):
            previous = view.chart()
            view.setChart(chart)  # releases the previous chart without deleting it
            previous.deleteLater()
        details_label = QLabel(("Sample accuracy" if self.stratified else "Accuracy") + " [Wilson 95 % CI]: "
                               + " · ".join(details) if details else "No comparable tracks yet.")
        details_label.setWordWrap(True)
        layout.addWidget(details_label)

        weighted_rows = [r for r in agreement.to_dicts() if r.get("accuracy_weighted") is not None]
        if weighted_rows and self.stratified:
            note = " · ".join(
                f"{r['rank']}: {r['accuracy_weighted']:.2f}"
                + (f" [{r['accuracy_weighted_ci_low']:.2f}-{r['accuracy_weighted_ci_high']:.2f}]"
                   if r["accuracy_weighted_ci_low"] is not None else "")
                for r in weighted_rows
            )
            n_weighted = int(sum(self.session.meta["strata_population"].values()))
            n_run = int(self.session.meta["n_population_total"])
            n_selected = int(self.session.meta["n_population_selected"])
            why = ("the selected tracks" if n_weighted == n_selected else
                   "the sampled strata" if n_selected == n_run else "the sampled strata of the selection")
            scope = (f" over {n_weighted} of the {n_run} tracks of the run ({why})"
                     if n_run and n_weighted != n_run else "")
            layout.insertWidget(layout.indexOf(details_label),
                                QLabel(f"Stratum-weighted accuracy [95 % CI]{scope}: {note}"))
        if self._caveat_text:
            caveats = QLabel(self._caveat_text)
            caveats.setWordWrap(True)
            layout.addWidget(caveats)

    def _fit_columns(self, name: str) -> None:
        """Size columns to their content (measured once per fill) and share the spare width."""
        if not self.isVisible():
            return  # fitted in showEvent
        table = self.tables[name]
        header = table.horizontalHeader()
        if not header.count():
            return
        widths = self._column_widths.get(name)
        if widths is None or len(widths) != header.count():
            table.resizeColumnsToContents()
            widths = [header.sectionSize(i) for i in range(header.count())]
            self._column_widths[name] = widths
        extra = max(0, table.viewport().width() - sum(widths)) // header.count()
        for i, width in enumerate(widths):
            header.resizeSection(i, width + extra)

    def _fit_current(self) -> None:
        """Fit the columns of the visible tab's tables."""
        if not self.tables:
            return
        try:
            for name in self._tab_tables.get(self.tabs.currentWidget(), []):
                self._fit_columns(name)
        except RuntimeError:
            self.tables = {}

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._fit_current()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.isVisible():  # settled by showEvent
            self._fit_current()

    def _export(self) -> None:
        """Write the statistics CSVs with the dialog's settings."""
        out_dir = self.session.stats_dir
        try:
            meta = self.session.meta
            info = {
                **{key: meta[key] for key in ("validator", "session_id", "app_version", "blind", "run_name",
                                              "model_rank", "n_tracks_classified", "run_filters",
                                              "n_population_total")},
                "selection": _fmt_selection(meta["sampling"], full=True),
                "subsample": _fmt_subsample(meta["sampling"], full=True),
                **{key: meta[key] for key in ("sampling", "n_population_selected", "strata_excluded",
                                              "strata_population")},
                # The population the weighted estimates refer to
                "n_population_sampled_strata": sum(meta["strata_population"].values()),
            }
            paths = export_stats(self.frame, out_dir, self.model_rank, self.prob_edges,
                                 include_not_assessable=self.low_quality_cb.isChecked(),
                                 taxon_rank=self.rank_combo.currentData(),
                                 include_uncertain=self.uncertain_cb.isChecked(),
                                 info=info, other_sessions=self.session_comparison)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Statistics exported",
                                f"{len(paths) - 1} tables (CSV) and summary.json written to:\n{out_dir}")


class ValidationWindow(QMainWindow):
    """Validation window, standalone or opened from the processing GUI. Switches between setup and validation page.

    Signals:
        theme_changed(str): A theme was selected here.
    """
    theme_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None, standalone: bool = False) -> None:
        """Initialize the validation window.

        Args:
            parent: Parent widget.
            standalone: Started with the 'validate' entry point (offers updates).
        """
        super().__init__(parent)
        self.setWindowTitle(_WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        # Size when restored from maximized (the window opens maximized)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.85))
        icon_path = BASE_PATH / "resources" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setup_page = SetupPage(standalone=standalone)
        self.validation_page = ValidationPage()
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.validation_page)
        self.setup_page.session_ready.connect(self._on_session_ready)
        self.setup_page.session_deleted.connect(self.validation_page.forget_session)
        self.validation_page.back_requested.connect(self._on_back_requested)
        self.setup_page.theme_changed.connect(self._on_theme_change)

    def open_run(self, run_root: Path | str | None) -> None:
        """Show the setup page for a run directory (default: the last one), or keep an open session of that run."""
        if self.stack.currentWidget() is self.validation_page:
            open_run = self.validation_page.run
            if run_root is None or (open_run is not None and Path(run_root).resolve() == open_run.root.resolve()):
                return
            self.validation_page.leave_session()  # saves, closes the session log and refreshes the setup page
        self.stack.setCurrentWidget(self.setup_page)
        if run_root is None:
            last = _settings().value(_SETTINGS_KEY_LAST_RUN, "")
            run_root = last if isinstance(last, str) and last and Path(last).is_dir() else None
        if run_root is not None and (str(run_root) != self.setup_page.run_param.value() or self.setup_page.run is None):
            self.setup_page.set_run_root(run_root)

    @Slot(object)
    def _on_session_ready(self, session: ValidationSession) -> None:
        """Open a created or resumed session on the validation page."""
        run, taxonomy = self.setup_page.run, self.setup_page.taxonomy
        if run is None or taxonomy is None:
            return
        self.validation_page.load_session(run, session, taxonomy)
        self.stack.setCurrentWidget(self.validation_page)
        self.setWindowTitle(f"Validation - {run.root.name} - {session.validator} ({session.session_id})")

    @Slot()
    def _on_back_requested(self) -> None:
        """Return to the setup page with a refreshed session list."""
        self.setup_page.refresh_sessions()
        self.stack.setCurrentWidget(self.setup_page)
        self.setWindowTitle(_WINDOW_TITLE)

    @Slot(str)
    def _on_theme_change(self, theme_name: str) -> None:
        """Apply and store a theme selected in the setup page."""
        qt_themes.set_theme(theme_name, style=None)
        self.repolish()
        save_theme(theme_name)
        self.theme_changed.emit(theme_name)

    def set_theme_name(self, theme_name: str) -> None:
        """Follow a theme change made elsewhere."""
        self.setup_page.set_theme_name(theme_name)
        self.repolish()

    def repolish(self) -> None:
        """Re-apply stylesheets and theme colors."""
        self.setup_page.refresh_theme()
        self.validation_page.refresh_theme()
        for widget in self.findChildren(QWidget):
            if widget.styleSheet():
                widget.setStyleSheet(widget.styleSheet())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.setup_page.update_installing:
            QMessageBox.information(self, "Update Running",
                                    "The update is still being installed. Please wait until it is finished.")
            event.ignore()
            return

        self.validation_page.shutdown()
        self.setup_page.shutdown()
        event.accept()


def _settings() -> QSettings:
    """Return the QSettings store of the application."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _fmt(value: Any, digits: int = 2) -> str:
    """Format numbers compactly, '-' for missing values."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _device_name(device_id: str, numbers: dict[str, int]) -> str:
    """Return a device as 'D<n> <device_id>' (as numbered in the run), else its id."""
    number = numbers.get(device_id)
    return f"D{number} {device_id}" if number else device_id


def _fmt_time(value: Any) -> str:
    """Return an ISO timestamp as 'YYYY-MM-DD HH:MM:SS', '-' if missing."""
    if not value:
        return "-"
    text = str(value)
    return text.replace("T", " ")[:19]


def _header_item(table: QTableWidget, col: int) -> QTableWidgetItem:
    """Return the header item of a column labeled with setHorizontalHeaderLabels."""
    item = table.horizontalHeaderItem(col)
    assert item is not None, col
    return item


def _theme_color(attr: str, fallback: str) -> str:
    """Return a color of the active theme as hex, 'fallback' if the theme lacks it."""
    theme = qt_themes.get_theme() or qt_themes.get_theme(THEME_DEFAULT)
    color = getattr(theme, attr, None) if theme else None
    return color.name() if color is not None else fallback


def _error_color() -> str:
    """Return the error text color for the active theme."""
    theme = qt_themes.get_theme() or qt_themes.get_theme(THEME_DEFAULT)
    dark = theme.is_dark_theme() if theme is not None else True
    return COLOR_ERROR_TEXT_DARK if dark else COLOR_ERROR_TEXT_LIGHT


def _fmt_selection(sampling: dict[str, Any], full: bool = False) -> str:
    """Return the track selection of a session (taxa and setting filters)."""
    def listed(values: Sequence[str]) -> str:
        return ", ".join(values) if full or len(values) <= 4 else f"{', '.join(values[:4])}, +{len(values) - 4} more"

    rank, taxa = sampling["taxon_filter_rank"], sampling["taxon_filter_values"]
    settings = sampling["setting_filter_values"]
    parts = [f"{rank}: {listed(taxa)}"] if rank and taxa else []
    if settings:
        parts.append(f"settings: {listed(settings)}")
    return "; ".join(parts) or "all tracks of the run"


def _fmt_subsample(sampling: dict[str, Any], full: bool = False) -> str:
    """Return the subsample of a session (strata, tracks per stratum, min/max, seed)."""
    edges = sampling["prob_edges"]
    strata = [name for name, active in ((str(sampling["stratum_rank"]), sampling["stratum_rank"]),
                                        ("probability bins" + (f" ({', '.join(f'{e:g}' for e in edges)})"
                                                               if full and edges else ""), edges),
                                        ("camera setting", sampling["stratum_setting"])) if active]
    if not strata and not sampling["min_per_stratum"] and not sampling["max_per_stratum"]:
        return "all selected tracks"
    parts = ["strata by " + " + ".join(strata)] if strata else ["one stratum"]
    if sampling["min_per_stratum"]:
        parts.append(f"min {sampling['min_per_stratum']}")
    parts.append(f"{sampling['max_per_stratum']} per stratum" if sampling["max_per_stratum"] else "all tracks kept")
    parts.append(f"seed {sampling['seed']}")
    return "; ".join(parts)


def _session_tooltip(info: dict[str, Any]) -> str:
    """Return the tooltip of a sessions table row."""
    sampling = info["sampling"]
    return "\n".join([
        f"Validator: {info['validator']}" + ("   (blind determination)" if info["blind"] else ""),
        f"Session: {info['session_id']}",
        f"Started: {_fmt_time(info['created'])}   ·   last update: {_fmt_time(info['updated'])}",
        f"Progress: {info['n_validated']} of {info['n_sampled']} sampled tracks validated",
        f"Selection: {_fmt_selection(sampling, full=True)}",
        f"Subsample: {_fmt_subsample(sampling, full=True)}",
        f"File: {Path(info['csv_path']).name}",
    ])


def _parse_grid_zoom(text: str) -> GridZoom:
    """Return the grid zoom of its settings string ('px:<cell>' / 'x:<k>'), else the default."""
    mode, _, value = text.partition(":")
    if mode == "px" and value.isdigit() and int(value) in GRID_ZOOM_STEPS:
        return ("px", int(value))
    if mode == "x" and value.isdigit() and int(value) >= 1:
        return ("x", int(value))
    return ("px", GRID_ZOOM_STEPS[2])


def _parse_single_zoom(text: str) -> int | str:
    """Return the single-view zoom of its settings string (SINGLE_ZOOM_STEPS or FIT, the default)."""
    return int(text) if text.isdigit() and int(text) in SINGLE_ZOOM_STEPS else FIT


def _percent_bar_chart(title: str, bar_sets: list[QBarSet], categories: list[str], text_color: QColor) -> QChart:
    """Return a chart of bars stacked to 100 % (one bar per category, one segment per bar set)."""
    chart = QChart()
    chart.setTitle(title)
    chart.setBackgroundVisible(False)
    chart.setTitleBrush(text_color)
    series = QPercentBarSeries()
    for bar_set in bar_sets:
        series.append(bar_set)
    chart.addSeries(series)
    axis_x = QBarCategoryAxis()
    axis_x.append(categories)
    axis_x.setLabelsBrush(text_color)
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
    series.attachAxis(axis_x)
    axis_y = QValueAxis()
    axis_y.setRange(0, 100)
    axis_y.setLabelFormat("%d %%")
    axis_y.setLabelsBrush(text_color)
    chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
    series.attachAxis(axis_y)
    chart.legend().setLabelBrush(text_color)
    return chart


def main() -> None:
    """Start the validation window on its own (entry point 'validate')."""
    # Configure log levels and console output
    configure_logging()

    # Start Qt application
    app = QApplication(sys.argv)
    qt_themes.set_theme(load_theme())
    window = ValidationWindow(standalone=True)  # standalone mode offers updates
    window.showMaximized()
    window.open_run(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

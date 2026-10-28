# Insect Detect Post - Post-processing of camera trap data

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="540" alt="Insect Detect logo">

[![License AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://choosealicense.com/licenses/agpl-3.0/)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![DOI Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.21822140.svg)](https://doi.org/10.5281/zenodo.21822140)

This repository contains GUI-based software for post-processing of data captured
with the [Insect Detect](https://maxsitt.github.io/insect-detect-docs/) camera trap.
It turns the images and metadata captured by the camera trap into cropped,
classified and filtered results ready for analysis, by combining image processing,
AI-based classification and metadata aggregation into a single configurable pipeline.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Source structure](#source-structure)
- [Output structure](#output-structure)
- [Settings](#settings)
- [Validation](#validation)
- [AI use](#ai-use)
- [License](#license)
- [Citation](#citation)

## Features

- **Image processing** - crop individual detections from full frames (square or
  original aspect ratio) and/or draw bounding box + metadata overlays.
- **Classification** - classify cropped detections with either the
  [BioCLIP 2](https://imageomics.github.io/bioclip-2/) foundation model or a custom
  [Ultralytics](https://docs.ultralytics.com/tasks/classify) YOLO classification model.
- **Crop sorting** - automatically move cropped detections into subdirectories, either
  by each individual prediction or by the final prediction of the whole track.
- **Metadata processing** - aggregate image-level predictions into per-track results
  with weighted probabilities. Filter tracks by detection confidence, duration, and
  prediction probability.
- **Validation** - check the track-level predictions manually in a dedicated window,
  on all tracks or a stratified subsample, and get a corrected dataset plus the model's
  error rates (including accuracy per rank, precision per taxon).
- **Config profiles** - save, switch between and reuse multiple named configuration
  files (`.yaml`) for different processing setups.
- **Desktop GUI** - configure and run the pipeline with a PySide6-based interface
  that inspects the source dataset, reports live progress and supports cancelling
  a running pipeline at any time.

---

## Installation

### Prerequisites

> [!NOTE]
> Python does not need to be installed separately - `uv` automatically downloads
> and manages the required Python version for you when running `uv sync`.

**Install [uv](https://docs.astral.sh/uv/getting-started/installation/):**

Windows:

``` bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux and macOS:

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install [git](https://git-scm.com/install/):**

Windows:

``` bash
winget install --id Git.Git -e --source winget
```

Linux (Debian/Ubuntu):

``` bash
sudo apt update && sudo apt install git
```

macOS:

``` bash
xcode-select --install
```

### Install post-processing software

Clone the `insect-detect-post` repository:

``` bash
git clone https://github.com/maxsitt/insect-detect-post
```

Open a Terminal in your `insect-detect-post` directory or change directory
(use full path after `cd`):

``` bash
cd insect-detect-post
```

There are three installation options:

1. CPU-only version:

    ``` bash
    uv sync --extra cpu
    ```

2. GPU with CUDA 13 support:

    ``` bash
    uv sync --extra cuda132
    ```

3. Legacy GPU with CUDA 12 support:

    ``` bash
    uv sync --extra cuda126
    ```

> [!NOTE]
> The `cuda132`/`cuda126` options require a matching NVIDIA GPU driver.
> Run `nvidia-smi` and check the reported `CUDA Version` (top right)
> to see the maximum CUDA version your installed driver supports.

### Update

Running the GUI checks for updates on each start. If a new version is available,
show all changes and confirm the installation with the `Update available` button.
Restart the application afterwards to use the new version.

You can also update from the command line:

``` bash
cd insect-detect-post
uv run --no-sync update
```

Python packages are re-synced if the dependencies changed, keeping your installed
`cpu`/`cuda126`/`cuda132` extra. An update installed from the GUI never replaces
packages while they are in use, so if the dependencies changed, the required
`uv sync --extra <cpu|cuda126|cuda132>` command is shown after the update instead.

> [!NOTE]
> Your configuration is never modified by an update. Config files are not tracked by
> git, so all of your settings and profiles are kept as they are (see
> [Configuration profiles](#configuration-profiles)).

---

## Usage

Start the GUI by running:

``` bash
uv run --no-sync gui
```

> [!IMPORTANT]
> Always include `--no-sync`. Running `uv run gui` without it re-syncs the environment
> without your chosen extra, which uninstalls PyTorch and the ONNX runtime.

1. **Select a source directory** - select a directory that contains images and
   metadata captured with the [`insect-detect`](https://github.com/maxsitt/insect-detect)
   software. Subdirectories are scanned recursively, so per-device/per-session folder
   structures are supported.
2. **Select an output directory** - all processed results are written here.
3. **Configure the pipeline** - enable/adjust the settings described in
   [Settings](#settings) below.
4. **Run the pipeline** - progress and log messages are shown live, and a run can
   be cancelled at any time.

Models and supporting files referenced in the configuration (e.g. classification models,
the BioCLIP-to-GBIF species mapping of all Arthropoda in the BioCLIP Tree of Life, used to
build per-country and per-continent region filters) are downloaded automatically on first use
from the release assets.

> [!NOTE]
> The BioCLIP 2 model and associated files are downloaded from Hugging Face on first use,
> or again if the BioCLIP repo is updated. This produces extra log output, including
> HTTP requests and a warning about "unauthenticated requests to the HF Hub" - this is
> expected and can be safely ignored. No `HF_TOKEN` is required to use this pipeline.

### Configuration profiles

Configuration files are stored as `.yaml` files in the `configs/` directory. The
active profile is tracked in `config_selector.yaml` and can be switched, created
or updated from the GUI, so different processing setups (e.g. per project or per
classifier) can be saved and reused without editing YAML manually.

The default config file (`config.yaml`) and the config selector file are created
automatically on first launch. Both are ignored by git, together with all profiles
you create, which means your settings are never modified or removed by an update.

To restore the default values, use the `Reset to Defaults` button in the GUI.
This resets all settings of the active profile while keeping the selected source
and output paths. Alternatively, delete `config.yaml` and restart the GUI to
recreate it from scratch, which also clears both paths.

<details>
<summary>Default YAML configuration</summary>

``` yaml
source_path: null
output_path: null
device: cpu
processing:
  crop:
    enabled: true
    method: square
  overlay:
    enabled: false
classification:
  bioclip:
    enabled: true
    batch_size: 16
    rank: species
    filter_arthropods:
      enabled: true
      taxa:
      - Insecta
      - Arachnida
      countries: []
      continents: []
  ultralytics:
    enabled: false
    batch_size: 16
    model: platform_insect-detect_yolo26s-cls_v1-0-0.onnx
  sort_crops:
    enabled: false
  sort_tracks:
    enabled: true
metadata:
  filter_crops:
    enabled: false
    min_det_conf: 0.2
    min_pred_prob: 0.1
  filter_tracks:
    enabled: false
    min_dur_s: 2.0
    max_dur_s: 3600.0
    min_det_conf_mean: 0.2
    min_pred_prob_weighted: 0.2
  estimate_size:
    enabled: false
    frame_width_mm: 230
    frame_height_mm: 130
```

</details>

---

## Source structure

For data from a single camera trap, select the `insect-detect/data` folder
as source directory. For data from multiple devices, the following source
structure is recommended:

``` text
<source_path>/
├── insdet-cam01/
│   └── data/
├── insdet-cam02/
│   └── data/
├── insdet-cam03/
│   └── data/
└── ...
```

## Output structure

Each run creates a timestamped directory under `data_processed/` in your output directory:

``` text
<output_path>/
└── data_processed/
    └── 2026-08-05_14-30-12_<source>_processed/
        ├── 2026-08-05_14-30-12_<config>.json  # config snapshot for this run
        ├── 2026-08-05_14-30-12_run.log        # log file
        ├── 2026-08-05_14-30-12_stats.json     # per-step durations
        ├── filters/                           # BioCLIP region filters + mapping provenance
        ├── metadata/
        │   ├── <source>_metadata_merged.csv               # all source metadata, harmonized
        │   ├── <source>_metadata_merged_classified.csv    # + per-image predictions
        │   ├── ..._classified_candidates.csv              # per-track candidate predictions
        │   ├── ..._classified_final.csv                   # one row per track (best prediction)
        │   └── ..._classified_final_validated_<validator>_<started>.csv  # final + validation
        ├── validation/                                    # validation sessions + statistics
        └── images/
            ├── crops/      # cropped detections, optionally sorted into prediction subdirectories
            └── overlays/   # copies of full frames with bounding boxes + metadata drawn on
```

`..._classified_final.csv` is the file to use for analysis of the raw model predictions,
`..._classified_final_validated_<validator>_<started>.csv` for analyzing the corrected data
after validation. Both have one row per tracking ID including its final prediction and
weighted probability, the validated file adds the verdicts and determinations.

For BioCLIP predictions at species rank, the candidates and final CSVs also carry
`gbif_species` and `gbif_species_key`: the accepted GBIF species (Catalogue of Life) of the
predicted Tree of Life species, empty if it has none. Further details such as the GBIF
lineage or IUCN category can be joined from the species mapping
(`filters/bioclip_gbif_species_mapping_*.csv`) on `gbif_species_key`, whose provenance record
is copied into the run's `filters/` folder.

---

## Settings

All settings are validated with [Pydantic](https://pydantic.dev/docs/) - out-of-range
numeric values are automatically clamped to their allowed bounds and a warning is
logged. `bioclip` and `ultralytics` are mutually exclusive (enable only one
classifier), as are `sort_crops` and `sort_tracks`. Crop sorting and metadata processing
both require classification results, so one of the two classifiers must be enabled.
`metadata.filter_tracks.min_dur_s` must not be greater than `max_dur_s`.

### General

| Setting       | Type / Options   | Default | Description                                                               |
|---------------|------------------|---------|---------------------------------------------------------------------------|
| `source_path` | `string \| null` | `null`  | Full path to the source directory containing images and metadata.         |
| `output_path` | `string \| null` | `null`  | Full path to the output directory where all results will be saved.        |
| `device`      | `cpu`, `cuda`    | `cpu`   | Device used for model inference. `cuda` requires a GPU with CUDA support. |

### Image Processing

Post-processing settings applied to full-frame images.

| Setting                      | Type / Options       | Default  | Description                                                                                                                                      |
|------------------------------|----------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `processing.crop.enabled`    | `bool`               | `true`   | Save individual detections as separate crop files. Detections without a full image (crop-only datasets) are copied as-is.                        |
| `processing.crop.method`     | `square`, `original` | `square` | Crop method: `square` crops to a square bounding box, which can improve classification; `original` keeps the original bounding box aspect ratio. |
| `processing.overlay.enabled` | `bool`               | `false`  | Draw bounding boxes and metadata overlays on full images and save them as copies.                                                                |

### Classification

Classification settings applied to cropped detections.

| Setting                                               | Type / Options                                                                                  | Default                                          | Description                                                                                                                                                   |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `classification.bioclip.enabled`                      | `bool`                                                                                          | `true`                                           | Enable classification with the BioCLIP 2 model (via [`pybioclip`](https://github.com/Imageomics/pybioclip) package).                                          |
| `classification.bioclip.batch_size`                   | `int` (1-256)                                                                                   | `16`                                             | Batch size used for BioCLIP inference. Higher = faster, but uses more memory (GPU VRAM if device: cuda, system RAM otherwise).                                |
| `classification.bioclip.rank`                         | `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species`                             | `species`                                        | Predict to selected taxonomic level. For ranks above species, species-level probabilities are summed up to the target rank.                                   |
| `classification.bioclip.filter_arthropods.enabled`    | `bool`                                                                                          | `true`                                           | Restrict BioCLIP predictions to the selected arthropod groups and/or GBIF occurrence in the selected countries or continents.                                 |
| `classification.bioclip.filter_arthropods.taxa`       | `all`, `Insecta`, `Arachnida`, `Diplopoda`, `Chilopoda`, `Isopoda`                              | `[Insecta, Arachnida]`                           | Arthropod groups that predictions are restricted to; multiple can be selected. `all` covers the whole phylum Arthropoda.                                      |
| `classification.bioclip.filter_arthropods.countries`  | country codes                                                                                   | `[]`                                             | Countries that predictions are restricted to (based on GBIF occurrence records). Empty for no restriction by country. Cannot be combined with `continents`.   |
| `classification.bioclip.filter_arthropods.continents` | `AFRICA`, `ANTARCTICA`, `ASIA`, `EUROPE`, `NORTH_AMERICA`, `OCEANIA`, `SOUTH_AMERICA`           | `[]`                                             | Continents that predictions are restricted to (based on GBIF occurrence records). Empty for no restriction by continent. Cannot be combined with `countries`. |
| `classification.ultralytics.enabled`                  | `bool`                                                                                          | `false`                                          | Enable classification with a custom Ultralytics YOLO classification model.                                                                                    |
| `classification.ultralytics.batch_size`               | `int` (1-256)                                                                                   | `16`                                             | Batch size used for Ultralytics inference. Higher = faster, but uses more memory (GPU VRAM if device: cuda, system RAM otherwise).                            |
| `classification.ultralytics.model`                    | `string`                                                                                        | `platform_insect-detect_yolo26s-cls_v1-0-0.onnx` | Filename of the Ultralytics classification model, downloaded automatically from the models registry on first use.                                             |
| `classification.sort_crops.enabled`                   | `bool`                                                                                          | `false`                                          | Move cropped detections into subdirectories based on the individual prediction.                                                                               |
| `classification.sort_tracks.enabled`                  | `bool`                                                                                          | `true`                                           | Move all crops belonging to a track into one subdirectory based on the track's final prediction.                                                              |

#### Filter files

Country and continent filters are built on first use by querying the GBIF occurrence API
(this can take a few minutes) and are then cached in `filters/`. The scripts that regenerate
the underlying reference data (`filters/resolve_gbif_country_codes.py`,
`filters/build_bioclip_gbif_species_mapping.py`) only need to be re-run when GBIF's country
enumeration or the BioCLIP Tree of Life version changes.

A species is included in a region's filter if GBIF has at least 3 occurrence records of it in
that region (default of `MIN_OCCURRENCE_COUNT` in `constants.py`). With several countries or
continents selected, this applies to each region on its own, not to their sum. Rare or rarely
recorded species can therefore be missing from a filter even if they occur in the region.

Tree of Life species are matched to GBIF species keys of the
[Catalogue of Life](https://www.catalogueoflife.org/) Extended Release (COL XR), the taxonomy
GBIF uses to interpret occurrence records. The mapping is updated with new releases of this
software, and its COL release is recorded next to it (`filters/bioclip_gbif_species_mapping_*.json`).
When a filter is built, the log shows the COL release of the GBIF occurrence data and of the
mapping. If they differ, the filter still works, but species that COL has split, merged or
renamed since the mapping's release can be missing from it.

Countries and continents are assigned differently by GBIF, so a continent is not simply the
sum of its countries:

- **Countries** follow the country stated by the data publisher (or derived from the
  coordinates), including records from coastal waters. Overseas regions belong to their
  country, e.g. `ES` includes the Canary Islands.
- **Continents** are assigned from the coordinates using GBIF's
  [continent boundaries](https://github.com/gbif/continents) and cover land only. Records
  from the sea get no continent, and records without coordinates from a country spanning two
  continents (e.g. Russia, Turkey) get none either. Some islands fall on another continent
  than their country, e.g. the Canary Islands and Madeira count as `AFRICA`, Cyprus and most
  of Turkey as `ASIA`.

For a camera trap on an island or in a transcontinental country, select the country rather
than the continent.

### Metadata Processing

Metadata processing settings applied to the final results. Crop filters drop single crops before the
track values (duration, mean confidence, weighted probability) are calculated, track filters then
drop whole tracking IDs.

| Setting                                         | Type / Options    | Default  | Description                                                                                                     |
|-------------------------------------------------|-------------------|----------|-----------------------------------------------------------------------------------------------------------------|
| `metadata.filter_crops.enabled`                 | `bool`            | `false`  | Filter crops based on detection confidence and prediction probability, before track aggregation.                |
| `metadata.filter_crops.min_det_conf`            | `float` (0-1)     | `0.2`    | Minimum detection confidence required to keep a crop (`0` turns it off).                                        |
| `metadata.filter_crops.min_pred_prob`           | `float` (0-1)     | `0.1`    | Minimum probability of the crop's top-1 prediction required to keep a crop (`0` turns it off).                  |
| `metadata.filter_tracks.enabled`                | `bool`            | `false`  | Filter tracking IDs based on tracking duration, mean detection confidence and weighted prediction probability.  |
| `metadata.filter_tracks.min_dur_s`              | `float` (0-600)   | `2.0`    | Minimum total tracking duration (seconds, one decimal place) required to keep a tracking ID (`0` turns it off). |
| `metadata.filter_tracks.max_dur_s`              | `float` (0-21600) | `3600.0` | Maximum total tracking duration (seconds, one decimal place) to keep a tracking ID (`0`: single crop only).     |
| `metadata.filter_tracks.min_det_conf_mean`      | `float` (0-1)     | `0.2`    | Minimum mean detection confidence required to keep a tracking ID (`0` turns it off).                            |
| `metadata.filter_tracks.min_pred_prob_weighted` | `float` (0-1)     | `0.2`    | Minimum weighted mean probability of the final prediction required to keep a tracking ID (`0` turns it off).    |
| `metadata.estimate_size.enabled`                | `bool`            | `false`  | Estimate physical size based on bounding box dimensions and frame size (millimeters).                           |
| `metadata.estimate_size.frame_width_mm`         | `int` (10-1000)   | `230`    | Physical width of the camera frame (millimeters).                                                               |
| `metadata.estimate_size.frame_height_mm`        | `int` (10-1000)   | `130`    | Physical height of the camera frame (millimeters).                                                              |

---

## Validation

Check the model predictions of a processed run manually, one **track** at a time.
A validation gives you:

- a **corrected dataset** - your determinations next to the model predictions of the validated tracks
- the **error rates** of the model - accuracy per rank, precision per predicted taxon and accuracy per
  probability range, e.g. to report them or to choose an informed `min_pred_prob_weighted` threshold

Open the validation window with `Validate Results` in the main window (also offered when a run has
finished), or start it on its own:

``` bash
uv run --no-sync validate
```

Only runs classified with BioCLIP 2 can be validated. Press `F1` in the window for the validation
guidelines and all keyboard shortcuts.

### Workflow

1. **Start a session** - select a processed run (`data_processed/<timestamp>_<source>_processed`) and
   enter your name as validator. Optional:
   - **Blind determination** - the model prediction stays hidden until you have entered your own.
   - **Track selection** - only tracks within a duration or weighted-probability range, of selected
     camera settings, recording periods or predicted taxa.
   - **Stratified subsample** - a random sample of up to *n* tracks per stratum (predicted taxon,
     probability range, camera setting) instead of every track: representative error rates for a
     fraction of the work.
2. **Validate the tracks** - confirm the prediction (fully or down to a rank), correct it, or mark the
   track as not an arthropod (false positive) or not determinable. Qualifiers (uncertain, aggregate)
   and flags (same individual, multiple individuals, low image quality) record additional information.
   Select several tracks to handle them at once with batch verdicts/determinations.
3. **Review the results** - view/export the statistics and export the validated CSV (see below).

Your **verdict** follows from what you enter: *agree*, *partial* (confirmed, but not down to the rank the
run was classified to), *corrected*, *not an arthropod* or *not determinable*. Every change is saved
automatically, so you can close the window at any time and resume the session later.

### Results

- **`Statistics`** - verdicts, accuracy per rank and per probability range, precision per predicted taxon
  (each with 95 % confidence intervals), the most frequent confusions and the agreement with other
  sessions of the run. Tracks with low image quality or different taxa are left out of the accuracies by
  default. For a stratified subsample, report the `_weighted` columns: they estimate the results for all
  tracks of the sampled strata, not only the sample. `Export statistics (CSV)` saves every table.
- **`Export validated CSV`** - writes `..._classified_final_validated_<validator>_<started>.csv` next to
  the final CSV: every track of the run with your verdict and determination, plus `combined_<rank>` -
  your taxon for validated tracks, none for false positives, the model's for all other tracks
  (`combined_source` tells which). Tracks linked as the same individual share an `individual_id`.
  Use this file instead of `..._classified_final.csv` for analysis after a validation.

Every session is identified by its validator and its start time (`<started>` = `YYYY-MM-DD_HH-MM-SS`):

``` text
<run>/validation/
├── <validator>_<started>_validation.csv    # one row per sampled track: verdict and determination
├── <validator>_<started>_validation.json   # sampling settings, provenance and UI state
├── <validator>_<started>_validation.log    # application log while the session is open
├── <validator>_<started>_history.jsonl     # every verdict change (audit trail)
└── stats_<validator>_<started>/            # exported statistics tables and summary.json
```

### Several sessions per run

Each session keeps its own track selection and subsample, so a run can be split between validators
(e.g. one expert for Syrphidae, another for bees) or validated in parts (e.g. one month per session).
Validate the same tracks in two sessions only on purpose: `Statistics` compares them as a re-check of
the same validator or as agreement between validators.

- **Splitting one design** - create one session as a template, then start each part with
  `Copy settings` and add only a taxa filter, at the rank of the strata. With the same strata and
  seed, every part samples exactly the tracks the whole design would have sampled for its taxa.
- **Pre-filling a new session** - pick sessions in `Pre-fill new session from:` above `Start new validation
  session` to start with your verdicts, links, flags, tags and notes of those sessions (the latest wins)
  on the tracks the new session samples. The sample itself stays the same, and a blind session can only
  be pre-filled from blind sessions. For example, link the individuals of a whole run in one session,
  then start subsamples with the links in place, or take a larger subsample with the same strata and
  seed: it contains the smaller one, whose verdicts arrive pre-filled. Pre-filled tracks are marked
  (filter `Pre-filled`) and stay validated, a changed one counts as edited. The sessions table names the
  sources of a pre-filled session (`Pre-filled from`).
- **One validated CSV** - select the sessions with Ctrl- or Shift-click and press `Export validated CSV`.
  The dialog lists them by start time, latest first: a track validated in several sessions gets the
  verdict of the highest one (*priority*, move a session up to let it win, e.g. an expert's session
  over an overview session), or the deepest taxon all its sessions agree on (*lowest common ancestor*,
  a track marked *not an arthropod* in one session and determined in another becomes not determinable).
  Links, flags, tags and notes always come from the highest session. Sessions of a run processed again are refused.
  With one session selected, the button writes the same file as on the validation page.

``` text
<run>/metadata/..._classified_final_validated_multi_<time>/
├── ..._classified_final_validated_multi_<time>.csv   # every track of the run, as the session export
├── verdict_conflicts.csv                             # every verdict of the tracks validated in several sessions
├── provenance.json                                   # rule, session order, checksums, counts, removed links
└── validation_sessions/                              # copies of the session files the CSV is made from
```

The CSV adds `validation_session` (the session(s) of the verdict), `validation_n_sessions` (how many
sessions validated the track) and `validation_conflict` (how their verdicts differ: `different_depth`,
`different_taxa` or `false_positive_disputed`). Before analyzing it:

- Report the statistics of each session, they are not pooled. `stratum_id` and `stratum_n_population`
  belong to the session in `validation_session`.
- A session that selects tracks by predicted taxon removes wrong identifications of that taxon, but
  misses its tracks the model predicted as something else: counts of the validated taxa are corrected
  downwards, not upwards.
- The parts can differ in sampling and depth: `combined_source`, `val_lowest_rank` and
  `validation_session` tell them apart.
- The CSV is a snapshot of the sessions at the moment of the export: verdicts added or changed later
  are only included after exporting again.
- A pre-filled verdict that was not changed counts as the verdict of its source session when both are
  exported together. A track reset in one session has no verdict there, so another session's verdict
  applies to it.
- Use the same validator name in every session of one person: the `validator` column and
  `provenance.json` name each person by it, so two spellings look like two validators.

---

## AI use

Parts of this code are developed with AI coding assistance.
All code is reviewed and tested extensively by the author.

---

## License

This repository is licensed under the terms of the GNU Affero General Public
License v3.0 ([GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)).

---

## Citation

If you use resources from this repository, please cite it as:

``` text
Sittinger, M. (2026). Software for post-processing of data captured with the Insect Detect camera trap (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.21822140
```

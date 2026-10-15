"""Resolve BioCLIP's Tree of Life species list to GBIF species keys (Catalogue of Life).

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

One-off, manually run script. Writes filters/bioclip_gbif_species_mapping_<phylum>.csv and a
JSON provenance record next to it (checklist, COL release, inputs, match summary). Both are
published as release assets. Re-run when the BioCLIP/TOL model/dataset version changes, or to
move the mapping to a newer COL release (delete the existing CSV first).

Names are matched against GBIF_CHECKLIST_KEY (Catalogue of Life Extended Release, COL XR),
whose taxon keys are alphanumeric strings, via individual GBIF API calls in a ThreadPoolExecutor.

Each row holds the Tree of Life lineage ('bioclip_*') next to the GBIF lineage ('gbif_*'), both
from species up to kingdom. The GBIF lineage adds subfamily, superfamily and suborder.

Scoped to PHYLUM_FILTER (insectdetect_post.constants); set to None to resolve the full TOL species list.

Functions:
    fetch_tol_species_records():          Fetch the TOL species list without loading the CLIP model.
    resolve_species_to_gbif():            Resolve a single TOL species name to a GBIF species key.
    resolve_all_species():                Resolve all TOL species records to GBIF species keys concurrently.
    write_mapping_csv():                  Write resolved records to CSV, sorted by species name.
    write_provenance_json():              Write the provenance record of the mapping next to the CSV.
    load_previous_resolution():           Load an existing mapping CSV for retry runs.
    build_bioclip_gbif_species_mapping(): Fetch, filter and resolve TOL to GBIF species, then write the mapping.
"""

import json
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl
from bioclip._constants import BIOCLIP_MODEL_STR, HF_DATAFILE_REPO_TYPE, Rank
from bioclip.predict import create_classification_dict, get_tol_repo_id
from huggingface_hub import hf_hub_download
from pygbif import species as gbif_species
from requests.exceptions import HTTPError
from tqdm import tqdm

from insectdetect_post.asset_manager import compute_sha256
from insectdetect_post.build_region_filter import fetch_col_release, format_col_release
from insectdetect_post.constants import FILTERS_PATH, GBIF_CHECKLIST_KEY, PHYLUM_FILTER

# Create module-level logger
logger = logging.getLogger(__name__)

# Output CSV and provenance JSON paths
_scope = PHYLUM_FILTER if PHYLUM_FILTER is not None else "all"
OUTPUT_CSV = FILTERS_PATH / f"bioclip_gbif_species_mapping_{_scope}.csv"
OUTPUT_JSON = OUTPUT_CSV.with_suffix(".json")

# TOL species list in the BioCLIP data repo, the same file TreeOfLifeClassifier reads its labels from
TOL_SPECIES_FILE = "embeddings/txt_emb_species.json"

# Ranks of the Tree of Life and the GBIF lineage, from species up, as 'bioclip_*' and 'gbif_*' columns
TOL_RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
GBIF_RANKS = ["species", "genus", "subfamily", "family", "superfamily", "suborder", "order", "class",
              "phylum", "kingdom"]
BIOCLIP_COLUMNS = [f"bioclip_{rank}" for rank in TOL_RANKS]
GBIF_RANK_COLUMNS = [f"gbif_{rank}" for rank in GBIF_RANKS]

# Fixed column order for the output CSV
COLUMN_ORDER = [
    *BIOCLIP_COLUMNS, *GBIF_RANK_COLUMNS,
    "gbif_species_key", "gbif_authorship", "gbif_rank", "gbif_match_type", "gbif_status", "iucn_category",
]

# Max workers for the GBIF API calls and retry/backoff parameters for transient failures
MAX_WORKERS = 64
MAX_RETRIES = 6
REQUEST_TIMEOUT_S = 10
MAX_BACKOFF_S = 20.0

# GBIF taxon ranks allowed to resolve to a species key (species or lower)
SPECIES_OR_LOWER_RANKS = frozenset({
    "SPECIES", "SUBSPECIES", "VARIETY", "SUBVARIETY", "FORM", "SUBFORM",
})


def fetch_tol_species_records() -> tuple[list[dict[str, str]], dict[str, str]]:
    """Fetch the TOL species list without loading the CLIP model.

    Downloads the same TOL_SPECIES_FILE that TreeOfLifeClassifier.get_label_data() reads (from the
    repo's main branch, like pybioclip), then builds the same classification dicts via
    bioclip.predict.create_classification_dict(), without instantiating the full classifier
    (no CLIP model weights loaded, no torch needed).

    The file's SHA-256 identifies the species list independently of the revision it was downloaded
    from: the repo keeps the BioCLIP 2 labels unchanged across later data updates.

    Returns:
        Tuple of (records, source). Records is a list of dicts with the BIOCLIP_COLUMNS keys.
        Source has the keys file, revision (Hugging Face commit hash of the download, its snapshot
        directory in the cache) and sha256.
    """
    repo_id = get_tol_repo_id(BIOCLIP_MODEL_STR)
    txt_names_json = Path(hf_hub_download(
        repo_id=repo_id, filename=TOL_SPECIES_FILE, repo_type=HF_DATAFILE_REPO_TYPE
    ))
    with open(txt_names_json, encoding="utf-8") as f:
        txt_names = json.load(f)
    source = {
        "file": TOL_SPECIES_FILE,
        "revision": txt_names_json.parent.parent.name,
        "sha256": compute_sha256(txt_names_json),
    }
    records = [
        {f"bioclip_{rank}": lineage[rank] for rank in TOL_RANKS}
        for lineage in (create_classification_dict(names=name_ary, rank=Rank.SPECIES) for name_ary in txt_names)
    ]
    return records, source


def _backoff_seconds(error: Exception, attempt: int) -> float:
    """Compute the retry wait time.

    Honors a 'Retry-After' header if present, otherwise falls back to exponential backoff with jitter.
    """
    capped_wait = min(2 ** attempt, MAX_BACKOFF_S)
    if isinstance(error, HTTPError) and error.response is not None and error.response.status_code == 429:
        retry_after = error.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(capped_wait, float(retry_after)) + random.uniform(0, 1)
            except ValueError:
                pass
    return random.uniform(0, capped_wait)


def _unresolved(match_type: str) -> dict[str, object]:
    """Return the GBIF columns of a species without a match."""
    return {
        **dict.fromkeys(GBIF_RANK_COLUMNS),
        "gbif_species_key": None,
        "gbif_authorship": None,
        "gbif_rank": None,
        "gbif_match_type": match_type,
        "gbif_status": None,
        "iucn_category": None,
    }


def resolve_species_to_gbif(name: str, kingdom: str | None = None, phylum: str | None = None) -> dict[str, object]:
    """Resolve a single TOL species name to a GBIF species key.

    Queries GBIF's name match endpoint against GBIF_CHECKLIST_KEY with strict=True: the name must
    match exactly after normalization (e.g. of Latin gender endings, reported as 'VARIANT'), with no
    fuzzy matching of similar names and no fallback to a higher taxon. Strict scoring ignores every
    rank of a supplied classification except the kingdom, which is sent to tell apart names that
    exist in several kingdoms. `phylum` is only used to verify the result, never sent.

    For a synonym, the GBIF lineage, gbif_species_key, gbif_authorship and gbif_rank describe the
    accepted taxon, while gbif_status keeps describing the matched name, 'SYNONYM' still flags the
    TOL name. If the accepted taxon is below species rank (e.g. a subspecies), gbif_species and
    gbif_species_key are those of its species, while gbif_authorship and iucn_category, which would
    describe the subspecies, are left empty.

    A species key is only set at species rank or below and never for matchType 'HIGHERRANK', since
    a name match can answer with a taxon of higher rank than the requested species.

    Args:
        name: TOL species scientific name.
        kingdom: Optional TOL kingdom, sent to disambiguate names that exist in several kingdoms.
        phylum: Optional expected phylum, compared against the phylum GBIF returns.

    Returns:
        Dict with the GBIF_RANK_COLUMNS keys (str | None) and gbif_species_key, gbif_authorship,
        gbif_rank, gbif_match_type, gbif_status, iucn_category (IUCN Red List category code).
    """
    result: dict | None = None
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            result = gbif_species.name_backbone(
                scientificName=name, taxonRank="species", kingdom=kingdom, strict=True,
                checklistKey=GBIF_CHECKLIST_KEY, timeout=REQUEST_TIMEOUT_S
            )
            break
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise ValueError(
                    f"GBIF rejected the name match request for '{name}' (HTTP {status})."
                ) from e
            last_error = e
        except Exception as e:
            last_error = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(_backoff_seconds(last_error, attempt))
    if result is None:
        raise RuntimeError(f"Failed to resolve '{name}' after {MAX_RETRIES} attempts: {last_error}")

    match_type = result.get("diagnostics", {}).get("matchType", "NONE")
    usage = result.get("usage")

    if usage is None:
        return _unresolved(match_type)

    # For a synonym, the accepted taxon is the one the species key should point at
    accepted_usage = result.get("acceptedUsage")
    resolved_usage = accepted_usage if result.get("synonym", False) and accepted_usage else usage
    resolved_rank = resolved_usage.get("rank")

    # The classification is that of the accepted taxon, from the domain down to the taxon itself
    classification = {entry.get("rank"): entry for entry in result.get("classification") or []}
    lineage = {f"gbif_{rank}": classification.get(rank.upper(), {}).get("name") for rank in GBIF_RANKS}

    species_key = None
    if resolved_rank in SPECIES_OR_LOWER_RANKS and match_type != "HIGHERRANK":
        species_key = classification.get("SPECIES", {}).get("key")

    if phylum is not None and species_key is not None and lineage["gbif_phylum"] not in (None, phylum):
        logger.warning("'%s' resolved to GBIF phylum '%s', expected '%s' (species key %s)",
                       name, lineage["gbif_phylum"], phylum, species_key)

    at_species_rank = resolved_rank == "SPECIES"
    iucn_category = next(
        (status.get("statusCode") for status in result.get("additionalStatus") or []
         if status.get("datasetAlias") == "IUCN"), None
    )
    return {
        **lineage,
        "gbif_species_key": str(species_key) if species_key is not None else None,
        "gbif_authorship": resolved_usage.get("authorship") if at_species_rank else None,
        "gbif_rank": resolved_rank,
        "gbif_match_type": match_type,
        "gbif_status": usage.get("status"),
        "iucn_category": iucn_category if at_species_rank else None,
    }


def resolve_all_species(records: list[dict[str, str]]) -> list[dict[str, object]]:
    """Resolve all TOL species records to GBIF species keys concurrently.

    Runs resolve_species_to_gbif() over the unique species names using up to MAX_WORKERS
    threads, then merges each resolution back onto its source record. A species that still
    fails after MAX_RETRIES gets gbif_match_type='ERROR' instead of aborting the batch.

    Args:
        records: TOL species records as returned by fetch_tol_species_records().

    Returns:
        List of merged dicts (BIOCLIP_COLUMNS + GBIF columns), one per input record, in input order.
    """
    record_by_name: dict[str, dict[str, str]] = {record["bioclip_species"]: record for record in records}
    unique_names = sorted(record_by_name)
    resolved_by_name: dict[str, dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {
            # Missing TOL ranks are empty strings, which must not be sent as a kingdom
            executor.submit(resolve_species_to_gbif, name, record_by_name[name]["bioclip_kingdom"] or None,
                            record_by_name[name]["bioclip_phylum"] or None): name
            for name in unique_names
        }
        for future in tqdm(as_completed(future_to_name), total=len(future_to_name),
                           desc="Resolving TOL species to GBIF species keys"):
            name = future_to_name[future]
            try:
                resolved_by_name[name] = future.result()
            except Exception:
                logger.debug("Failed to resolve '%s' to GBIF", name, exc_info=True)
                resolved_by_name[name] = _unresolved("ERROR")

    return [{**record, **resolved_by_name[record["bioclip_species"]]} for record in records]


def write_mapping_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write resolved records to CSV, sorted by species name.

    Args:
        rows: Merged records as returned by resolve_all_species().
        output_path: Destination CSV path. Parent directory is created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = (
        pl.DataFrame(rows, schema_overrides=dict.fromkeys(COLUMN_ORDER, pl.String))
        .select(COLUMN_ORDER).sort("bioclip_species").unique(subset=["bioclip_species"], keep="first")
    )
    df.write_csv(output_path)


def write_provenance_json(
    csv_path: Path,
    output_path: Path,
    col_release: dict,
    tol_source: dict | None,
) -> None:
    """Write the provenance record of the mapping next to the CSV.

    Args:
        csv_path: The written mapping CSV, summarized and referenced by its SHA-256 checksum.
        output_path: Destination JSON path.
        col_release: COL release the names were matched against (see fetch_col_release()).
        tol_source: TOL species list file, revision and SHA-256 (see fetch_tol_species_records()), if known.
    """
    df = pl.read_csv(csv_path, schema_overrides=dict.fromkeys(COLUMN_ORDER, pl.String))

    def counts(column: str) -> dict[str, int]:
        """Return the value counts of a column, most frequent first, missing values as 'null'."""
        return {str(value) if value is not None else "null": n
                for value, n in Counter(df[column].to_list()).most_common()}

    output_path.write_text(
        json.dumps({
            "csv_sha256": compute_sha256(csv_path),
            "resolved_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "gbif_checklist_key": GBIF_CHECKLIST_KEY,
            "col_release": col_release,
            "bioclip_model": BIOCLIP_MODEL_STR,
            "tol_repo_id": get_tol_repo_id(BIOCLIP_MODEL_STR),
            "tol_species_list": tol_source,
            "phylum_filter": _scope,
            "pygbif_version": version("pygbif"),
            "species": df.height,
            "species_with_key": df["gbif_species_key"].is_not_null().sum(),
            "match_types": counts("gbif_match_type"),
            "statuses": counts("gbif_status"),
            "ranks": counts("gbif_rank"),
            "iucn_categories": counts("iucn_category"),
        }, indent=2) + "\n",
        encoding="utf-8"
    )


def load_previous_resolution(output_path: Path) -> pl.DataFrame | None:
    """Load an existing mapping CSV for retry runs.

    Args:
        output_path: Path to the output CSV.

    Returns:
        The existing DataFrame with every column as String, or None if output_path does not exist yet.
    """
    if not output_path.exists():
        return None
    return pl.read_csv(output_path, schema_overrides=dict.fromkeys(COLUMN_ORDER, pl.String))


def _log_names(message: str, names: list[str], match_type: str, preview_limit: int = 20) -> None:
    """Log a warning with up to preview_limit of the names and a pointer to the rest in the CSV."""
    logger.warning(message, len(names))
    for name in names[:preview_limit]:
        logger.warning("  - %s", name)
    if len(names) > preview_limit:
        logger.warning("  ... and %d more (see gbif_match_type == '%s' rows in the output CSV)",
                       len(names) - preview_limit, match_type)


def build_bioclip_gbif_species_mapping() -> bool:
    """Fetch, filter and resolve TOL to GBIF species, then write to CSV.

    If output CSV already exists, only species with gbif_match_type == 'ERROR'
    are re-resolved; all other rows are kept as-is with no new GBIF calls.

    Returns:
        True on success (CSV written, or nothing to do), False on failure.
    """
    previous = load_previous_resolution(OUTPUT_CSV)
    keep_rows: list[dict[str, object]] = []
    tol_source: dict | None = None

    if previous is not None:
        retry_df = previous.filter(pl.col("gbif_match_type") == "ERROR")
        keep_df = previous.filter(pl.col("gbif_match_type") != "ERROR")
        if retry_df.is_empty():
            logger.info("'%s' already exists with no gbif_match_type == 'ERROR' rows to retry. "
                        "Nothing to do.", OUTPUT_CSV)
            return True
        logger.info("Found existing '%s': retrying %d species with gbif_match_type == 'ERROR', "
                    "keeping %d other rows as-is.", OUTPUT_CSV, retry_df.height, keep_df.height)
        records = retry_df.select(BIOCLIP_COLUMNS).to_dicts()
        keep_rows = keep_df.to_dicts()
    else:
        logger.info("Fetching Tree of Life species list from Hugging Face...")
        try:
            records, tol_source = fetch_tol_species_records()
        except Exception:
            logger.exception("Failed to fetch TOL species list")
            return False
        logger.info("Fetched %d species records (revision %s, SHA-256 %s...).",
                    len(records), tol_source["revision"], tol_source["sha256"][:12])

        if PHYLUM_FILTER is not None:
            records = [record for record in records if record["bioclip_phylum"] == PHYLUM_FILTER]
            logger.info("Filtered to phylum '%s': %d species records.", PHYLUM_FILTER, len(records))

    try:
        col_release = fetch_col_release()
    except Exception:
        logger.exception("Failed to fetch the COL release GBIF currently uses")
        return False
    logger.info("Matching against %s (DOI %s).", format_col_release(col_release), col_release.get("doi"))

    if keep_rows:
        try:
            previous_meta = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_meta = {}
        tol_source = previous_meta.get("tol_species_list")
        previous_release = previous_meta.get("col_release") or {}
        if previous_release.get("clb_dataset_key") != col_release.get("clb_dataset_key"):
            logger.warning("Kept rows were matched against %s, retried rows will be matched against %s. "
                           "Delete '%s' and re-run to resolve all species against one release.",
                           format_col_release(previous_release), format_col_release(col_release), OUTPUT_CSV)

    logger.info("Resolving species to GBIF species keys using %d workers...", MAX_WORKERS)
    try:
        newly_resolved = resolve_all_species(records)
    except Exception:
        logger.exception("Failed to resolve species to GBIF")
        return False

    if keep_rows:
        recovered = sum(1 for row in newly_resolved if row["gbif_match_type"] != "ERROR")
        logger.info("Retry pass: %d/%d previously-failed species now resolved.",
                    recovered, len(newly_resolved))

    resolved = keep_rows + newly_resolved
    with_key = [row for row in resolved if row["gbif_species_key"] is not None]
    logger.info("Resolved %d/%d species to a GBIF species key.", len(with_key), len(resolved))
    statuses = Counter(str(row["gbif_status"]) for row in with_key)
    logger.info("Status of the matched names: %s",
                ", ".join(f"{status} {count}" for status, count in statuses.most_common()))
    below_species = sum(1 for row in with_key if row["gbif_rank"] != "SPECIES")
    if below_species:
        logger.info("%d names resolved to a taxon below species rank, mapped to its species.", below_species)

    def names_with(match_type: str) -> list[str]:
        return [str(row["bioclip_species"]) for row in resolved if row["gbif_match_type"] == match_type]

    if unmatched := names_with("NONE"):
        _log_names("%d species had no GBIF match (review manually):", unmatched, "NONE")
    if higher_rank_only := names_with("HIGHERRANK"):
        _log_names("%d species only matched at a higher rank than species (genus/family/etc.) -- "
                   "gbif_species_key left empty:", higher_rank_only, "HIGHERRANK")
    if failed := names_with("ERROR"):
        _log_names("%d species failed to resolve after retries (network/rate-limit issues -- "
                   "re-run the script to retry just these):", failed, "ERROR")

    logger.info("Writing '%s' and '%s'...", OUTPUT_CSV, OUTPUT_JSON.name)
    try:
        write_mapping_csv(resolved, OUTPUT_CSV)
        write_provenance_json(OUTPUT_CSV, OUTPUT_JSON, col_release, tol_source)
    except Exception:
        logger.exception("Failed to write output files")
        return False

    logger.info("Done.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(0 if build_bioclip_gbif_species_mapping() else 1)

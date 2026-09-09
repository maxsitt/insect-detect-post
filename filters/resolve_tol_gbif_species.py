"""Resolve BioCLIP's Tree of Life species list to GBIF backbone taxon keys.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

One-off, manually run script. Writes filters/tol_gbif_taxon_keys_<phylum>.csv.
Re-run only when the BioCLIP/TOL model/dataset version changes.

Resolves species names via individual GBIF API calls, using a ThreadPoolExecutor.

Scoped to PHYLUM_FILTER (insectdetect_post.constants); set to None to resolve the full
TOL species list. Kept taxonomic context columns (class, order, family, genus) allow
deriving a narrower filter later without further GBIF calls.

Functions:
    fetch_tol_species_records(): Fetch the TOL species list without loading the CLIP model.
    resolve_species_to_gbif():   Resolve a single TOL species name to a GBIF backbone taxon key.
    resolve_all_species():       Resolve all TOL species records to GBIF taxon keys concurrently.
    write_taxon_keys_csv():      Write resolved records to CSV, sorted by species name.
    load_previous_resolution():  Load existing CSV with resolved taxon keys for retry runs.
    resolve_tol_gbif_species():  Fetch, filter and resolve TOL to GBIF species, then write to CSV.
"""

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from bioclip._constants import BIOCLIP_MODEL_STR, HF_DATAFILE_REPO_TYPE, Rank
from bioclip.predict import create_classification_dict, get_tol_repo_id
from huggingface_hub import hf_hub_download
from pygbif import species as gbif_species
from requests.exceptions import HTTPError
from tqdm import tqdm

from insectdetect_post.constants import FILTERS_PATH, PHYLUM_FILTER

# Create module-level logger
logger = logging.getLogger(__name__)

# Output CSV path
_scope = PHYLUM_FILTER if PHYLUM_FILTER is not None else "all"
OUTPUT_CSV = FILTERS_PATH / f"tol_gbif_taxon_keys_{_scope}.csv"

# Fixed column order for the output CSV
COLUMN_ORDER = [
    "species", "genus", "family", "order", "class", "phylum", "kingdom",
    "gbif_taxon_key", "gbif_match_type", "gbif_rank", "gbif_status", "gbif_confidence",
    "gbif_canonical_name",
]

# Max workers for the GBIF API calls and retry/backoff parameters for transient failures
MAX_WORKERS = 64
MAX_RETRIES = 6
REQUEST_TIMEOUT_S = 10
MAX_BACKOFF_S = 20.0

# GBIF taxon ranks allowed to resolve to a taxon key (species or lower)
SPECIES_OR_LOWER_RANKS = frozenset({
    "SPECIES", "SUBSPECIES", "VARIETY", "SUBVARIETY", "FORM", "SUBFORM",
})


def fetch_tol_species_records() -> list[dict[str, str]]:
    """Fetch the TOL species list without loading the CLIP model.

    Downloads the same 'embeddings/txt_emb_species.json' file that
    TreeOfLifeClassifier.get_label_data() reads, then builds the same classification
    dicts via bioclip.predict.create_classification_dict(), without instantiating the
    full classifier (no CLIP model weights loaded, no torch needed).

    Returns:
        List of dicts, each with keys: kingdom, phylum, class, order, family, genus,
        species_epithet, species, common_name.
    """
    repo_id = get_tol_repo_id(BIOCLIP_MODEL_STR)
    txt_names_json = hf_hub_download(
        repo_id=repo_id, filename="embeddings/txt_emb_species.json", repo_type=HF_DATAFILE_REPO_TYPE
    )
    with open(txt_names_json, encoding="utf-8") as f:
        txt_names = json.load(f)
    return [create_classification_dict(names=name_ary, rank=Rank.SPECIES) for name_ary in txt_names]


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


def resolve_species_to_gbif(name: str, phylum: str | None = None) -> dict[str, object]:
    """Resolve a single TOL species name to a GBIF backbone taxon key.

    Queries GBIF's name match endpoint with strict=True, so a misspelling yields matchType
    'NONE' instead of silently matching a different species. strict=True also makes GBIF ignore
    any classification context, so `phylum` is only used to verify the result, never sent.

    For a synonym, gbif_taxon_key, gbif_rank and gbif_canonical_name describe the accepted taxon,
    while gbif_status keeps describing the matched name, 'SYNONYM' still flags the TOL name.

    A taxon key is only set at species rank or below and never for matchType 'HIGHERRANK', since
    GBIF answers a genus name with a placeholder usage such as 'Bombus spec' at species rank.

    Args:
        name: TOL species scientific name.
        phylum: Optional expected phylum, compared against the phylum GBIF returns.

    Returns:
        Dict with keys: gbif_taxon_key (int | None), gbif_match_type (str),
        gbif_rank (str | None), gbif_status (str | None), gbif_confidence (int | None),
        gbif_canonical_name (str | None).
    """
    result: dict | None = None
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            result = gbif_species.name_backbone(
                scientificName=name, taxonRank="species", strict=True, timeout=REQUEST_TIMEOUT_S
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

    diagnostics = result.get("diagnostics", {})
    match_type = diagnostics.get("matchType", "NONE")
    usage = result.get("usage")

    if usage is None:
        return {
            "gbif_taxon_key": None,
            "gbif_match_type": match_type,
            "gbif_rank": None,
            "gbif_status": None,
            "gbif_confidence": None,  # GBIF reports confidence 100 for a 'NONE' match
            "gbif_canonical_name": None,
        }

    # For a synonym, the accepted taxon is the one the taxon key should point at
    accepted_usage = result.get("acceptedUsage")
    resolved_usage = accepted_usage if result.get("synonym", False) and accepted_usage else usage

    resolved_rank = resolved_usage.get("rank")
    taxon_key = resolved_usage.get("key")
    if resolved_rank not in SPECIES_OR_LOWER_RANKS or match_type == "HIGHERRANK":
        taxon_key = None

    if phylum is not None and taxon_key is not None:
        gbif_phylum = next(
            (entry.get("name") for entry in result.get("classification", [])
             if entry.get("rank") == "PHYLUM"), None
        )
        if gbif_phylum is not None and gbif_phylum != phylum:
            logger.warning("'%s' resolved to GBIF phylum '%s', expected '%s' (taxon key %s)",
                           name, gbif_phylum, phylum, taxon_key)

    return {
        "gbif_taxon_key": int(taxon_key) if taxon_key is not None else None,
        "gbif_match_type": match_type,
        "gbif_rank": resolved_rank,
        "gbif_status": usage.get("status"),
        "gbif_confidence": diagnostics.get("confidence"),
        "gbif_canonical_name": resolved_usage.get("canonicalName"),
    }


def resolve_all_species(records: list[dict[str, str]]) -> list[dict[str, object]]:
    """Resolve all TOL species records to GBIF taxon keys concurrently.

    Runs resolve_species_to_gbif() over the unique species names using up to MAX_WORKERS
    threads, then merges each resolution back onto its source record. A species that still
    fails after MAX_RETRIES gets gbif_match_type='ERROR' instead of aborting the batch.

    Args:
        records: TOL species records as returned by fetch_tol_species_records().

    Returns:
        List of merged dicts (taxonomic context + gbif_* columns), one per input
        record, in input order.
    """
    phylum_by_name: dict[str, str] = {record["species"]: record["phylum"] for record in records}
    unique_names = sorted(phylum_by_name)
    resolved_by_name: dict[str, dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(resolve_species_to_gbif, name, phylum_by_name[name]): name
            for name in unique_names
        }
        for future in tqdm(as_completed(future_to_name), total=len(future_to_name),
                           desc="Resolving TOL species to GBIF taxon keys"):
            name = future_to_name[future]
            try:
                resolved_by_name[name] = future.result()
            except Exception:
                logger.debug("Failed to resolve '%s' to GBIF", name, exc_info=True)
                resolved_by_name[name] = {
                    "gbif_taxon_key": None,
                    "gbif_match_type": "ERROR",
                    "gbif_rank": None,
                    "gbif_status": None,
                    "gbif_confidence": None,
                    "gbif_canonical_name": None,
                }

    return [{**record, **resolved_by_name[record["species"]]} for record in records]


def write_taxon_keys_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write resolved records to CSV, sorted by species name.

    Args:
        rows: Merged records as returned by resolve_all_species().
        output_path: Destination CSV path. Parent directory is created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(COLUMN_ORDER).sort("species").unique(subset=["species"], keep="first")
    df.write_csv(output_path)


def load_previous_resolution(output_path: Path) -> pl.DataFrame | None:
    """Load existing CSV with resolved taxon keys for retry runs.

    Args:
        output_path: Path to the output CSV.

    Returns:
        The existing DataFrame, or None if output_path does not exist yet.
    """
    if not output_path.exists():
        return None
    return pl.read_csv(output_path).with_columns(pl.col("gbif_taxon_key").cast(pl.Int64, strict=False))


def resolve_tol_gbif_species() -> bool:
    """Fetch, filter and resolve TOL to GBIF species, then write to CSV.

    If output CSV already exists, only species with gbif_match_type == 'ERROR'
    are re-resolved; all other rows are kept as-is with no new GBIF calls.

    Returns:
        True on success (CSV written, or nothing to do), False on failure.
    """
    previous = load_previous_resolution(OUTPUT_CSV)
    keep_rows: list[dict[str, object]] = []

    if previous is not None:
        retry_df = previous.filter(pl.col("gbif_match_type") == "ERROR")
        keep_df = previous.filter(pl.col("gbif_match_type") != "ERROR")
        if retry_df.is_empty():
            logger.info("'%s' already exists with no gbif_match_type == 'ERROR' rows to retry. "
                        "Nothing to do.", OUTPUT_CSV)
            return True
        logger.info("Found existing '%s': retrying %d species with gbif_match_type == 'ERROR', "
                    "keeping %d other rows as-is.", OUTPUT_CSV, retry_df.height, keep_df.height)
        records = retry_df.select(
            ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
        ).to_dicts()
        keep_rows = keep_df.to_dicts()
    else:
        logger.info("Fetching Tree of Life species list from Hugging Face...")
        try:
            records = fetch_tol_species_records()
        except Exception:
            logger.exception("Failed to fetch TOL species list")
            return False
        logger.info("Fetched %d species records.", len(records))

        if PHYLUM_FILTER is not None:
            records = [record for record in records if record["phylum"] == PHYLUM_FILTER]
            logger.info("Filtered to phylum '%s': %d species records.", PHYLUM_FILTER, len(records))

    logger.info("Resolving species to GBIF taxon keys using %d workers...", MAX_WORKERS)
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
    matched = sum(1 for row in resolved if row["gbif_taxon_key"] is not None)
    unmatched = [row["species"] for row in resolved if row["gbif_match_type"] == "NONE"]
    higher_rank_only = [row["species"] for row in resolved if row["gbif_match_type"] == "HIGHERRANK"]
    failed = [row["species"] for row in resolved if row["gbif_match_type"] == "ERROR"]
    logger.info("Resolved %d/%d species to a GBIF taxon key.", matched, len(resolved))

    preview_limit = 20
    if unmatched:
        logger.warning("%d species had no GBIF match (review manually):", len(unmatched))
        for name in unmatched[:preview_limit]:
            logger.warning("  - %s", name)
        if len(unmatched) > preview_limit:
            logger.warning("  ... and %d more (see gbif_match_type == 'NONE' rows in the output CSV)",
                            len(unmatched) - preview_limit)
    if higher_rank_only:
        logger.warning("%d species only matched at a higher rank than species (genus/family/etc.) -- "
                       "gbif_taxon_key left empty:", len(higher_rank_only))
        for name in higher_rank_only[:preview_limit]:
            logger.warning("  - %s", name)
        if len(higher_rank_only) > preview_limit:
            logger.warning("  ... and %d more (see gbif_match_type == 'HIGHERRANK' rows in the output CSV)",
                           len(higher_rank_only) - preview_limit)
    if failed:
        logger.warning("%d species failed to resolve after retries (network/rate-limit issues -- "
                       "re-run the script to retry just these):", len(failed))
        for name in failed[:preview_limit]:
            logger.warning("  - %s", name)
        if len(failed) > preview_limit:
            logger.warning("  ... and %d more (see gbif_match_type == 'ERROR' rows in the output CSV)",
                           len(failed) - preview_limit)

    logger.info("Writing '%s'...", OUTPUT_CSV)
    try:
        write_taxon_keys_csv(resolved, OUTPUT_CSV)
    except Exception:
        logger.exception("Failed to write output CSV")
        return False

    logger.info("Done.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(0 if resolve_tol_gbif_species() else 1)

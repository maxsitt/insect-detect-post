"""Build per-region GBIF region-filter CSVs consumable by TreeOfLifeClassifier (pybioclip).

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Resolves which BioCLIP Tree of Life species have GBIF occurrence records in a given
region, by combining a BioCLIP-to-GBIF species mapping with GBIF's occurrence search
API. A region is either a country (ISO 3166-1 alpha-2 code) or a GBIF continent, which
GBIF assigns differently: a country is the one the publisher supplied (or derived from
coordinates), coastal waters included, while a continent covers land only and is left
blank for marine records and for records without coordinates from a country spanning two
continents (boundaries: https://github.com/gbif/continents).

Taxon keys are those of GBIF_CHECKLIST_KEY (Catalogue of Life Extended Release, COL XR):
alphanumeric strings, some of them digits only, so they are never converted to integers.

Results are cached per region so repeat runs need no network access, alongside a JSON
provenance record so a cache built from an outdated mapping is rebuilt automatically.

Functions:
    load_species_mapping():            Load the BioCLIP-to-GBIF species mapping, downloading it on first use.
    load_species_mapping_provenance(): Load the provenance record of the species mapping.
    fetch_col_release():               Fetch the COL release GBIF currently matches names against.
    format_col_release():              Return a short label of a COL release record for log messages.
    get_region_species_counts():       Fetch all GBIF species keys with occurrence records in a region.
    region_filter_path():              Return the path of a region's cached species-list CSV.
    build_region_filter_csv():         Resolve a species-list CSV for a region from cache or by querying GBIF.
    load_region_species():             Load species with GBIF occurrence records in any of the given regions.
    copy_region_filters():             Copy the region filters and the mapping provenance into a run.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
import requests
from pygbif import occurrences
from requests.exceptions import HTTPError, RequestException

from insectdetect_post.asset_manager import compute_sha256, ensure_asset
from insectdetect_post.constants import (
    FILTER_ASSETS_JSON,
    FILTERS_PATH,
    GBIF_CHECKLIST_KEY,
    MIN_OCCURRENCE_COUNT,
    PHYLUM_FILTER,
    PHYLUM_TAXON_KEYS,
)

# Create module-level logger
logger = logging.getLogger(__name__)

# BioCLIP-to-GBIF species mapping and provenance record, produced by filters/build_bioclip_gbif_species_mapping.py
_scope = PHYLUM_FILTER if PHYLUM_FILTER is not None else "all"
SPECIES_MAPPING_CSV = FILTERS_PATH / f"bioclip_gbif_species_mapping_{_scope}.csv"
SPECIES_MAPPING_JSON = SPECIES_MAPPING_CSV.with_suffix(".json")

# COL XR taxon key to restrict occurrence facet queries to
PHYLUM_TAXON_KEY: str | None = None
if PHYLUM_FILTER is not None:
    PHYLUM_TAXON_KEY = PHYLUM_TAXON_KEYS.get(PHYLUM_FILTER)
    if PHYLUM_TAXON_KEY is None:
        logger.warning(
            "PHYLUM_FILTER '%s' has no entry in PHYLUM_TAXON_KEYS -- occurrence facet "
            "queries will not be restricted by phylum.", PHYLUM_FILTER
        )

# Kinds of region, named after the GBIF occurrence search parameter each one is sent as
RegionType = Literal["country", "continent"]

# GBIF occurrence facet query settings
FACET_LIMIT = 1_200_000
MAX_RETRIES = 4
REQUEST_TIMEOUT_S = 120
MAX_BACKOFF_S = 20.0

# Age at which a cached region filter is reported as outdated
CACHE_MAX_AGE_DAYS = 90

# Endpoints naming the COL release behind GBIF_CHECKLIST_KEY: GBIF's name matching service
# reports the ChecklistBank dataset it currently uses, ChecklistBank its issue date and DOI
GBIF_MATCH_METADATA_URL = "https://api.gbif.org/v2/species/match/metadata"
CHECKLISTBANK_DATASET_URL = "https://api.checklistbank.org/dataset/{}"
RELEASE_TIMEOUT_S = 15

# Request headers for direct GBIF/ChecklistBank API calls (not made through pygbif)
REQUEST_HEADERS = {"User-Agent": "insect-detect-post (https://github.com/maxsitt/insect-detect-post)"}


def load_species_mapping(
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pl.DataFrame:
    """Load the BioCLIP-to-GBIF species mapping, downloading it on first use.

    Args:
        progress_callback: Optional callback(current, total, message) for download progress.

    Raises:
        KeyError: If the mapping CSV is not a registered asset.
        ValueError: If the downloaded file fails checksum verification.

    Returns:
        DataFrame with every column of the mapping CSV (COLUMN_ORDER in
        filters/build_bioclip_gbif_species_mapping.py), 'gbif_species_key' as a nullable String.
    """
    ensure_asset(SPECIES_MAPPING_CSV.name, FILTER_ASSETS_JSON, progress_callback=progress_callback)
    return pl.read_csv(SPECIES_MAPPING_CSV, schema_overrides={"gbif_species_key": pl.String})


def load_species_mapping_provenance() -> dict:
    """Load the provenance record of the species mapping, downloading it on first use.

    Raises:
        KeyError: If the provenance JSON is not a registered asset.
        ValueError: If the downloaded file fails checksum verification or is not valid JSON.

    Returns:
        Provenance dict as written by filters/build_bioclip_gbif_species_mapping.py, including
        'col_release' (see fetch_col_release()).
    """
    ensure_asset(SPECIES_MAPPING_JSON.name, FILTER_ASSETS_JSON)
    return json.loads(SPECIES_MAPPING_JSON.read_text(encoding="utf-8"))


def fetch_col_release(timeout: float = RELEASE_TIMEOUT_S) -> dict[str, str | None]:
    """Fetch the COL release GBIF currently matches names against.

    GBIF moves GBIF_CHECKLIST_KEY to each new COL XR release, so the key alone does not
    identify which release produced a taxon key. Single attempt, no retries.

    Raises:
        RequestException: If either API request fails.
        KeyError: If GBIF's response does not name the ChecklistBank dataset.

    Returns:
        Dict with keys: checklist_key, clb_dataset_key, alias (e.g. 'COL26.6 XR'),
        issued (ISO date), doi.
    """
    response = requests.get(
        GBIF_MATCH_METADATA_URL, params={"checklistKey": GBIF_CHECKLIST_KEY},
        headers=REQUEST_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    main_index = response.json()["mainIndex"]
    clb_dataset_key = str(main_index["clbDatasetKey"])

    response = requests.get(
        CHECKLISTBANK_DATASET_URL.format(clb_dataset_key), headers=REQUEST_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    dataset = response.json()
    return {
        "checklist_key": GBIF_CHECKLIST_KEY,
        "clb_dataset_key": clb_dataset_key,
        "alias": main_index.get("datasetAlias") or dataset.get("alias"),
        "issued": dataset.get("issued"),
        "doi": dataset.get("doi"),
    }


def format_col_release(release: dict | None) -> str:
    """Return a short label of a COL release record for log messages, e.g. 'COL26.6 XR (2026-06-19)'."""
    if not release:
        return "unknown"
    alias = release.get("alias") or f"ChecklistBank dataset {release.get('clb_dataset_key', '?')}"
    return f"{alias} ({release['issued']})" if release.get("issued") else alias


def _cache_stale_reason(meta_path: Path) -> str | None:
    """Return why a cached region filter should be rebuilt, or None if it is still valid.

    The cache is keyed by region and min_occurrence_count through its filename, so the
    remaining input that can change underneath it is the species mapping. A cache
    written before provenance was recorded has no way to prove which mapping produced it
    and is therefore treated as stale once.
    """
    if not meta_path.exists():
        return "it has no provenance record, so the mapping that produced it is unknown"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"its provenance record '{meta_path.name}' could not be read"

    if not SPECIES_MAPPING_CSV.exists():
        logger.debug("Species mapping is not present locally, keeping the cached region filter.")
        return None

    if meta.get("mapping_sha256") != compute_sha256(SPECIES_MAPPING_CSV):
        return "the species mapping has changed since it was built"

    built = meta.get("built_utc")
    if isinstance(built, str):
        try:
            age = datetime.now(UTC) - datetime.fromisoformat(built)
        except ValueError:
            age = timedelta(0)
        if age > timedelta(days=CACHE_MAX_AGE_DAYS):
            logger.warning(
                "Cached region filter is %d days old: GBIF may have added new species since "
                "it was built. Delete '%s' to rebuild it automatically at the next run.",
                age.days, meta_path.with_suffix(".csv")
            )
    return None


def _write_cache_metadata(
    meta_path: Path,
    region_type: RegionType,
    region: str,
    min_occurrence_count: int,
    mapping_sha256: str,
    mapping_col_release: dict | None,
    gbif_col_release: dict | None,
    facet_entries: int,
    species: int
) -> None:
    """Record which inputs produced a cached region filter, alongside the CSV."""
    meta_path.write_text(
        json.dumps({
            "region_type": region_type,
            "region": region,
            "phylum_filter": _scope,
            "min_occurrence_count": min_occurrence_count,
            "gbif_checklist_key": GBIF_CHECKLIST_KEY,
            "mapping_sha256": mapping_sha256,
            "mapping_col_release": mapping_col_release,
            "gbif_col_release": gbif_col_release,
            "gbif_facet_entries": facet_entries,
            "species": species,
            "built_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }, indent=2) + "\n",
        encoding="utf-8"
    )


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


def _fetch_facet(region_type: RegionType, region: str, min_occurrence_count: int) -> dict:
    """Fetch the complete GBIF 'speciesKey' occurrence facet for a region in one request.

    Retries transient failures (429/5xx or network errors) up to MAX_RETRIES times.
    A 4xx error other than 429 is treated as an invalid region code and raised
    immediately as a ValueError, not retried.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return occurrences.search(
                country=region if region_type == "country" else None,
                continent=region if region_type == "continent" else None,
                checklistKey=GBIF_CHECKLIST_KEY, phylumKey=PHYLUM_TAXON_KEY, facet="speciesKey", limit=0,
                speciesKey_facetLimit=FACET_LIMIT, speciesKey_facetOffset=0,
                facetMincount=min_occurrence_count,
                timeout=REQUEST_TIMEOUT_S,
            )
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                expected = ("a valid ISO 3166-1 alpha-2 code" if region_type == "country"
                            else "one of GBIF's continents")
                raise ValueError(
                    f"GBIF rejected {region_type} '{region}' (HTTP {status}). "
                    f"Verify it is {expected}."
                ) from e
            last_error = e
        except RequestException as e:
            last_error = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(_backoff_seconds(last_error, attempt))
    raise RuntimeError(
        f"Failed to fetch GBIF occurrence facet for {region_type} '{region}' "
        f"after {MAX_RETRIES} attempts: {last_error}"
    )


def get_region_species_counts(region_type: RegionType, region: str, min_occurrence_count: int) -> dict[str, int]:
    """Fetch all GBIF species keys with occurrence records in a region.

    Requests GBIF's complete 'speciesKey' occurrence facet in one call, restricted server-side
    to PHYLUM_TAXON_KEY if set. The facet holds the accepted species of each record's lineage, so
    records identified to a subspecies or by a synonym count for their accepted species. The
    mapping's keys are accepted species keys for that reason: a synonym or subspecies key would
    never appear in this facet (unlike in the 'taxonKey' facet, which holds every key of a lineage).

    Args:
        region_type: GBIF occurrence search parameter the region is passed as.
        region: ISO 3166-1 alpha-2 country code or GBIF continent, matching region_type.
        min_occurrence_count: Minimum occurrence records for a taxon to be included.

    Raises:
        ValueError: If GBIF rejects the region code as invalid.
        RuntimeError: If fetching fails after exhausting retries, or if the facet fills
            FACET_LIMIT and is therefore truncated.

    Returns:
        Mapping of COL XR species key (string) to its occurrence record count in the region.
    """
    facet = _fetch_facet(region_type, region, min_occurrence_count)
    counts = facet["facets"][0]["counts"] if facet.get("facets") else []
    if len(counts) >= FACET_LIMIT:
        raise RuntimeError(
            f"GBIF occurrence facet for {region_type} '{region}' returned {len(counts)} entries, "
            f"filling FACET_LIMIT ({FACET_LIMIT}) -- the result is truncated."
        )
    logger.debug("%s '%s': fetched %d facet entries", region_type.capitalize(), region, len(counts))
    return {str(entry["name"]): int(entry["count"]) for entry in counts}


def region_filter_path(region: str, min_occurrence_count: int) -> Path:
    """Return the path of a region's cached species-list CSV (its JSON record has the same stem)."""
    return FILTERS_PATH / f"region_filter_{_scope}_{region.upper()}_minocc{min_occurrence_count}.csv"


def build_region_filter_csv(
    region_type: RegionType,
    region: str,
    min_occurrence_count: int,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Resolve a species-list CSV for a region from cache or by querying GBIF.

    The cached CSV leads with a 'species' column (the TOL name), so it can be handed directly to
    pybioclip, followed by 'gbif_species', 'gbif_species_key' and 'gbif_occurrence_count'. It is
    kept lean on purpose: further details of a species are in the species mapping, joined on
    'gbif_species_key'. A JSON file of the same name records which mapping and settings produced
    it; the cache is rebuilt automatically if the species mapping has changed since. If
    rebuilding fails while an existing cache is present (typically an offline run), the outdated
    cache is used and a warning is logged.

    'gbif_occurrence_count' is the number of records GBIF holds for the species in that region,
    so TOL species that are synonyms of each other share a key and repeat the same count.

    Args:
        region_type: GBIF occurrence search parameter the region is passed as.
        region: ISO 3166-1 alpha-2 country code or GBIF continent, matching region_type.
        min_occurrence_count: Minimum occurrence records for a taxon to be included.
        force_refresh: If True, re-query GBIF even if a valid cached CSV already exists.
        progress_callback: Optional callback(current, total, message), reporting 0-100% of this
            function's own work. Callers embedding it in a wider range should scale accordingly.

    Raises:
        KeyError: If the species mapping is not a registered asset.
        ValueError: If GBIF rejects the region code, or no TOL species in the mapping have
                    any occurrence record in the given region.
        RuntimeError: If fetching from GBIF fails after exhausting retries and no cached
                      CSV exists to fall back on.

    Returns:
        Path to the (now-cached) per-region species-list CSV.
    """
    region = region.upper()
    label = f"{region_type} '{region}'"
    cache_path = region_filter_path(region, min_occurrence_count)
    meta_path = cache_path.with_suffix(".json")
    cache_exists = cache_path.exists()

    if cache_exists and not force_refresh:
        stale_reason = _cache_stale_reason(meta_path)
        if stale_reason is None:
            logger.info("Using cached region filter for %s: '%s'", label, cache_path)
            return cache_path
        logger.info("Rebuilding region filter for %s: %s.", label, stale_reason)

    if progress_callback:
        progress_callback(
            0, 100,
            f"Building region filter for {label} from GBIF occurrence data "
            "-- this can take a few minutes..."
        )

    # Mapping download and GBIF query both report raw 0-100%, so each gets its own sub-range
    def download_progress(pct: int, _total: int, message: str) -> None:
        if progress_callback:
            progress_callback(int(pct / 10), 100, message)  # 0-10%

    try:
        mapping = load_species_mapping(download_progress)
        mapping_sha256 = compute_sha256(SPECIES_MAPPING_CSV)

        if progress_callback:
            progress_callback(
                10, 100, f"Querying GBIF for {_scope} taxa recorded in {label}..."
            )
        logger.info("Querying GBIF for %s taxa recorded in %s...", _scope, label)

        region_species_counts = get_region_species_counts(region_type, region, min_occurrence_count)
    except (OSError, RuntimeError, RequestException) as e:
        if not cache_exists:
            raise
        logger.warning(
            "Could not rebuild the region filter for %s (%s). Falling back to the existing "
            "cached filter, which may be outdated: '%s'", label, e, cache_path
        )
        return cache_path

    logger.info("%s: %d GBIF species with >=%d occurrence records",
                label[0].upper() + label[1:], len(region_species_counts), min_occurrence_count)

    # COL releases are informational only: a mapping on an older release still works, but taxa
    # that COL has split, merged or renamed since then can be missing from the region filter
    try:
        mapping_col_release = load_species_mapping_provenance().get("col_release")
    except (OSError, KeyError, ValueError) as e:
        logger.warning("Could not load the provenance record of the species mapping: %s", e)
        mapping_col_release = None
    try:
        gbif_col_release = fetch_col_release()
    except (RequestException, KeyError, ValueError) as e:
        logger.warning("Could not fetch the COL release GBIF currently uses: %s", e)
        gbif_col_release = None
    same = (mapping_col_release is not None and gbif_col_release is not None
            and mapping_col_release.get("clb_dataset_key") == gbif_col_release.get("clb_dataset_key"))
    logger.info("COL release of GBIF occurrence data: %s, of the species mapping: %s%s",
                format_col_release(gbif_col_release), format_col_release(mapping_col_release),
                " (same release)" if same else "")

    if progress_callback:
        progress_callback(90, 100, f"Matching species for {label}...")

    counts = pl.DataFrame(
        {
            "gbif_species_key": list(region_species_counts.keys()),
            "gbif_occurrence_count": list(region_species_counts.values()),
        },
        schema={"gbif_species_key": pl.String, "gbif_occurrence_count": pl.Int64},
    )
    matched = (
        mapping.join(counts, on="gbif_species_key", how="inner")
        .select(pl.col("bioclip_species").alias("species"), "gbif_species", "gbif_species_key",
                "gbif_occurrence_count")
        .unique(subset=["species"], keep="first")
        .sort("species")
    )
    if matched.is_empty():
        raise ValueError(
            f"No TOL species in the mapping have any GBIF occurrence record in {label}."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matched.write_csv(cache_path)
    _write_cache_metadata(
        meta_path, region_type, region, min_occurrence_count, mapping_sha256,
        mapping_col_release, gbif_col_release, len(region_species_counts), matched.height
    )
    logger.info("Built region filter for %s: %d of them match a BioCLIP species -> '%s'",
                label, matched.height, cache_path)

    if progress_callback:
        progress_callback(100, 100, f"Built region filter for {label}: {matched.height} species")

    return cache_path


def load_region_species(
    region_type: RegionType,
    regions: Sequence[str],
    min_occurrence_count: int = MIN_OCCURRENCE_COUNT,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Load species with GBIF occurrence records in any of the given regions.

    Resolves each region separately, so every region keeps its own cache, then
    merges the results: a species is included if it occurs in at least one of them.

    Args:
        region_type: GBIF occurrence search parameter the regions are passed as.
        regions: ISO 3166-1 alpha-2 country codes or GBIF continents, matching region_type.
        min_occurrence_count: Minimum occurrence records for a taxon to be included.
        force_refresh: If True, re-query GBIF even if valid cached CSVs already exist.
        progress_callback: Optional callback(current, total, message), reporting 0-100% across
            all regions. Callers embedding it in a wider range should scale accordingly.

    Raises:
        KeyError: If the species mapping is not a registered asset.
        ValueError: If GBIF rejects a region code, or a region has no matching species.
        RuntimeError: If fetching from GBIF fails and no cached CSV exists to fall back on.

    Returns:
        Sorted species names recorded in at least one of the regions.
    """
    species: set[str] = set()
    total = len(regions)
    for index, region in enumerate(regions):
        def region_progress(pct: int, _total: int, message: str, index: int = index) -> None:
            """Wrapper callback that maps one region's progress into its own slice."""
            if progress_callback:
                progress_callback(int((index * 100 + pct) / total), 100, message)

        cache_path = build_region_filter_csv(
            region_type, region, min_occurrence_count, force_refresh, region_progress
        )
        species.update(pl.read_csv(cache_path, columns=["species"])["species"].to_list())

    logger.info("Region filter for %s %s: %d unique species across all of them",
                region_type, ", ".join(regions), len(species))
    return sorted(species)


def copy_region_filters(
    regions: Sequence[str],
    dest_dir: Path,
    min_occurrence_count: int = MIN_OCCURRENCE_COUNT,
) -> list[Path]:
    """Copy the region filters and the mapping provenance into a run.

    Copies each region's species-list CSV and its JSON record, as used by load_region_species(),
    plus the species mapping's provenance JSON. The mapping CSV itself is not copied: the
    provenance identifies it by its checksum (the region JSONs' 'mapping_sha256'). Files that do
    not exist are skipped.

    Args:
        regions: ISO 3166-1 alpha-2 country codes or GBIF continents of the applied filter.
        dest_dir: Destination directory, created if missing (e.g. OutputLayout.filters_dir).
        min_occurrence_count: Minimum occurrence count the region filters were built with.

    Raises:
        OSError: If a file cannot be copied.

    Returns:
        Paths of the copied files.
    """
    sources: list[Path] = []
    for region in regions:
        csv_path = region_filter_path(region, min_occurrence_count)
        sources += [csv_path, csv_path.with_suffix(".json")]
    try:
        sources.append(ensure_asset(SPECIES_MAPPING_JSON.name, FILTER_ASSETS_JSON))
    except (OSError, KeyError, ValueError) as e:
        logger.warning("Could not provide the provenance record of the species mapping: %s", e)

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sources:
        if source.exists():
            copied.append(Path(shutil.copy2(source, dest_dir / source.name)))
        else:
            logger.warning("Filter file '%s' not found, not copied to the run.", source.name)
    return copied

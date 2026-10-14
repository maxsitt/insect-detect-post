"""Build per-region GBIF region-filter CSVs consumable by TreeOfLifeClassifier (pybioclip).

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Resolves which BioCLIP Tree of Life species have GBIF occurrence records in a given
region, by combining a TOL-to-GBIF taxon key mapping with GBIF's occurrence search
API. A region is either a country (ISO 3166-1 alpha-2 code) or a GBIF continent, which
GBIF assigns differently: a country is the one the publisher supplied (or derived from
coordinates), coastal waters included, while a continent covers land only and is left
blank for marine records and for records without coordinates from a country spanning two
continents (boundaries: https://github.com/gbif/continents).

Results are cached per region so repeat runs need no network access, alongside a JSON
provenance record so a cache built from an outdated mapping is rebuilt automatically.

Functions:
    load_tol_gbif_taxon_keys(): Load the TOL-to-GBIF taxon key mapping, downloading it on first use.
    get_region_taxon_counts():  Fetch all GBIF taxon keys with occurrence records in a region.
    build_region_filter_csv():  Resolve a species-list CSV for a region from cache or by querying GBIF.
    load_region_species():      Load species with GBIF occurrence records in any of the given regions.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
from pygbif import occurrences
from requests.exceptions import HTTPError, RequestException

from insectdetect_post.asset_manager import compute_sha256, ensure_asset
from insectdetect_post.constants import (
    FILTER_ASSETS_JSON,
    FILTERS_PATH,
    MIN_OCCURRENCE_COUNT,
    PHYLUM_FILTER,
    PHYLUM_TAXON_KEYS,
)

# Create module-level logger
logger = logging.getLogger(__name__)

# TOL-to-GBIF taxon key mapping, produced by filters/resolve_tol_gbif_species.py
_scope = PHYLUM_FILTER if PHYLUM_FILTER is not None else "all"
TOL_GBIF_TAXON_KEYS_CSV = FILTERS_PATH / f"tol_gbif_taxon_keys_{_scope}.csv"

# GBIF backbone taxonKey to restrict occurrence facet queries to
PHYLUM_TAXON_KEY: int | None = None
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


def load_tol_gbif_taxon_keys(
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pl.DataFrame:
    """Load the TOL-to-GBIF taxon key mapping, downloading it on first use.

    Args:
        progress_callback: Optional callback(current, total, message) for download progress.

    Raises:
        KeyError: If the mapping CSV is not a registered asset.
        ValueError: If the downloaded file fails checksum verification.

    Returns:
        DataFrame with every column of the mapping CSV (COLUMN_ORDER in
        filters/resolve_tol_gbif_species.py), 'gbif_taxon_key' as a nullable Int64.
    """
    ensure_asset(TOL_GBIF_TAXON_KEYS_CSV.name, FILTER_ASSETS_JSON, progress_callback=progress_callback)
    return pl.read_csv(TOL_GBIF_TAXON_KEYS_CSV, schema_overrides={"gbif_taxon_key": pl.Int64})


def _cache_stale_reason(meta_path: Path) -> str | None:
    """Return why a cached region filter should be rebuilt, or None if it is still valid.

    The cache is keyed by region and min_occurrence_count through its filename, so the
    remaining input that can change underneath it is the TOL-to-GBIF mapping. A cache
    written before provenance was recorded has no way to prove which mapping produced it
    and is therefore treated as stale once.
    """
    if not meta_path.exists():
        return "it has no provenance record, so the mapping that produced it is unknown"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"its provenance record '{meta_path.name}' could not be read"

    if not TOL_GBIF_TAXON_KEYS_CSV.exists():
        logger.debug("TOL-to-GBIF mapping is not present locally, keeping the cached region filter.")
        return None

    if meta.get("mapping_sha256") != compute_sha256(TOL_GBIF_TAXON_KEYS_CSV):
        return "the TOL-to-GBIF mapping has changed since it was built"

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
            "mapping_sha256": mapping_sha256,
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
    """Fetch the complete GBIF 'taxonKey' occurrence facet for a region in one request.

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
                phylumKey=PHYLUM_TAXON_KEY, facet="taxonKey", limit=0,
                taxonKey_facetLimit=FACET_LIMIT, taxonKey_facetOffset=0,
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


def get_region_taxon_counts(region_type: RegionType, region: str, min_occurrence_count: int) -> dict[int, int]:
    """Fetch all GBIF taxon keys with occurrence records in a region.

    Requests GBIF's complete 'taxonKey' occurrence facet in one call, restricted server-side
    to PHYLUM_TAXON_KEY if set. The facet still blends every taxonomic rank together within
    that scope, so it must be intersected against known species-level keys to be meaningful.

    Args:
        region_type: GBIF occurrence search parameter the region is passed as.
        region: ISO 3166-1 alpha-2 country code or GBIF continent, matching region_type.
        min_occurrence_count: Minimum occurrence records for a taxon to be included.

    Raises:
        ValueError: If GBIF rejects the region code as invalid.
        RuntimeError: If fetching fails after exhausting retries, or if the facet fills
            FACET_LIMIT and is therefore truncated.

    Returns:
        Mapping of GBIF taxon key to its occurrence record count in the region.
    """
    facet = _fetch_facet(region_type, region, min_occurrence_count)
    counts = facet["facets"][0]["counts"] if facet.get("facets") else []
    if len(counts) >= FACET_LIMIT:
        raise RuntimeError(
            f"GBIF occurrence facet for {region_type} '{region}' returned {len(counts)} entries, "
            f"filling FACET_LIMIT ({FACET_LIMIT}) -- the result is truncated."
        )
    logger.debug("%s '%s': fetched %d facet entries", region_type.capitalize(), region, len(counts))
    return {int(entry["name"]): int(entry["count"]) for entry in counts}


def build_region_filter_csv(
    region_type: RegionType,
    region: str,
    min_occurrence_count: int,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Resolve a species-list CSV for a region from cache or by querying GBIF.

    The cached CSV leads with a 'species' column, so it can be handed directly to pybioclip,
    followed by the TOL taxonomic context and the GBIF columns for inspection. A JSON file of
    the same name records which mapping and settings produced it; the cache is rebuilt
    automatically if the TOL-to-GBIF mapping has changed since. If rebuilding fails while an
    existing cache is present (typically an offline run), the outdated cache is used and a
    warning is logged.

    'gbif_occurrence_count' is the number of records GBIF holds for the taxon in that region,
    so TOL species that are synonyms of each other share a key and repeat the same count.

    Args:
        region_type: GBIF occurrence search parameter the region is passed as.
        region: ISO 3166-1 alpha-2 country code or GBIF continent, matching region_type.
        min_occurrence_count: Minimum occurrence records for a taxon to be included.
        force_refresh: If True, re-query GBIF even if a valid cached CSV already exists.
        progress_callback: Optional callback(current, total, message), reporting 0-100% of this
            function's own work. Callers embedding it in a wider range should scale accordingly.

    Raises:
        KeyError: If the TOL-to-GBIF mapping is not a registered asset.
        ValueError: If GBIF rejects the region code, or no TOL species in the mapping have
                    any occurrence record in the given region.
        RuntimeError: If fetching from GBIF fails after exhausting retries and no cached
                      CSV exists to fall back on.

    Returns:
        Path to the (now-cached) per-region species-list CSV.
    """
    region = region.upper()
    label = f"{region_type} '{region}'"
    cache_path = FILTERS_PATH / f"tol_gbif_species_{_scope}_minocc{min_occurrence_count}_{region}.csv"
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
        mapping = load_tol_gbif_taxon_keys(download_progress)
        mapping_sha256 = compute_sha256(TOL_GBIF_TAXON_KEYS_CSV)

        if progress_callback:
            progress_callback(
                10, 100, f"Querying GBIF for {_scope} taxa recorded in {label}..."
            )
        logger.info("Querying GBIF for %s taxa recorded in %s...", _scope, label)

        region_taxon_counts = get_region_taxon_counts(region_type, region, min_occurrence_count)
    except (OSError, RuntimeError, RequestException) as e:
        if not cache_exists:
            raise
        logger.warning(
            "Could not rebuild the region filter for %s (%s). Falling back to the existing "
            "cached filter, which may be outdated: '%s'", label, e, cache_path
        )
        return cache_path

    logger.info("%s: %d GBIF taxa at any rank with >=%d occurrence records",
                label[0].upper() + label[1:], len(region_taxon_counts), min_occurrence_count)

    if progress_callback:
        progress_callback(90, 100, f"Matching species for {label}...")

    counts = pl.DataFrame(
        {
            "gbif_taxon_key": list(region_taxon_counts.keys()),
            "gbif_occurrence_count": list(region_taxon_counts.values()),
        },
        schema={"gbif_taxon_key": pl.Int64, "gbif_occurrence_count": pl.Int64},
    )
    matched = (
        mapping.join(counts, on="gbif_taxon_key", how="inner")
        .select("species", "genus", "family", "order", "class",
                "gbif_taxon_key", "gbif_canonical_name", "gbif_occurrence_count")
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
        meta_path, region_type, region, min_occurrence_count,
        mapping_sha256, len(region_taxon_counts), matched.height
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
        KeyError: If the TOL-to-GBIF mapping is not a registered asset.
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

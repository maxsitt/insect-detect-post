"""Resolve all ISO 3166-1 alpha-2 country codes recognized by the GBIF API.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

One-off, manually run script. Writes filters/gbif_country_codes.csv.
Re-run required only when GBIF's country enumeration changes.

Functions:
    fetch_gbif_country_codes():   Fetch GBIF's country enumeration, with retry on failures.
    write_country_codes_csv():    Write resolved country codes to CSV, sorted by country code.
    resolve_gbif_country_codes(): Fetch GBIF's country enumeration and write the mapping CSV.
"""

import logging
import random
import sys
import time
from pathlib import Path

import polars as pl
import requests
from requests.exceptions import RequestException

from insectdetect_post.constants import FILTERS_PATH

# Create module-level logger
logger = logging.getLogger(__name__)

# Output CSV path
OUTPUT_CSV = FILTERS_PATH / "gbif_country_codes.csv"

# GBIF's country vocabulary endpoint
GBIF_COUNTRY_ENUM_URL = "https://api.gbif.org/v1/enumeration/country"

# Request headers for GBIF API calls
REQUEST_HEADERS = {"User-Agent": "insect-detect-post (https://github.com/maxsitt/insect-detect-post)"}

# Retry/backoff parameters for transient failures
MAX_RETRIES = 4
REQUEST_TIMEOUT_S = 15
MAX_BACKOFF_S = 20.0


def fetch_gbif_country_codes() -> list[dict[str, str]]:
    """Fetch GBIF's country enumeration, with retry on failures.

    Returns:
        List of dicts with keys: iso2, name.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                GBIF_COUNTRY_ENUM_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
            entries = response.json()
            return [{"iso2": entry["iso2"], "name": entry["title"]} for entry in entries]
        except RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                capped_wait = min(2 ** attempt, MAX_BACKOFF_S)
                time.sleep(random.uniform(0, capped_wait))
    raise RuntimeError(f"Failed to fetch GBIF country enumeration after {MAX_RETRIES} attempts: {last_error}")


def write_country_codes_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write resolved country codes to CSV, sorted by country code.

    Args:
        rows: Records as returned by fetch_gbif_country_codes().
        output_path: Destination CSV path. Parent directory is created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(["iso2", "name"]).sort("iso2")
    df.write_csv(output_path)


def resolve_gbif_country_codes() -> bool:
    """Fetch GBIF's country enumeration and write the mapping CSV.

    Returns:
        True on success, False on failure.
    """
    logger.info("Fetching GBIF country enumeration from '%s'...", GBIF_COUNTRY_ENUM_URL)
    try:
        rows = fetch_gbif_country_codes()
    except Exception:
        logger.exception("Failed to fetch GBIF country enumeration")
        return False
    logger.info("Fetched %d country codes.", len(rows))

    logger.info("Writing '%s'...", OUTPUT_CSV)
    try:
        write_country_codes_csv(rows, OUTPUT_CSV)
    except Exception:
        logger.exception("Failed to write output CSV")
        return False

    logger.info("Done.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(0 if resolve_gbif_country_codes() else 1)

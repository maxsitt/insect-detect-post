"""Taxon names for the validation dropdowns (BioCLIP Tree of Life matched against GBIF).

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Built from 'filters/bioclip_gbif_species_mapping_Arthropoda.csv': species with a GBIF species key,
named by their GBIF species name (the canonical name), higher ranks from the Tree of Life.
Taxa predicted in the run and free-text entries are added on top.

Functions:
    file_provenance(): Name, size and SHA-256 checksum of a file.
    rank_index(): Position of a rank in VALIDATION_RANKS.
    ranks_up_to(): Ranks from class down to a rank.
    ranks_below(): Ranks deeper than a rank.

Classes:
    TaxonomyIndex: Per-rank name lists, parent -> children maps and lineage lookup.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import polars as pl

from insectdetect_post.asset_manager import compute_sha256, ensure_asset
from insectdetect_post.build_region_filter import SPECIES_MAPPING_CSV, load_species_mapping
from insectdetect_post.constants import FILTER_ASSETS_JSON, RANK_ORDER

# Create module-level logger
logger = logging.getLogger(__name__)

# Ranks a validator can assign: class to species (kingdom and phylum are implied for arthropods)
VALIDATION_RANKS = RANK_ORDER[RANK_ORDER.index("class"):]

# Value of a rank the validator could not determine
INDET_LABEL = "indet."

# Parent rank of each validation rank (None for class)
PARENT_RANK: dict[str, str | None] = {
    rank: (VALIDATION_RANKS[i - 1] if i > 0 else None) for i, rank in enumerate(VALIDATION_RANKS)
}

# Type alias of a taxon's lineage: rank -> name for VALIDATION_RANKS (None if unknown, deeper ranks may be missing)
Lineage = dict[str, str | None]


def file_provenance(path: Path) -> dict[str, Any]:
    """Describe a file for the session record: the checksum identifies its content, also in a copy.

    Args:
        path: File to describe.

    Returns:
        Dict with 'file', 'bytes' and 'sha256' (None if unreadable).
    """
    info: dict[str, Any] = {"file": path.name, "bytes": None, "sha256": None}
    try:
        info["bytes"] = path.stat().st_size
        info["sha256"] = compute_sha256(path)
    except OSError:
        logger.warning("Cannot describe %s for the session record", path)
    return info


def rank_index(rank: str) -> int:
    """Return the position of a rank in VALIDATION_RANKS (class=0 ... species=4)."""
    return VALIDATION_RANKS.index(rank)


def ranks_up_to(rank: str) -> list[str]:
    """Return the validation ranks from class down to (and including) 'rank'."""
    return VALIDATION_RANKS[: rank_index(rank) + 1]


def ranks_below(rank: str) -> list[str]:
    """Return the validation ranks deeper than 'rank'."""
    return VALIDATION_RANKS[rank_index(rank) + 1:]


class TaxonomyIndex:
    """Per-rank name lists, parent -> children maps and lineage lookup."""

    def __init__(self, lineages: pl.DataFrame | None = None) -> None:
        """Initialize the index.

        Args:
            lineages: One row per lineage with the VALIDATION_RANKS columns (deeper ranks may be null)
                      and an optional 'source' column. None for an empty index.
        """
        schema = dict.fromkeys(VALIDATION_RANKS, pl.String) | {"source": pl.String}
        self._df = pl.DataFrame(schema=schema)
        if lineages is not None and not lineages.is_empty():
            if "source" not in lineages.columns:
                lineages = lineages.with_columns(pl.lit("taxonomy_list").alias("source"))
            # Empty strings (incomplete Tree of Life lineages) count as missing
            self._df = (
                lineages.select(list(schema)).cast(pl.String)
                .with_columns([
                    pl.when(pl.col(r).str.strip_chars() == "").then(None).otherwise(pl.col(r)).alias(r)
                    for r in VALIDATION_RANKS
                ])
                .unique(subset=VALIDATION_RANKS, keep="first", maintain_order=True)
            )
        # Tree of Life names that differ from their GBIF canonical name
        self.tol_to_canonical: dict[str, str] = {}
        # Source of the names, written to the session JSON
        self.provenance: dict[str, Any] = {"source": "run_predictions"}
        self._names: dict[str, list[str]] = {}
        self._children: dict[str, dict[str, list[str]]] = {}
        self._lineages: dict[str, dict[str, Lineage]] = {}
        # Case-insensitive lookup: casefolded name -> indexed spelling, per rank
        self._folded: dict[str, dict[str, str]] = {}
        self._sources: dict[tuple[str, str], str] = {}
        # Names under several parents (e.g. a genus in two families): all lineages, the default first
        self._homonyms: dict[str, dict[str, list[Lineage]]] = {}
        self._rebuild()

    @property
    def n_species(self) -> int:
        """Number of species names in the index."""
        return len(self._names.get("species", []))

    @property
    def is_empty(self) -> bool:
        """Whether the index holds no taxa at all."""
        return self._df.is_empty()

    @property
    def uses_species_mapping(self) -> bool:
        """Whether the index was built from the BioCLIP-to-GBIF species mapping file."""
        return self.provenance.get("source") == "species_mapping"

    @staticmethod
    def mapping_file_available() -> bool:
        """Return whether the BioCLIP-to-GBIF species mapping file is present."""
        return SPECIES_MAPPING_CSV.exists()

    @staticmethod
    def ensure_mapping_file(progress_callback: Callable[[int, int, str], None] | None = None) -> None:
        """Download the BioCLIP-to-GBIF species mapping file if it is missing or outdated.

        Args:
            progress_callback: Optional callback(current, total, message) for the download.
        """
        ensure_asset(SPECIES_MAPPING_CSV.name, FILTER_ASSETS_JSON, progress_callback=progress_callback)

    @classmethod
    def from_species_mapping(cls) -> TaxonomyIndex:
        """Build the index from the BioCLIP-to-GBIF species mapping file (downloaded if missing).

        Returns:
            Index of the GBIF-matched species, with 'tol_to_canonical' filled (differing names only).
        """
        mapping = load_species_mapping()
        matched = mapping.filter(pl.col("gbif_species_key").is_not_null() & pl.col("gbif_species").is_not_null())
        lineages = (
            matched.select(
                pl.col("bioclip_class").alias("class"), pl.col("bioclip_order").alias("order"),
                pl.col("bioclip_family").alias("family"), pl.col("bioclip_genus").alias("genus"),
                pl.col("gbif_species").alias("species"),
            )
            .unique(subset=["species"], keep="first", maintain_order=True)
            .with_columns(pl.lit("taxonomy_list").alias("source"))
        )
        index = cls(lineages)
        # Only the names that differ, every other name is its own canonical name
        renamed = matched.filter(pl.col("bioclip_species") != pl.col("gbif_species"))
        index.tol_to_canonical = dict(
            zip(renamed["bioclip_species"].to_list(), renamed["gbif_species"].to_list(), strict=True)
        )
        index.provenance = file_provenance(SPECIES_MAPPING_CSV) | {
            "source": "species_mapping",
            "n_species_tol": mapping.height,
            "n_species_matched": index.n_species,
        }
        logger.info("Taxonomy index built: %d GBIF-matched species of %d Tree of Life species (%s)",
                    index.n_species, mapping.height, index.provenance.get("sha256", "?")[:12])
        return index

    def canonical_species(self, tol_name: str | None) -> str | None:
        """Return the GBIF canonical name of a BioCLIP/Tree of Life species name (unchanged if unknown)."""
        if tol_name is None:
            return None
        return self.tol_to_canonical.get(tol_name, tol_name)

    def canonical_lineage(self, lineage: Lineage) -> Lineage:
        """Return a copy of a lineage with the species mapped to its GBIF canonical name."""
        result = dict(lineage)
        result["species"] = self.canonical_species(lineage.get("species"))
        return result

    def add_lineages(self, lineages: Iterable[Lineage], source: str) -> int:
        """Add lineages that are not yet indexed (predicted taxa, free-text entries).

        Args:
            lineages: Lineage dicts keyed by rank, deeper ranks may be missing.
            source: 'model' or 'free_text'.

        Returns:
            Number of added lineages.
        """
        rows = []
        for lineage in lineages:
            row = {rank: ((lineage.get(rank) or "").strip() or None) for rank in VALIDATION_RANKS}
            if not any(row.values()) or self._is_known(row):
                continue
            row["source"] = source
            rows.append(row)
            self._index_row(row)
        if not rows:
            return 0
        new = pl.DataFrame(rows, schema=dict.fromkeys(VALIDATION_RANKS, pl.String) | {"source": pl.String})
        self._df = pl.concat([self._df, new], how="vertical")
        return len(rows)

    def names(self, rank: str) -> list[str]:
        """Return all names at a rank, sorted case-insensitively."""
        return self._names.get(rank, [])

    def children(self, rank: str, parent_name: str | None) -> list[str]:
        """Return the names at 'rank' below 'parent_name' (all names at the rank if it is None)."""
        if parent_name is None or PARENT_RANK[rank] is None:
            return self.names(rank)
        return self._children.get(rank, {}).get(parent_name, [])

    def lineage_for(self, rank: str, name: str, context: Lineage | None = None) -> Lineage | None:
        """Return a copy of the lineage (class ... rank) of a name, or None if unknown.

        The name is matched case-insensitively, the lineage holds the indexed spelling. A homonym
        takes the lineage that matches the deepest rank entered in 'context', otherwise the default one.

        Args:
            rank: Rank of the name.
            name: Taxon name.
            context: Ranks entered so far ('indet.' is ignored), or None.
        """
        name = self._folded.get(rank, {}).get(name.casefold(), name)
        options = self._homonyms.get(rank, {}).get(name)
        if options and context:
            for higher in reversed(ranks_up_to(rank)[:-1]):
                entered = context.get(higher)
                if entered and entered != INDET_LABEL:
                    match = next((o for o in options if o.get(higher) == entered), None)
                    if match is not None:
                        return dict(match)
        lineage = self._lineages.get(rank, {}).get(name)
        return dict(lineage) if lineage is not None else None

    def contains(self, rank: str, name: str) -> bool:
        """Return whether a name exists at a rank."""
        return name in self._lineages.get(rank, {})

    def source_of(self, rank: str, name: str) -> str:
        """Return the source of a name: 'taxonomy_list', 'model' or 'free_text'."""
        if not self.contains(rank, name):
            return "free_text"
        return self._sources.get((rank, name), "taxonomy_list")

    def _index_row(self, row: dict[str, str | None]) -> None:
        """Add one lineage to the lookups without a rebuild."""
        for rank in VALIDATION_RANKS:
            name = row.get(rank)
            if not name:
                continue
            lineages = self._lineages.setdefault(rank, {})
            lineage = {r: row.get(r) for r in ranks_up_to(rank)}
            parent = PARENT_RANK[rank]
            known = lineages.get(name)
            if known is not None and parent and row.get(parent) and row.get(parent) != known.get(parent):
                options = self._homonyms.setdefault(rank, {}).setdefault(name, [known])
                if lineage not in options:
                    options.append(lineage)
            if name not in lineages:
                lineages[name] = lineage
                self._folded.setdefault(rank, {}).setdefault(name.casefold(), name)
                names = self._names.setdefault(rank, [])
                bisect.insort(names, name, key=str.casefold)
                if row.get("source") and row["source"] != "taxonomy_list":
                    self._sources[(rank, name)] = row["source"]  # type: ignore[assignment]
            parent_name = row.get(parent) if parent else None
            if parent_name:
                siblings = self._children.setdefault(rank, {}).setdefault(parent_name, [])
                if name not in siblings:
                    bisect.insort(siblings, name, key=str.casefold)

    def _is_known(self, row: dict[str, str | None]) -> bool:
        """Return whether the deepest name of 'row' is indexed with the same higher ranks."""
        deepest = next((r for r in reversed(VALIDATION_RANKS) if row.get(r)), None)
        if deepest is None:
            return True
        known = self._lineages.get(deepest, {}).get(row[deepest])  # type: ignore[index]
        if known is None:
            return False
        return all(known.get(r) == row.get(r) for r in ranks_up_to(deepest) if row.get(r))

    def _rebuild(self) -> None:
        """Rebuild the per-rank lookups from the lineage frame."""
        self._names = {}
        self._children = {}
        self._lineages = {}
        self._folded = {}
        self._sources = {}
        self._homonyms = {}
        for rank in VALIDATION_RANKS:
            parent = PARENT_RANK[rank]
            upto = ranks_up_to(rank)
            # File order decides the default lineage of a homonym
            sub = self._df.filter(pl.col(rank).is_not_null()).unique(subset=[rank], keep="first", maintain_order=True)
            names = sorted(sub[rank].to_list(), key=str.casefold)
            self._names[rank] = names
            lineages: dict[str, Lineage] = {}
            for row in sub.select([*upto, "source"]).iter_rows(named=True):
                name = row[rank]
                lineages[name] = {r: row.get(r) for r in upto}
                if row["source"] != "taxonomy_list":
                    self._sources[(rank, name)] = row["source"]
            self._lineages[rank] = lineages
            folded: dict[str, str] = {}
            for name in names:
                folded.setdefault(name.casefold(), name)
            self._folded[rank] = folded
            if parent is not None:
                children: dict[str, list[str]] = {}
                pairs = (
                    self._df.filter(pl.col(rank).is_not_null() & pl.col(parent).is_not_null())
                    .select([parent, rank]).unique()
                )
                for parent_name, name in pairs.iter_rows():
                    children.setdefault(parent_name, []).append(name)
                for values in children.values():
                    values.sort(key=str.casefold)
                self._children[rank] = children
                ambiguous = pairs.group_by(rank).len().filter(pl.col("len") > 1)[rank].to_list()
                if ambiguous:
                    rows = (self._df.filter(pl.col(rank).is_in(ambiguous) & pl.col(parent).is_not_null())
                            .unique(subset=upto, keep="first", maintain_order=True))
                    homonyms: dict[str, list[Lineage]] = {}
                    for row in rows.select(upto).iter_rows(named=True):
                        homonyms.setdefault(row[rank], []).append(dict(row))
                    for name, options in homonyms.items():  # default lineage first
                        first = lineages[name]
                        options.sort(key=lambda o, first=first: o != first)
                    self._homonyms[rank] = homonyms

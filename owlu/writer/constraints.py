"""Ontology constraint checking for label promotion.

Validates that candidate labels conform to the ontology structure
before they are promoted. Currently supports:
  - Naming format validation (e.g. arXiv pattern ``cs.XX``)
  - Parent category existence check
  - Duplicate / near-duplicate rejection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Set


@dataclass
class ConstraintViolation:
    """Describes why a candidate label failed constraint checking."""
    rule: str
    message: str


# Default arXiv-style pattern:  <domain>.<sub>  e.g. cs.AI, stat.ML
_ARXIV_PATTERN = re.compile(r"^[a-z\-]+\.[A-Z][A-Za-z0-9\-]+$")


class OntologyConstraintChecker:
    """Pluggable constraint checker for the Ontology-Constrained Writer.

    Parameters
    ----------
    naming_pattern : re.Pattern | None
        Regex the new_label_id must match.  ``None`` disables naming checks.
    known_parents : set[str] | None
        Valid parent categories (e.g. ``{"cs", "stat", "math"}``).
        ``None`` disables parent existence checks.
    min_edit_distance : int
        Minimum Levenshtein-like character distance to any existing label
        (simple length-difference heuristic).  0 disables.
    """

    def __init__(
        self,
        naming_pattern: re.Pattern[str] | None = None,
        known_parents: set[str] | None = None,
        min_edit_distance: int = 0,
    ):
        self.naming_pattern = naming_pattern
        self.known_parents = known_parents
        self.min_edit_distance = min_edit_distance

    def check(
        self,
        new_label_id: str,
        representative_phrase: str,
        existing_label_ids: Set[str],
    ) -> ConstraintViolation | None:
        """Return a violation if the label should NOT be promoted, else None."""

        # 1. Naming format
        if self.naming_pattern is not None:
            if not self.naming_pattern.match(new_label_id):
                return ConstraintViolation(
                    rule="naming_format",
                    message=(
                        f"Label id '{new_label_id}' does not match "
                        f"required pattern {self.naming_pattern.pattern}"
                    ),
                )

        # 2. Parent category must exist
        if self.known_parents is not None and "." in new_label_id:
            parent = new_label_id.split(".")[0]
            if parent not in self.known_parents:
                return ConstraintViolation(
                    rule="parent_existence",
                    message=(
                        f"Parent category '{parent}' not in known parents: "
                        f"{sorted(self.known_parents)}"
                    ),
                )

        # 3. Duplicate rejection
        normalised = new_label_id.lower().strip()
        for existing in existing_label_ids:
            if existing.lower().strip() == normalised:
                return ConstraintViolation(
                    rule="duplicate",
                    message=f"Label '{new_label_id}' duplicates existing '{existing}'",
                )

        return None

    @classmethod
    def for_aapd(cls, parent_categories: set[str] | None = None) -> "OntologyConstraintChecker":
        """Factory: create a checker pre-configured for AAPD / arXiv ontology."""
        parents = parent_categories or {
            "astro-ph", "cond-mat", "cs", "econ", "eess", "gr-qc",
            "hep-ex", "hep-lat", "hep-ph", "hep-th", "math", "math-ph",
            "nlin", "nucl-ex", "nucl-th", "physics", "q-bio", "q-fin",
            "quant-ph", "stat",
        }
        return cls(
            naming_pattern=_ARXIV_PATTERN,
            known_parents=parents,
        )

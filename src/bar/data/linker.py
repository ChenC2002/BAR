"""Fixed ICD-to-PrimeKG anchor linker."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class LinkResult:
    code: str
    cui: str
    primekg_node_id: str


class ICDToPrimeKGLinker:
    """Two-step fixed linker: ICD -> UMLS CUI -> PrimeKG node id."""

    def __init__(self, icd_to_cui: Dict[str, str], cui_to_primekg: Dict[str, str]) -> None:
        self.icd_to_cui = icd_to_cui
        self.cui_to_primekg = cui_to_primekg

    @classmethod
    def from_csv(cls, icd_to_cui_csv: str | Path, cui_to_primekg_csv: str | Path) -> "ICDToPrimeKGLinker":
        return cls(
            icd_to_cui=_read_two_column_csv(icd_to_cui_csv),
            cui_to_primekg=_read_two_column_csv(cui_to_primekg_csv),
        )

    def link_code(self, code: str) -> LinkResult | None:
        cui = self.icd_to_cui.get(code)
        if cui is None:
            return None
        node_id = self.cui_to_primekg.get(cui)
        if node_id is None:
            return None
        return LinkResult(code=code, cui=cui, primekg_node_id=node_id)

    def anchor_set(self, codes: Iterable[str]) -> List[str]:
        anchors = []
        seen = set()
        for code in codes:
            result = self.link_code(code)
            if result is None or result.primekg_node_id in seen:
                continue
            anchors.append(result.primekg_node_id)
            seen.add(result.primekg_node_id)
        return anchors


def _read_two_column_csv(path: str | Path) -> Dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return {}
    start = 1 if any(cell.lower() in {"icd", "code", "cui", "primekg_node_id"} for cell in rows[0]) else 0
    return {row[0]: row[1] for row in rows[start:] if len(row) >= 2}

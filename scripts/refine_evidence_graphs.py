#!/usr/bin/env python3
"""Scaffold entry point for disease-specific evidence graph refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate graph-refinement inputs and write the refinement manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--primekg", required=True)
    parser.add_argument("--pubmed-index", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for label, path in [
        ("config", args.config),
        ("PrimeKG", args.primekg),
        ("PubMed index", args.pubmed_index),
        ("cohorts", args.cohorts),
    ]:
        if not Path(path).exists():
            raise SystemExit(f"{label} path not found: {path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "evidence_graph_refinement",
        "status": "scaffold_entry_point",
        "implements_full_pipeline": False,
        "config": args.config,
        "expected_output": ["one EvidenceGraph JSON per target disease", "equivalence groups", "expansion guides"],
        "next_step": "Load PrimeKG nodes/edges, cohort anchor frequencies, and PubMed text scores, then call bar.evidence.refinement.EvidenceGraphRefiner.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

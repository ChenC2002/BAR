#!/usr/bin/env python3
"""Refine PrimeKG into disease-specific evidence graphs G_d."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--primekg", required=True)
    parser.add_argument("--pubmed-index", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for label, path in [("PrimeKG", args.primekg), ("PubMed index", args.pubmed_index), ("cohorts", args.cohorts)]:
        if not Path(path).exists():
            raise SystemExit(f"{label} path not found: {path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "evidence_graph_refinement",
        "config": args.config,
        "note": "Use bar.evidence.refinement.EvidenceGraphRefiner after loading PrimeKG nodes, edges, anchor sets, and PubMed text scores.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

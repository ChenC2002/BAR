#!/usr/bin/env python3
"""Scaffold entry point for first-onset post-discharge cohort preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cohort inputs and write the cohort-preparation manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"config path not found: {args.config}")
    mimic_root = Path(args.mimic_root)
    if not mimic_root.exists():
        raise SystemExit(f"MIMIC root not found: {mimic_root}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "cohort_preparation",
        "status": "scaffold_entry_point",
        "implements_full_pipeline": False,
        "config": args.config,
        "mimic_root": str(mimic_root),
        "expected_output": ["train cohort", "validation cohort", "test cohort", "target/horizon labels"],
        "next_step": "Load admissions/diagnoses tables, create PatientVisit objects, then call bar.data.cohort.build_first_onset_samples and temporal_split_samples.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

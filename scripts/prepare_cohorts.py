#!/usr/bin/env python3
"""Prepare first-onset post-discharge cohorts from credentialed MIMIC tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)
    if not mimic_root.exists():
        raise SystemExit(f"MIMIC root not found: {mimic_root}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "cohort_preparation",
        "config": args.config,
        "mimic_root": str(mimic_root),
        "note": "Use bar.data.cohort.build_first_onset_samples after loading admissions/diagnoses tables.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

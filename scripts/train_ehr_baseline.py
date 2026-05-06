#!/usr/bin/env python3
"""Scaffold entry point for Stage 1 EHR-only baseline training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 1 inputs and write the baseline output manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for label, path in [("config", args.config), ("cohorts", args.cohorts)]:
        if not Path(path).exists():
            raise SystemExit(f"{label} path not found: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "stage1_ehr_only_baseline",
        "status": "scaffold_entry_point",
        "implements_full_pipeline": False,
        "config": args.config,
        "expected_output": ["ehr_only_checkpoint", "y_ehr", "L_raw"],
        "next_step": "Connect an EHR encoder and binary-cross-entropy training loop with z_kg fixed to zero.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

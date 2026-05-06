#!/usr/bin/env python3
"""Scaffold entry point for Stage 2 supervised warm-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 2 inputs and write the warm-up output manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for label, path in [("config", args.config), ("graphs", args.graphs), ("stage1", args.stage1)]:
        if not Path(path).exists():
            raise SystemExit(f"{label} path not found: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "stage2_supervised_warmup",
        "status": "scaffold_entry_point",
        "implements_full_pipeline": False,
        "config": args.config,
        "expected_output": ["warmup_predictor_checkpoint", "reasoning_traces"],
        "next_step": "Run the reasoning loop with a fixed acquisition policy and train the predictor on acquired evidence.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

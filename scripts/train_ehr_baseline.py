#!/usr/bin/env python3
"""Stage 1: train the EHR-only baseline and save y_ehr/L_raw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not Path(args.cohorts).exists():
        raise SystemExit(f"cohorts path not found: {args.cohorts}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(
        json.dumps({"stage": "stage1_ehr_only_baseline", "config": args.config}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

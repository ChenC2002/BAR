#!/usr/bin/env python3
"""Stage 3: REINFORCE optimization of the reasoning policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not Path(args.stage2).exists():
        raise SystemExit(f"stage2 path not found: {args.stage2}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(
        json.dumps({"stage": "stage3_reinforce_policy_optimization", "config": args.config}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

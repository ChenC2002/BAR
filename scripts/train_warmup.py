#!/usr/bin/env python3
"""Stage 2: supervised warm-up with fixed budget-aware reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for label, path in [("graphs", args.graphs), ("stage1", args.stage1)]:
        if not Path(path).exists():
            raise SystemExit(f"{label} path not found: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(
        json.dumps({"stage": "stage2_supervised_warmup", "config": args.config}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

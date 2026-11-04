# Tests

```bash
python -m pip install ".[dev,llm,experiment]"
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
bar smoke --config configs/bar_default.json
```

`fixtures/` contains small, fixed inputs for regression tests of graph refinement,
training, checkpoint loading, and the CLI. The public demo lives in `demo/` and
uses recorded predictions from the MIMIC-III experiment.

The suite also builds a wheel from the source distribution, installs it outside
the checkout, and replays all 36 examples from the released model package.
After changing viewer assets, refresh the offline download with
`python scripts/package_demo.py`. To rebuild both downloads from an authorized
run, use `python scripts/package_demo.py --model-run /path/to/run`; this requires
`bar.pt`, `features.json`, and `config.json` and verifies the recorded examples.

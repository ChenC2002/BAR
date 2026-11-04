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

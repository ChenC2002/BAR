import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_demo():
    text = (ROOT / "demo/data.js").read_text()
    return json.loads(text.removeprefix("window.BAR_DEMO = ").removesuffix(";\n"))


def test_public_inputs_match_runner_examples(public_demo):
    tree = ast.parse((ROOT / "scripts/run_mimic_demo.py").read_text())
    definitions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "examples" for target in node.targets)
    ]
    assert len(definitions) == 1
    examples = {row["id"]: row for row in ast.literal_eval(definitions[0])}
    assert {case["id"] for case in public_demo["cases"]} == set(examples)
    for case in public_demo["cases"]:
        assert case["inputs"] == {
            key: examples[case["id"]][key]
            for key in ("codes", "history", "age", "male", "stay_days", "prior_visits", "labs")
        }


def test_public_episodes_keep_budget_provenance_and_fallback(public_demo):
    expected = {f"{h}:{b}" for h in (30, 90, 365) for b in ("auto", "2", "6", "12")}
    for case in public_demo["cases"]:
        assert case["id"].startswith("example-")
        assert set(case["outputs"]) == expected
        for output in case["outputs"].values():
            trace = output["trace"]
            assert trace["sample_id"] == case["id"]
            assert 0 <= output["probability"] <= 1
            assert sum(step["cost"] for step in trace["steps"]) == pytest.approx(
                trace["total_cost"]
            )
            assert trace["total_cost"] <= trace["budget"]
            node_ids = {node["node_id"] for node in output["nodes"]}
            edge_ids = {edge["edge_id"] for edge in output["edges"]}
            assert public_demo["target"] in node_ids
            for edge in output["edges"]:
                assert {edge["head"], edge["tail"]} <= node_ids
            assert {edge["edge_id"] for edge in output["citations"]} == set(trace["citation_ids"])
            for edge in output["citations"]:
                assert edge["edge_id"] in edge_ids
                assert edge["provenance"]
            if trace["fallback"]:
                assert output["probability"] == trace["frozen_ehr_probability"]
                assert not output["citations"]


def test_demo_reports_only_bar_and_frozen_ehr_baseline(public_demo):
    baseline = public_demo["summary"]["baseline"]
    assert baseline["frozen"] is True
    assert baseline["hidden_dim"] == 32
    assert "Stage-1 EHR-only MLP" in baseline["name"]
    assert {row["horizon"] for row in public_demo["summary"]["metrics"]} == {30, 90, 365}
    for row in public_demo["summary"]["metrics"]:
        assert set(row) == {"horizon", "samples", "positives", "prevalence", "ehr", "bar"}
        assert row["samples"] == public_demo["summary"]["cohort"]["test"]["admissions"]


def test_release_version_is_consistent():
    import tomllib

    from bar import __version__

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert __version__ == config["project"]["version"] == "2.0.0"
    assert "V2.0" in (ROOT / "demo/index.html").read_text()


def test_readme_local_links_resolve():
    for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), target


def test_live_demo_entry_and_clean_display():
    readme = (ROOT / "README.md").read_text()
    assert "[**Open the interactive demo**](https://chenc2002.github.io/BAR/demo/)" in readme
    assert "[Release]" not in readme
    html = (ROOT / "demo/index.html").read_text()
    assert "Manually defined example" not in html
    assert 'id="run-date"' not in html
    assert '$("run-date")' not in (ROOT / "demo/app.js").read_text()


def test_demo_assets_are_packaged():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packaged = {
        path for paths in config["tool"]["setuptools"]["data-files"].values() for path in paths
    }
    html = (ROOT / "demo/index.html").read_text()
    references = re.findall(r'(?:src|href)="([^"#]+\.(?:js|css))"', html)
    assert {f"demo/{path}" for path in references} <= packaged
    assert all((ROOT / path).is_file() for path in packaged)


def test_downloads_match_sources_and_checksums(public_demo):
    import hashlib
    from zipfile import ZipFile

    for name, prefix in (("offline", "bar-demo"), ("model", "bar-model")):
        with ZipFile(ROOT / "downloads" / f"bar-demo-{name}.zip") as bundle:
            manifest = json.loads(bundle.read(f"{prefix}/manifest.json"))
            assert manifest["format_version"] == 1
            assert set(bundle.namelist()) == {
                f"{prefix}/{path}" for path in [*manifest["files"], "manifest.json"]
            }
            for path, checksum in manifest["files"].items():
                content = bundle.read(f"{prefix}/{path}")
                assert hashlib.sha256(content).hexdigest() == checksum
                if name == "offline" and path != "README.txt":
                    assert content == (ROOT / "demo" / path).read_bytes()
            if name == "model":
                assert set(manifest["files"]) == {
                    "bar.pt",
                    "features.json",
                    "config.json",
                    "public_demo.json",
                    "README.txt",
                }
                assert json.loads(bundle.read(f"{prefix}/public_demo.json")) == public_demo


def test_released_checkpoint_reproduces_all_examples():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/predict_demo.py"), "--verify-all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["verified_episodes"] == 36
    assert report["max_probability_error"] < 1e-6

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
    readme = (ROOT / "README.md").read_text()
    targets = re.findall(r"\]\(([^)]+)\)", readme) + re.findall(r'src="([^"]+)"', readme)
    for target in targets:
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), target


def test_live_demo_entry_and_clean_display():
    readme = (ROOT / "README.md").read_text()
    assert "[**Open the interactive demo**](https://chenc2002.github.io/BAR/demo/)" in readme
    assert readme.index("## Demo\n") < readme.index("## Method\n")
    assert "## Quick Start" not in readme
    assert readme.index("assets/bar_demo.png") < readme.index("<details>")
    assert readme.index("<details>") < readme.index("git clone") < readme.index("</details>")
    assert readme.index("## Method\n") < readme.index("assets/bar_overview.png")
    demo_section = readme.split("## Demo\n", 1)[1].split("## Method\n", 1)[0]
    folds = re.findall(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", demo_section, re.S)
    assert [title for title, _ in folds] == ["Run locally (Python 3.11+)", "MIMIC-III experiment"]
    assert "scripts/run_mimic_demo.py" in folds[1][1]
    assert "## MIMIC-III Experiment" not in readme
    assert "#mimic-iii-experiment" not in readme
    assert re.search(
        r'<a href="https://chenc2002.github.io/BAR/demo/">\s*<img src="assets/bar_demo.png"',
        readme,
    )
    assert "[Release]" not in readme
    html = (ROOT / "demo/index.html").read_text()
    assert "Manually defined example" not in html
    assert 'id="run-date"' not in html
    assert '$("run-date")' not in (ROOT / "demo/app.js").read_text()


def test_demo_assets_are_packaged():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    settings = config["tool"]["setuptools"]
    base = ROOT / settings["package-dir"]["bar._demo"]
    packaged = {
        path for pattern in settings["package-data"]["bar._demo"] for path in base.glob(pattern)
    }
    html = (ROOT / "demo/index.html").read_text()
    references = re.findall(r'(?:src|href)="([^"#]+\.(?:js|css))"', html)
    assert {ROOT / "demo" / path for path in references} <= packaged
    assert all(path.is_file() for path in packaged)


@pytest.mark.parametrize(
    "field", ["provenance", "reason", "utility", "probability", "edges", "nodes"]
)
def test_replay_verifies_citation_contents_and_complete_trace(
    public_demo, tmp_path, monkeypatch, field
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from mimic_features import FeatureEncoder
    from predict_demo import read_bundle, verify

    from bar.policy import DemoPolicy
    from bar.training import load_checkpoint

    files = read_bundle(ROOT / "downloads/bar-demo-model.zip")
    checkpoint = tmp_path / "bar.pt"
    checkpoint.write_bytes(files["bar.pt"])
    engine = load_checkpoint(checkpoint, DemoPolicy())
    encoder = FeatureEncoder.from_dict(json.loads(files["features.json"]))
    recorded = next(iter(public_demo["cases"][0]["outputs"].values()))
    if field == "provenance":
        recorded["citations"][0]["provenance"][0]["source_id"] = "altered"
    elif field == "probability":
        recorded["probability"] = float("nan")
    elif field == "edges":
        recorded["edges"].pop()
    elif field == "nodes":
        recorded["nodes"][0]["name"] = "altered"
    else:
        recorded["trace"]["steps"][0][field] = "altered" if field == "reason" else 100.0
    with pytest.raises(ValueError, match="Replay mismatch"):
        verify(engine, encoder, public_demo)


def test_failed_model_packaging_preserves_existing_downloads(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from package_demo import build

    destination = tmp_path / "downloads"
    destination.mkdir()
    for name in ("bar-demo-offline.zip", "bar-demo-model.zip"):
        (destination / name).write_bytes(b"previous release")
    with pytest.raises(FileNotFoundError):
        build(destination, tmp_path / "missing-model")
    assert {path.name for path in destination.iterdir()} == {
        "bar-demo-offline.zip",
        "bar-demo-model.zip",
    }
    assert all(path.read_bytes() == b"previous release" for path in destination.iterdir())


def test_replay_comparison_does_not_treat_booleans_as_numbers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from predict_demo import assert_match

    for actual, expected in ((1.0, True), (True, 1)):
        with pytest.raises(ValueError, match="Replay mismatch"):
            assert_match(actual, expected, "value")


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

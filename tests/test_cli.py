import json

from bar.cli import main


def test_version(monkeypatch, capsys):
    import pytest

    monkeypatch.setattr("sys.argv", ["bar", "--version"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == "BAR 2.0.0"


def test_prepared_input_commands(tmp_path, monkeypatch, capsys):
    graph = tmp_path / "new-directory" / "graph.json"
    checkpoint = tmp_path / "checkpoints" / "model.pt"

    def run(*args):
        monkeypatch.setattr("sys.argv", ["bar", *map(str, args)])
        main()
        return capsys.readouterr().out

    run(
        "refine",
        "--graph",
        "tests/fixtures/tiny_evidence_graph.json",
        "--disease",
        "CKD",
        "--anchor",
        "DIABETES",
        "--anchor",
        "HTN",
        "--out",
        graph,
    )
    assert "valid graph" in run("validate", "--graph", graph)
    training = json.loads(
        run(
            "train",
            "--graph",
            graph,
            "--data",
            "tests/fixtures/tiny_cases.json",
            "--policy",
            "demo",
            "--config",
            "configs/bar_default.json",
            "--out",
            checkpoint,
        )
    )
    assert {item["stage"] for item in training["history"]} == {"ehr_only", "warmup", "reinforce"}
    prediction = json.loads(
        run(
            "predict",
            "--checkpoint",
            checkpoint,
            "--case",
            "tests/fixtures/tiny_case.json",
            "--policy",
            "demo",
        )
    )
    assert 0 <= prediction["probability"] <= 1
    assert prediction["trace"]["total_cost"] <= prediction["trace"]["budget"]
    assert not prediction["trace"]["fallback"]


def test_smoke_checkpoint_matches_prediction(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "demo.pt"
    monkeypatch.setattr(
        "sys.argv",
        ["bar", "smoke", "--config", "configs/bar_default.json", "--out", str(checkpoint)],
    )
    main()
    demo = json.loads(capsys.readouterr().out)
    assert demo["mode"] == "example data and offline policy"
    assert demo["trace"]["sample_id"] == "example_prediction"
    assert all(
        source["source_id"].startswith("example:")
        for edge in demo["citations"]
        for source in edge["provenance"]
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "bar",
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--case",
            "tests/fixtures/tiny_case.json",
            "--policy",
            "demo",
        ],
    )
    main()
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["probability"] == demo["probability"]
    assert prediction["trace"] == demo["trace"]


def test_demo_opens_recorded_viewer(monkeypatch, capsys):
    from pathlib import Path

    opened = []
    monkeypatch.setattr("bar.cli.webbrowser.open", lambda url: opened.append(url) or True)
    monkeypatch.setattr("sys.argv", ["bar", "demo"])
    main()
    page = Path(capsys.readouterr().out.strip())
    assert page.name == "index.html" and page.is_file()
    assert opened == [page.as_uri()]
    monkeypatch.setattr("sys.argv", ["bar", "demo", "--no-open"])
    main()
    assert len(opened) == 1

import json

from bar.cli import main


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
        "examples/tiny_evidence_graph.json",
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
            "examples/tiny_cases.json",
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
            "examples/tiny_case.json",
            "--policy",
            "demo",
        )
    )
    assert 0 <= prediction["probability"] <= 1
    assert prediction["trace"]["total_cost"] <= prediction["trace"]["budget"]
    assert not prediction["trace"]["fallback"]


def test_demo_checkpoint_matches_prediction(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "demo.pt"
    monkeypatch.setattr(
        "sys.argv",
        ["bar", "demo", "--config", "configs/bar_default.json", "--out", str(checkpoint)],
    )
    main()
    demo = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bar",
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--case",
            "examples/tiny_case.json",
            "--policy",
            "demo",
        ],
    )
    main()
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["probability"] == demo["probability"]
    assert prediction["trace"] == demo["trace"]

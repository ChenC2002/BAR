"""Small executable commands for prepared inputs and examples."""

import argparse
import json
import webbrowser
from importlib.resources import files
from pathlib import Path

from pydantic import TypeAdapter

from bar import __version__
from bar.config import Config
from bar.graph import GraphStore, refine_graph
from bar.policy import DemoPolicy, LLMPolicy
from bar.schemas import Case, TrainingCase
from bar.training import load_checkpoint, save_checkpoint, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="bar")
    parser.add_argument("--version", action="version", version=f"BAR {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    viewer = commands.add_parser("demo", help="Open the recorded CKD demo in a browser")
    viewer.add_argument(
        "--no-open", action="store_true", help="Print the viewer path without opening it"
    )
    demo = commands.add_parser("smoke", help="Run the small regression fixtures from a checkout")
    demo.add_argument("--graph", default="tests/fixtures/tiny_evidence_graph.json")
    demo.add_argument("--data", default="tests/fixtures/tiny_cases.json")
    demo.add_argument("--case", default="tests/fixtures/tiny_case.json")
    demo.add_argument("--out", help="Optional inference checkpoint")
    demo.add_argument("--config")
    training = commands.add_parser("train", help="Train on a JSON array of prepared training cases")
    training.add_argument("--graph", required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--config")
    predict = commands.add_parser("predict", help="Predict using a saved checkpoint")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--case", required=True)
    for command in (training, predict):
        command.add_argument("--policy", choices=["demo", "llm"], required=True)
        command.add_argument("--model", help="Structured-output model ID for the LLM policy")
        command.add_argument("--base-url", help="Optional compatible inference endpoint")
    refine = commands.add_parser(
        "refine", help="Refine a graph with precomputed support/provenance"
    )
    refine.add_argument("--graph", required=True)
    refine.add_argument("--disease", required=True)
    refine.add_argument("--anchor", action="append", required=True, help="Frequent TRAINING anchor")
    refine.add_argument("--out", required=True)
    refine.add_argument("--config")
    validate = commands.add_parser("validate", help="Validate an evidence graph")
    validate.add_argument("--graph", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        try:
            page = Path(str(files("bar._demo").joinpath("index.html")))
        except ModuleNotFoundError:
            page = Path(__file__).resolve().parents[2] / "demo/index.html"
        if not page.is_file():
            parser.error("demo assets are missing; reinstall BAR or run from the repository")
        print(page.resolve())
        if not args.no_open and not webbrowser.open(page.resolve().as_uri()):
            parser.error("no browser was found; open the printed HTML path manually")
        return
    config = Config.load(args.config) if getattr(args, "config", None) else Config()
    if args.command in ("validate", "refine"):
        store = GraphStore.load(args.graph)
        if args.command == "refine":
            graph = refine_graph(store.graph, args.disease, args.anchor, config)
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
        else:
            print(f"valid graph: {len(store.nodes)} nodes, {len(store.edges)} edges")
        return
    policy = DemoPolicy()
    if getattr(args, "policy", "demo") == "llm":
        if not args.model:
            parser.error("--model is required with --policy llm")
        policy = LLMPolicy(args.model, base_url=args.base_url)
    if args.command == "predict":
        engine = load_checkpoint(args.checkpoint, policy)
    else:
        cases = TypeAdapter(list[TrainingCase]).validate_json(
            Path(args.data).read_text(encoding="utf-8")
        )
        store = GraphStore.load(args.graph)
        disease = store.graph.metadata.get("target_disease_id")
        if not disease:
            parser.error("graph metadata.target_disease_id is required")
        engine, history = train(cases, {disease: store}, policy, config)
        if args.out:
            save_checkpoint(engine, args.out)
        if args.command == "train":
            print(
                json.dumps(
                    {"checkpoint": args.out, "policy": args.policy, "history": history}, indent=2
                )
            )
            return
    case = Case.model_validate_json(Path(args.case).read_text(encoding="utf-8"))
    prediction = engine.predict(case)
    output = prediction.model_dump()
    if args.command == "smoke":
        output["mode"] = "example data and offline policy"
        output["training_stages"] = sorted({item["stage"] for item in history})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

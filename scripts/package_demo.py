"""Build bounded, checksum-verified downloads from the public demo and audited model files."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
VIEWER_FILES = (
    "index.html",
    "app.js",
    "styles.css",
    "data.js",
    "vendor/lucide.min.js",
    "vendor/lucide-LICENSE",
)
MODEL_FILES = ("bar.pt", "features.json", "config.json", "public_demo.json")
RELEASE_FILES = ("LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md")


def archive(path, folder, files):
    manifest = {
        "format_version": 1,
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    files = {**files, "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
        for name, content in sorted(files.items()):
            info = ZipInfo(f"{folder}/{name}")
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, content)
    return path.stat().st_size


def build(destination, model_run):
    notices = {name: (ROOT / name).read_bytes() for name in RELEASE_FILES}
    viewer = {name: (ROOT / "demo" / name).read_bytes() for name in VIEWER_FILES}
    viewer.update(notices)
    viewer["README.txt"] = (
        b"BAR Evidence Explorer\nOpen index.html in a browser. No server or installation is required.\n"
    )
    packages = {"bar-demo-offline.zip": ("bar-demo", viewer)}
    if model_run:
        import torch
        from mimic_features import FeatureEncoder
        from predict_demo import verify

        from bar.policy import DemoPolicy
        from bar.training import load_checkpoint

        checkpoint = torch.load(model_run / "bar.pt", map_location="cpu", weights_only=True)
        if set(checkpoint) != {
            "format_version",
            "model_spec",
            "model",
            "baseline",
            "router",
            "config",
            "budgets",
            "graphs",
        }:
            raise ValueError("Unexpected checkpoint sections")
        encoder = FeatureEncoder.from_dict(json.loads((model_run / "features.json").read_text()))
        public = json.loads(
            (ROOT / "demo/data.js")
            .read_text()
            .removeprefix("window.BAR_DEMO = ")
            .removesuffix(";\n")
        )
        if json.loads((model_run / "config.json").read_text()) != checkpoint["config"]:
            raise ValueError("Run configuration does not match the checkpoint")
        torch.set_num_threads(2)
        verify(load_checkpoint(model_run / "bar.pt", DemoPolicy()), encoder, public)
        files = {
            name: (model_run / name).read_bytes()
            for name in MODEL_FILES
            if name != "public_demo.json"
        }
        files["public_demo.json"] = (json.dumps(public, allow_nan=False) + "\n").encode()
        files.update(notices)
        files["README.txt"] = (
            "BAR CKD demo inference bundle\n\n"
            "bar.pt: trained BAR, frozen EHR baseline, query router, evidence graph and uncertainty CDFs.\n"
            "features.json: fitted vocabulary, imputation, missingness indicators and scaling.\n"
            "config.json: recorded run configuration.\n"
            "public_demo.json: example inputs, recorded outputs and aggregate metrics.\n\n"
            "From a BAR repository checkout:\n"
            "python -m pip install .\n"
            "python scripts/predict_demo.py --bundle downloads/bar-demo-model.zip --verify-all\n\n"
            "Evaluate the full Experiment on the authorized MIMIC-III data host, without training:\n"
            'python -m pip install ".[experiment]"\n'
            "python scripts/evaluate_mimic_demo.py --data /path/to/MIMICIII/data "
            "--mapping /path/to/KG_mapping/ICD9CM_to_UMLS.csv "
            "--out artifacts/ckd-evaluation.json\n"
            "The report recomputes aggregate metrics and checks them against this release.\n\n"
            "The JSON preprocessor does not require loading joblib or pickle objects.\n"
            "Load checkpoint tensors with torch.load(weights_only=True).\n"
            "Reference runtime: Python 3.11, PyTorch 2.15.0, NumPy 2.5.4.\n"
            "Research use; these scores are not clinical recommendations.\n"
            "UMLS content remains subject to its source license terms:\n"
            "https://www.nlm.nih.gov/research/umls/faq_main.html\n"
        ).encode()
        packages["bar-demo-model.zip"] = ("bar-model", files)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as directory:
        staged = Path(directory)
        sizes = {
            name: archive(staged / name, folder, files)
            for name, (folder, files) in packages.items()
        }
        for name in packages:
            (staged / name).replace(destination / name)
    print(json.dumps(sizes, indent=2))
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "downloads")
    parser.add_argument(
        "--model-run", type=Path, help="Audited inference files only; never a raw data directory"
    )
    args = parser.parse_args()
    build(args.out, args.model_run)

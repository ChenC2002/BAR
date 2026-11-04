import os
import subprocess
import sys
import tarfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_and_wheel_work_outside_checkout(tmp_path):
    build = pytest.importorskip("build")
    archive = build.ProjectBuilder(ROOT).build("sdist", tmp_path / "dist")
    with tarfile.open(archive) as source:
        names = {name.split("/", 1)[1] for name in source.getnames() if "/" in name}
        required = {
            "LICENSE",
            "CITATION.cff",
            "THIRD_PARTY_NOTICES.md",
            "tests/conftest.py",
            "tests/fixtures/tiny_cases.json",
            "configs/bar_default.json",
            "scripts/predict_demo.py",
            "scripts/evaluate_mimic_demo.py",
            "scripts/package_ckd_results.py",
            "scripts/run_mimic_demo.py",
            "scripts/mimic_features.py",
            "downloads/bar-demo-model.zip",
            "downloads/bar-ckd-results.zip",
            "demo/vendor/lucide-LICENSE",
        }
        assert required <= names
        source.extractall(tmp_path / "source", filter="data")
    checkout = next((tmp_path / "source").iterdir())
    wheel = build.ProjectBuilder(checkout).build("wheel", tmp_path / "dist")
    with ZipFile(wheel) as bundle:
        metadata_path = next(name for name in bundle.namelist() if name.endswith("/METADATA"))
        metadata = BytesParser().parsebytes(bundle.read(metadata_path))
        licenses = {"LICENSE", "THIRD_PARTY_NOTICES.md", "demo/vendor/lucide-LICENSE"}
        assert set(metadata.get_all("License-File")) == licenses
        for name in licenses:
            path = metadata_path.removesuffix("METADATA") + "licenses/" + name
            assert bundle.read(path) == (ROOT / name).read_bytes()
    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), wheel],
        check=True,
        capture_output=True,
        text=True,
    )
    env = {**os.environ, "PYTHONPATH": str(target)}
    # An unrelated cwd must not override the viewer belonging to the installed package.
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/index.html").write_text("unrelated")
    viewer = subprocess.run(
        [sys.executable, "-m", "bar.cli", "demo", "--no-open"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    page = Path(viewer.stdout.strip())
    assert page == target / "bar/_demo/index.html"
    assert page.read_bytes() == (ROOT / "demo/index.html").read_bytes()
    assert (page.parent / "vendor/lucide.min.js").is_file()
    for command in (
        ["-m", "bar.cli", "smoke", "--config", "configs/bar_default.json"],
        ["scripts/predict_demo.py", "--verify-all"],
        ["scripts/evaluate_mimic_demo.py", "--help"],
        ["scripts/package_ckd_results.py", "--help"],
    ):
        subprocess.run(
            [sys.executable, *command],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

"""Build and validate a release source ZIP, including imports after clean extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "0.20"
BUNDLE_DIR = "rsi-zhixun-harness-v20"
FOLDERS = {"src", "vendor", "harness", "scripts", "tests", "configs", "docs", "reports"}
TOP_FILES = {
    "AGENTS.md",
    "README.md",
    "SOURCE_README.md",
    "THIRD_PARTY_LICENSES.md",
    "Dockerfile",
    "harness.Dockerfile",
    "requirements.txt",
    "plan0.md",
    "plan1.md",
    ".gitignore",
    ".dockerignore",
    "release_manifest.json",
}
EXCLUDED = {"__pycache__", ".git", "node_modules", "_state", "dist", "experiments", "_sdk"}


def source_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if not (rel.parts[0] in FOLDERS or rel.as_posix() in TOP_FILES):
            continue
        if any(part in EXCLUDED for part in rel.parts):
            continue
        if path.suffix.lower() in {".pyc", ".npz", ".tar", ".zip", ".key", ".env", ".mhtml"}:
            continue
        yield rel.as_posix(), path


def _normalized_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".py", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_manifest():
    hashes = {
        name: _normalized_sha256(path)
        for name, path in source_files()
        if (name.split("/")[0] in {"src", "vendor", "harness"} and path.suffix == ".py")
        or name == "scripts/check_release.py"
    }
    manifest = {
        "schema": 1,
        "release": RELEASE,
        "runtime_sha256": hashes,
        "purpose": "detect mixed or incomplete releases; not a cryptographic signature",
    }
    (ROOT / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    write_manifest()
    if args.manifest_only:
        return
    if not args.output:
        parser.error("--output required unless --manifest-only")
    if args.output.exists():
        parser.error("output already exists; use a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name, path in source_files():
                z.write(path, f"{BUNDLE_DIR}/{name}")
        clean = Path(temporary) / "clean"
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP CRC check failed")
            z.extractall(clean)
        root = clean / BUNDLE_DIR
        # A fresh process prevents an already-imported good module from masking
        # a missing or older module in the newly exported ZIP.
        subprocess.run(
            [sys.executable, str(root / "scripts/check_release.py"), "--root", str(root)],
            check=True,
            cwd=root,
        )
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            check=True,
            cwd=root,
        )
        args.output.write_bytes(archive.read_bytes())
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()

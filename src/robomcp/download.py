"""`robomcp-download`: fetch the documentation corpus and the question set, at pinned versions.

    data/ros2_documentation/   ROS 2 docs sources (CC BY 4.0), ros2/ros2_documentation @ commit
    data/eval/questions.jsonl  66 hand-written questions from robotics-docs-rag @ commit
    data/SOURCES.json          URL, size and SHA-256 of every downloaded file

Re-running is a no-op for files that already exist (use --force to refetch).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from robomcp.config import Settings, load_settings

ROS2_ZIP = "https://codeload.github.com/ros2/ros2_documentation/zip/{commit}"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
QUESTIONS_PATH = "eval/questions.jsonl"
UA = {"User-Agent": "agent-mcp-robotics/0.1 (portfolio project)"}


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class Sources:
    """Record of what was downloaded from where (data/SOURCES.json)."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict = json.loads(path.read_text()) if path.exists() else {}

    def add(self, name: str, url: str, blob: bytes) -> None:
        self.items[name] = {
            "url": url, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=1, sort_keys=True) + "\n")


def safe_members(zf: zipfile.ZipFile, suffixes: tuple[str, ...]) -> list[str]:
    """Members to extract: wanted suffixes only, and never a path that escapes the target."""
    out = []
    for name in zf.namelist():
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path in archive: {name!r}")
        if name.endswith(suffixes):
            out.append(name)
    return out


def extract_ros2_zip(blob: bytes, dest: Path, commit: str) -> int:
    tmp = dest.parent / "_ros2_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = safe_members(zf, (".rst", "LICENSE"))  # text only, not 40 MB of images
        zf.extractall(tmp, members)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"expected one top-level folder in the archive, found {len(roots)}")
    shutil.rmtree(dest, ignore_errors=True)
    roots[0].rename(dest)
    shutil.rmtree(tmp)
    (dest / "COMMIT").write_text(commit + "\n")
    return len(members)


def download_ros2_docs(cfg: Settings, sources: Sources, force: bool = False) -> Path:
    dest = cfg.data_dir / "ros2_documentation"
    if (dest / "source").exists() and not force:
        print(f"ROS 2 docs already in {dest}")
        return dest
    commit = cfg.corpus.ros2_commit
    url = ROS2_ZIP.format(commit=commit)
    print(f"downloading ros2_documentation @ {commit[:10]} ...")
    blob = fetch(url)
    sources.add("ros2_documentation.zip", url, blob)
    n = extract_ros2_zip(blob, dest, commit)
    print(f"saved {n} files to {dest}")
    return dest


def download_questions(cfg: Settings, sources: Sources, force: bool = False) -> Path:
    out = cfg.data_dir / "eval" / "questions.jsonl"
    if out.exists() and not force:
        print(f"questions already in {out}")
        return out
    url = RAW.format(repo=cfg.eval.source_repo, commit=cfg.eval.source_commit,
                     path=QUESTIONS_PATH)
    blob = fetch(url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    sources.add("eval/questions.jsonl", url, blob)
    print(f"saved {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    sources = Sources(cfg.data_dir / "SOURCES.json")
    download_ros2_docs(cfg, sources, args.force)
    download_questions(cfg, sources, args.force)


if __name__ == "__main__":
    main()

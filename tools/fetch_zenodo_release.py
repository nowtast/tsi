#!/usr/bin/env python3
"""Download and verify an archived TSI evidence release."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile, ZipInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "artifacts" / "paper03-04-v1.0.0.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / ".artifacts"


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("artifact manifest must declare exactly one file")
    artifact = files[0]
    required = {"name", "url", "size_bytes", "sha256", "zenodo_md5"}
    if not isinstance(artifact, dict) or not required <= artifact.keys():
        raise ValueError("artifact manifest has incomplete file metadata")
    return payload


def digest_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_file(path: Path, metadata: dict[str, Any]) -> None:
    actual_digest, actual_size = digest_file(path)
    if actual_size != metadata["size_bytes"]:
        raise ValueError(
            f"size mismatch for {path.name}: {actual_size} != {metadata['size_bytes']}"
        )
    if actual_digest != metadata["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: {actual_digest} != {metadata['sha256']}"
        )


def download_file(metadata: dict[str, Any], destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(metadata["url"]) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    verify_file(partial, metadata)
    partial.replace(destination)


def _is_symlink(info: ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_IFMT(mode) == stat.S_IFLNK


def extract_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if _is_symlink(info):
                raise ValueError(f"archive contains a symbolic link: {info.filename}")
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as error:
                raise ValueError(
                    f"archive member escapes extraction root: {info.filename}"
                ) from error
        archive.extractall(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    metadata = manifest["files"][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / metadata["name"]
    if archive_path.exists():
        verify_file(archive_path, metadata)
        print(f"verified existing {archive_path}")
    else:
        download_file(metadata, archive_path)
        print(f"downloaded and verified {archive_path}")
    if args.extract:
        extraction_root = args.output_dir / archive_path.stem
        extract_safely(archive_path, extraction_root)
        print(f"extracted {extraction_root}")


if __name__ == "__main__":
    main()

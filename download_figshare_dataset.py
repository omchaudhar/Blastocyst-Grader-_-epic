#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import requests

DEFAULT_ARTICLE_ID = 20123153  # Figshare DOI: 10.6084/m9.figshare.20123153.v3
FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download files from public Figshare blastocyst dataset.")
    parser.add_argument("--article-id", type=int, default=DEFAULT_ARTICLE_ID)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only download files containing this substring (can be used multiple times).",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit")
    parser.add_argument("--extract-zip", action="store_true", help="Extract downloaded .zip files")
    return parser.parse_args()


def should_download(filename: str, filters: list[str]) -> bool:
    if not filters:
        return True
    lowered = filename.lower()
    return any(f.lower() in lowered for f in filters)


def fetch_article(article_id: int) -> dict:
    response = requests.get(FIGSHARE_API.format(article_id=article_id), timeout=30)
    response.raise_for_status()
    return response.json()


def download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def maybe_extract_zip(path: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    article = fetch_article(args.article_id)
    files = article.get("files", [])

    selected = []
    for item in files:
        name = item.get("name", "")
        if should_download(name, args.include):
            selected.append(item)

    if args.max_files > 0:
        selected = selected[: args.max_files]

    downloaded = []
    for i, item in enumerate(selected, start=1):
        name = item["name"]
        url = item["download_url"]
        dest = output_dir / name
        print(f"[{i}/{len(selected)}] Downloading {name}")
        download_file(url, dest)
        if args.extract_zip:
            maybe_extract_zip(dest)
        downloaded.append({"name": name, "path": str(dest), "size": item.get("size")})

    manifest = {
        "article_id": args.article_id,
        "title": article.get("title"),
        "doi": article.get("doi"),
        "url": article.get("url"),
        "files_downloaded": downloaded,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

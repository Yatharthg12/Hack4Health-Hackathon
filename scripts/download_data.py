"""Download the public organizer dataset folder with validation, retries, and resume."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown
import requests

DEFAULT_FOLDER_ID = "1R9ka23jnBsNDyPh6l03f2Zv3d7gyk3tR"
THREAD_LOCAL = threading.local()


def _session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        THREAD_LOCAL.session = requests.Session()
        THREAD_LOCAL.session.headers["User-Agent"] = "Mozilla/5.0 MindFuse-Dataset-Audit/1.0"
    return THREAD_LOCAL.session


def _download(item, attempts: int = 4) -> tuple[str, str]:
    target = Path(item.local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return "skipped", str(target)
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"https://drive.usercontent.google.com/download?id={item.id}&export=download&confirm=t"
    for attempt in range(attempts):
        try:
            with _session().get(url, stream=True, timeout=(12, 90)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("content-type", "").lower():
                    raise RuntimeError("Google returned HTML rather than dataset content")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(256 * 1024):
                        if chunk:
                            stream.write(chunk)
            if temporary.stat().st_size <= 0:
                raise RuntimeError("Google returned an empty file")
            os.replace(temporary, target)
            return "downloaded", str(target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                return "failed", f"{item.id}: {target}: {exc}"
            time.sleep(2**attempt)
    return "failed", str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9), metavar="1-8")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = gdown.download_folder(
        id=args.folder_id,
        output=str(args.output),
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if not items:
        raise SystemExit("Unable to list the public Google Drive folder")
    priority = {".csv": 0, ".png": 1, ".jpg": 1, ".jpeg": 1, ".wav": 2}
    items = sorted(items, key=lambda item: priority.get(Path(item.local_path).suffix.lower(), 3))
    counters = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    print(f"Discovered {len(items):,} organizer files", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download, item) for item in items]
        for completed, future in enumerate(as_completed(futures), start=1):
            status, detail = future.result()
            counters[status] += 1
            if status == "failed":
                failures.append(detail)
            if completed % 100 == 0 or status == "failed":
                print(f"{completed:,}/{len(items):,}: {counters}", flush=True)
    print(f"Download summary: {counters}")
    for failure in failures:
        print(f"FAILED: {failure}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


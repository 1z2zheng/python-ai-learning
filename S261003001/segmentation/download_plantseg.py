"""Download PlantSeg v7 from Zenodo with resumable parallel ranges."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


URL = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"
TOTAL_SIZE = 1_057_281_724
EXPECTED_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
PARTS = 16
PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT / "downloads"
TARGET = DOWNLOAD_DIR / "plantseg-v7.zip"
PART_DIR = DOWNLOAD_DIR / "plantseg-v7.parts"
PRINT_LOCK = threading.Lock()


def ranges() -> list[tuple[int, int, int]]:
    chunk = (TOTAL_SIZE + PARTS - 1) // PARTS
    result = []
    for index in range(PARTS):
        start = index * chunk
        if start >= TOTAL_SIZE:
            break
        end = min(TOTAL_SIZE - 1, start + chunk - 1)
        result.append((index, start, end))
    return result


def download_part(index: int, start: int, end: int) -> Path:
    part_path = PART_DIR / f"part-{index:02d}.bin"
    expected = end - start + 1
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > expected:
        part_path.unlink()
        existing = 0
    if existing == expected:
        return part_path

    for attempt in range(1, 11):
        current = part_path.stat().st_size if part_path.exists() else 0
        if current == expected:
            return part_path
        request_start = start + current
        request = urllib.request.Request(
            URL,
            headers={
                "Range": f"bytes={request_start}-{end}",
                "User-Agent": "PlantSeg-thesis-downloader/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Part {index}: expected HTTP 206, got {response.status}"
                    )
                with part_path.open("ab") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            wait = min(60, attempt * 5)
            with PRINT_LOCK:
                print(f"Part {index:02d} retry {attempt}/10 after {error}; wait {wait}s")
            time.sleep(wait)
            continue

    actual = part_path.stat().st_size if part_path.exists() else 0
    if actual != expected:
        raise RuntimeError(f"Part {index}: expected {expected} bytes, got {actual}")
    return part_path


def downloaded_bytes() -> int:
    return sum(path.stat().st_size for path in PART_DIR.glob("part-*.bin"))


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == TOTAL_SIZE:
        existing_hash = hashlib.md5()
        with TARGET.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                existing_hash.update(block)
        digest = existing_hash.hexdigest()
        if digest == EXPECTED_MD5:
            print(f"Already downloaded and verified: {TARGET}")
            return

    work = ranges()
    print(f"Downloading PlantSeg v7: {TOTAL_SIZE / 1024**3:.2f} GiB in {len(work)} parts")
    with ThreadPoolExecutor(max_workers=len(work)) as executor:
        futures = {
            executor.submit(download_part, index, start, end): index
            for index, start, end in work
        }
        pending = set(futures)
        while pending:
            done = {future for future in pending if future.done()}
            for future in done:
                future.result()
                pending.remove(future)
            completed = downloaded_bytes()
            print(
                f"Progress: {completed / 1024**2:.1f}/{TOTAL_SIZE / 1024**2:.1f} MiB "
                f"({completed / TOTAL_SIZE * 100:.1f}%), parts remaining={len(pending)}",
                flush=True,
            )
            if pending:
                time.sleep(10)

    temporary = TARGET.with_suffix(".zip.tmp")
    with temporary.open("wb") as output:
        for index, _, _ in work:
            with (PART_DIR / f"part-{index:02d}.bin").open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != TOTAL_SIZE:
        raise RuntimeError("Combined archive has the wrong size")

    digest = hashlib.md5()
    with temporary.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual_md5 = digest.hexdigest()
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 mismatch: expected {EXPECTED_MD5}, got {actual_md5}"
        )
    os.replace(temporary, TARGET)
    for part_path in PART_DIR.glob("part-*.bin"):
        part_path.unlink()
    PART_DIR.rmdir()
    print(f"Download verified: {TARGET}")
    print(f"MD5: {actual_md5}")


if __name__ == "__main__":
    main()

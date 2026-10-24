"""Download the labeled AppleLeaf9 frogeye class and add deduplicated training photos.

Uses the dataset authors' expert-screened labels, not predicted labels. Frogeye
leaf spot is the leaf manifestation of apple black rot (see SOURCE_APPLELEAF9.md).
No segmentation masks are inferred from these classification labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/JasonYangCode/AppleLeaf9"
CLASS = "Apple_Black_rot"
PREFIX = "appleleaf9_blackrot_"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CropDiseaseResearch/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def bits(values: np.ndarray) -> int:
    return int.from_bytes(np.packbits(values.reshape(-1)).tobytes(), "big")


def signatures(image: Image.Image) -> tuple[int, int]:
    small = image.resize((32, 32), Image.Resampling.LANCZOS).convert("L")
    pixels = np.asarray(small.resize((9, 8), Image.Resampling.LANCZOS))
    dhash = bits(pixels[:, 1:] > pixels[:, :-1])
    coefficients = dctn(np.asarray(small, dtype=np.float32), type=2, norm="ortho")[:8, :8]
    phash = bits(coefficients > np.median(coefficients.reshape(-1)[1:]))
    return dhash, phash


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value: int, name: str):
        node = [value, name, {}]
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = (value ^ current[0]).bit_count()
            if distance == 0:
                return
            if distance not in current[2]:
                current[2][distance] = node
                return
            current = current[2][distance]

    def match(self, value: int, radius: int):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                return node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)
        return None


def reference_signature(path: Path):
    with Image.open(path) as source:
        return signatures(ImageOps.exif_transpose(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=PROJECT / "external/appleleaf9")
    parser.add_argument("--extra", type=Path, default=PROJECT / "dataset_natural_extra")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    staging, extra = args.staging.resolve(), args.extra.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    tree_path = staging / "tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(fetch("https://api.github.com/repos/JasonYangCode/AppleLeaf9/git/trees/main?recursive=1"))
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise RuntimeError("Incomplete repository tree")
    commit = tree["sha"]
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob"
               and entry["path"].startswith("Frogeye leaf spot/")
               and Path(entry["path"]).suffix.lower() in EXTENSIONS]
    if not entries:
        raise RuntimeError("Published frogeye class was not found")
    raw = staging / "raw"
    raw.mkdir(exist_ok=True)

    def download(entry):
        path = raw / (entry["sha"] + ".jpg")
        url = f"https://raw.githubusercontent.com/JasonYangCode/AppleLeaf9/{commit}/" + urllib.parse.quote(entry["path"], safe="/")
        try:
            payload = path.read_bytes() if path.exists() else fetch(url)
            blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if blob != entry["sha"]:
                raise ValueError("Git blob checksum mismatch")
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            if not path.exists():
                path.write_bytes(payload)
            return {"source_path": entry["path"], "source_sha": entry["sha"], "source_url": url,
                    "raw_file": str(path), "download_error": ""}
        except Exception as error:
            return {"source_path": entry["path"], "source_sha": entry["sha"], "source_url": url,
                    "raw_file": str(path), "download_error": str(error)}

    rows = []
    print(f"Downloading {len(entries)} labeled photos (repository tree {commit})...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, row in enumerate(executor.map(download, entries), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(entries):
                print(f"Downloaded/checked {index}/{len(entries)}", flush=True)
    (staging / "download_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.download_only:
        return

    references = []
    for root in (PROJECT / "dataset_classification", PROJECT / "dataset_segmentation_plantseg/images", extra / "train"):
        references.extend(path for path in root.rglob("*") if path.is_file()
                          and path.suffix.lower() in EXTENSIONS and not path.name.startswith(PREFIX))
    references.sort()
    dhashes, phashes = HashTree(), HashTree()
    print(f"Hashing {len(references)} existing images across all splits...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, (path, hashes) in enumerate(zip(references, executor.map(reference_signature, references)), 1):
            dhashes.add(hashes[0], str(path))
            phashes.add(hashes[1], str(path))
            if index % 2000 == 0:
                print(f"Reference hashes: {index}/{len(references)}", flush=True)

    destination_dir = extra / "train" / CLASS
    destination_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    for index, row in enumerate(rows, 1):
        row.update(class_name=CLASS, license="CC BY 4.0", status="", duplicate_of="", output_file="")
        if row["download_error"]:
            row["status"] = "download_failed"
            continue
        with Image.open(row["raw_file"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if min(image.size) < 128:
            row["status"] = "too_small"
            continue
        match = None
        variants = [image] + [image.transpose(method) for method in (
            Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
            Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE)]
        hashes = []
        for variant in variants:
            dhash, phash = signatures(variant)
            hashes.append((dhash, phash))
            match = dhashes.match(dhash, 2) or phashes.match(phash, 6)
            if match:
                break
        if match:
            row.update(status="near_duplicate", duplicate_of=match)
            previous = destination_dir / (PREFIX + row["source_sha"][:20] + ".jpg")
            if previous.is_file() and previous not in accepted:
                # On reruns, a newly added reference may invalidate an old import.
                # Keep a recoverable copy outside the training directory.
                quarantine = extra / "rejected/appleleaf9" / previous.name
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                if quarantine.exists():
                    if previous.read_bytes() != quarantine.read_bytes():
                        raise RuntimeError(f"Quarantine filename conflict: {quarantine}")
                    previous.unlink()
                else:
                    shutil.move(str(previous), str(quarantine))
        else:
            destination = destination_dir / (PREFIX + row["source_sha"][:20] + ".jpg")
            if not destination.exists():
                shutil.copy2(row["raw_file"], destination)
            row.update(status="accepted", output_file=str(destination))
            accepted.append(destination)
            # Add normal and transformed hashes to suppress augmented copies too.
            for dhash, phash in hashes:
                dhashes.add(dhash, str(destination))
                phashes.add(phash, str(destination))
        if index % 100 == 0:
            print(f"Deduplicated {index}/{len(rows)}; accepted {len(accepted)}", flush=True)

    manifest = extra / "appleleaf9_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"repository": REPOSITORY, "repository_tree_sha": commit, "license": "CC BY 4.0",
              "source_class": "Frogeye leaf spot", "mapped_class": CLASS,
              "label_basis": "Published expert-screened class; visual sample checks only, not an independent expert review of every image",
              "counts": dict(Counter(row["status"] for row in rows)),
              "reference_images": len(references), "split": "train only",
              "deduplication": "64-bit dHash <=2 OR pHash <=6, with all 8 rotations/reflections; conservative similarity filter, not proof of scene independence",
              "segmentation_masks_added": 0}
    (extra / "appleleaf9_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted:
        chosen = [accepted[int(i)] for i in np.linspace(0, len(accepted) - 1, min(24, len(accepted))) ]
        sheet = Image.new("RGB", (1200, 4 * 160), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(chosen):
            with Image.open(path) as image:
                preview = ImageOps.contain(image.convert("RGB"), (195, 130))
            x, y = index % 6 * 200, index // 6 * 160
            sheet.paste(preview, (x, y))
            draw.text((x, y + 132), path.stem[-12:], fill="black")
        sheet.save(staging / "accepted_preview.jpg")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

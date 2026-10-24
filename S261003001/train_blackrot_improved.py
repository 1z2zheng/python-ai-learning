"""Train the clean mixed baseline, then refine with additional field photos."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check inputs without starting training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    splits = root / "runs/plantseg_split_audit/splits_deduplicated.json"
    mixed = root / "runs/mobilenet_v3_large_mixed_clean_appleleaf9"
    refined = root / "runs/mobilenet_v3_large_refined_appleleaf9"
    if not splits.is_file():
        raise FileNotFoundError(f"Missing reviewed splits: {splits}")
    images = list((root / "dataset_natural_extra/train/Apple_Black_rot").glob("appleleaf9_blackrot_*.jpg"))
    if not images:
        raise RuntimeError("No accepted AppleLeaf9 training images found")
    report_path = root / "dataset_natural_extra/appleleaf9_report.json"
    if not report_path.is_file():
        raise RuntimeError("Wait for the AppleLeaf9 import and duplicate screening to finish")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("counts", {}).get("accepted") != len(images):
        raise RuntimeError("AppleLeaf9 file count does not match the completed import report")
    for output in (mixed, refined):
        if output.exists():
            raise FileExistsError(f"Preserve the existing experiment; choose a new output name: {output}")
    print(f"AppleLeaf9 field training photos: {len(images)}", flush=True)
    if args.check_only:
        print("Training inputs are ready. No training was started.")
        return
    subprocess.run([sys.executable, str(root / "classification/train_mixed.py"),
                    "--plantseg-splits", str(splits), "--output", str(mixed)], cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "classification/train_refined.py"),
                    "--plantseg-splits", str(splits), "--checkpoint", str(mixed / "best.pt"),
                    "--output", str(refined), "--extra-share", "0.30"], cwd=root, check=True)
    print(f"Finished. Evaluation: {refined / 'metrics.json'}", flush=True)
    print("App checkpoint unchanged; compare the clean-split experiments before deploying.")


if __name__ == "__main__":
    main()

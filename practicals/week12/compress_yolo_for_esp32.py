"""Create board-compatible YOLO artifacts for the Practical 12 homework.

This is a student-facing wrapper around the validated ESP32 exporter from
week7. It produces FP16 and strict split-output INT8 TFLite artifacts and
prints the files that should be evaluated before flashing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WEEK12_DIR = Path(__file__).resolve().parent
WEEK7_DIR = WEEK12_DIR.parent / "week7"
EXPORTER = WEEK7_DIR / "export_yolo11n_esp32.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Retrained YOLO .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML for calibration/evaluation")
    parser.add_argument("--imgsz", type=int, default=224, help="Board input size")
    parser.add_argument("--representative-limit", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEEK12_DIR / "esp32_compression_artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "fp16", "int8"),
        default="both",
        help="Produce FP16, INT8, or both artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not EXPORTER.exists():
        raise FileNotFoundError(f"Week 7 exporter not found: {EXPORTER}")
    if not args.model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(EXPORTER),
        "--model",
        str(args.model.resolve()),
        "--data",
        str(args.data.resolve()),
        "--imgsz",
        str(args.imgsz),
        "--representative-limit",
        str(args.representative_limit),
        "--mode",
        args.mode,
        "--int8-output",
        "split",
        "--output-dir",
        str(output_dir),
    ]

    print("Running validated ESP32 exporter:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, cwd=WEEK7_DIR, check=True)

    stem = args.model.stem
    summary_path = output_dir / f"{stem}_compression_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("failures"):
            raise RuntimeError(f"Conversion failures reported in {summary_path}")

    print("\nArtifacts to evaluate:")
    for path in sorted(output_dir.glob(f"{stem}_*224.*")):
        print(f"  {path} ({path.stat().st_size / 1024**2:.3f} MiB)")
    print("\nThe board candidate is the *_int8_split_224.tflite file and its matching *_model.h header.")
    print("The firmware must use the INT8 clipping fix documented in week7/ESP32_YOLO_DEPLOYMENT.md.")


if __name__ == "__main__":
    main()

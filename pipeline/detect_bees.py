#!/usr/bin/env python3
"""
Pipeline Step 2: MegaDetector Animal / Bee Bounding Box Detection.

Runs MegaDetector batch inference across harvested images to produce bounding boxes.
"""

import argparse
from pathlib import Path


def detect_bees(image_dir: str, output_path: str, model_name: str = "MDV5A"):
    """Run MegaDetector batch inference on images in image_dir."""
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file

    img_path = Path(image_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    print(f"Starting MegaDetector ({model_name}) on: {image_dir}")
    results = load_and_run_detector_batch(
        model_file=model_name,
        image_file_names=str(img_path)
    )

    write_results_to_file(
        results,
        str(out_path),
        relative_path_base=str(img_path),
        detector_file=model_name
    )
    print(f"Detection complete. Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MegaDetector bounding box detection on images.")
    parser.add_argument("--image-dir", default="data/SEHbees/images", help="Path to images directory")
    parser.add_argument("--output-path", default="data/SEHbees/bbox_detections.json", help="Path to output bbox JSON")
    parser.add_argument("--model", default="MDV5A", help="MegaDetector model identifier (default: MDV5A)")
    args = parser.parse_args()

    detect_bees(args.image_dir, args.output_path, model_name=args.model)


if __name__ == "__main__":
    main()

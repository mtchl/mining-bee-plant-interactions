#!/usr/bin/env python3
"""
BioCLIP Mining Relations: Main Pipeline CLI Entry Point.

Extract ecological interactions (pollinator-plant relationships) from ALA occurrence records.

Usage:
  python run_pipeline.py harvest  [options]
  python run_pipeline.py detect   [options]
  python run_pipeline.py classify [options]
  python run_pipeline.py export   [options]
  python run_pipeline.py all      [options]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="BioCLIP Mining Relations: Ecological Interaction Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # Step 1: Harvest
    p_harvest = subparsers.add_parser("harvest", help="Step 1: Harvest ALA images and create manifest")
    p_harvest.add_argument("--csv-path", default="data/SEHbees/records.csv", help="Path to ALA occurrences CSV")
    p_harvest.add_argument("--output-dir", default="data/SEHbees/images", help="Directory for images")
    p_harvest.add_argument("--manifest-path", default="data/SEHbees/manifest.json", help="Path to manifest JSON")
    p_harvest.add_argument("--limit", type=int, default=None, help="Limit images harvested (for testing)")
    p_harvest.add_argument("--delay", type=float, default=0.5, help="Polite delay between requests (seconds)")

    # Step 2: Detect
    p_detect = subparsers.add_parser("detect", help="Step 2: Run MegaDetector animal/bee detection")
    p_detect.add_argument("--image-dir", default="data/SEHbees/images", help="Images directory")
    p_detect.add_argument("--output-path", default="data/SEHbees/bbox_detections.json", help="Output bbox JSON")
    p_detect.add_argument("--model", default="MDV5A", help="MegaDetector model identifier")

    # Step 3: Classify
    p_classify = subparsers.add_parser("classify", help="Step 3: Redact bees, SVM pre-filter, and BioCLIP classify")
    p_classify.add_argument("--manifest", default="data/SEHbees/manifest.json", help="Manifest JSON")
    p_classify.add_argument("--bbox-file", default="data/SEHbees/bbox_detections.json", help="Bbox detections JSON")
    p_classify.add_argument("--images-dir", default="data/SEHbees/images", help="Images directory")
    p_classify.add_argument("--svm-model", default="models/plant_filter_svm.joblib", help="SVM filter model path")
    p_classify.add_argument("--output-path", default="data/SEHbees/detections-genus-annotated.json", help="Output annotated JSON")
    p_classify.add_argument("--threshold", type=float, default=0.30, help="SVM decision threshold (default: 0.30)")
    p_classify.add_argument("--limit", type=int, default=None, help="Limit number of images processed")
    p_classify.add_argument("--device", default=None, help="Compute device ('mps', 'cuda', 'cpu')")

    # Step 4: Export
    p_export = subparsers.add_parser("export", help="Step 4: Export clean UI and public research datasets")
    p_export.add_argument("--input-json", default="data/SEHbees/detections-genus-annotated.json", help="Input annotated JSON")
    p_export.add_argument("--output-dir", default="data/SEHbees", help="Output directory")
    p_export.add_argument("--decimals", type=int, default=4, help="Float decimal precision")

    # All steps
    p_all = subparsers.add_parser("all", help="Run entire pipeline end-to-end")
    p_all.add_argument("--csv-path", default="data/SEHbees/records.csv", help="Path to ALA occurrences CSV")
    p_all.add_argument("--output-dir", default="data/SEHbees", help="Output root directory")
    p_all.add_argument("--limit", type=int, default=None, help="Limit number of records processed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "harvest":
        from pipeline.harvest import harvest
        harvest(args.csv_path, args.output_dir, args.manifest_path, limit=args.limit, delay=args.delay)

    elif args.command == "detect":
        from pipeline.detect_bees import detect_bees
        detect_bees(args.image_dir, args.output_path, model_name=args.model)

    elif args.command == "classify":
        from pipeline.classify_plants import classify_plants
        svm_path = args.svm_model
        if not Path(svm_path).exists() and Path("outputs/plant_filter_svm.joblib").exists():
            svm_path = "outputs/plant_filter_svm.joblib"
        classify_plants(
            manifest_path=args.manifest,
            bbox_path=args.bbox_file,
            images_dir=args.images_dir,
            svm_model_path=svm_path,
            output_path=args.output_path,
            svm_threshold=args.threshold,
            limit=args.limit,
            device=args.device
        )

    elif args.command == "export":
        from pipeline.export import export_clean_datasets
        export_clean_datasets(args.input_json, args.output_dir, round_decimals=args.decimals)

    elif args.command == "all":
        out_base = Path(args.output_dir)
        images_dir = out_base / "images"
        manifest_path = out_base / "manifest.json"
        bbox_path = out_base / "bbox_detections.json"
        annotated_path = out_base / "detections-genus-annotated.json"

        print("=== Step 1: Harvesting ===")
        from pipeline.harvest import harvest
        harvest(args.csv_path, str(images_dir), str(manifest_path), limit=args.limit)

        print("\n=== Step 2: MegaDetector Animal Detection ===")
        from pipeline.detect_bees import detect_bees
        detect_bees(str(images_dir), str(bbox_path))

        print("\n=== Step 3: Redaction, Plant Filter & BioCLIP Classification ===")
        from pipeline.classify_plants import classify_plants
        svm_path = "models/plant_filter_svm.joblib"
        if not Path(svm_path).exists() and Path("outputs/plant_filter_svm.joblib").exists():
            svm_path = "outputs/plant_filter_svm.joblib"
        classify_plants(
            manifest_path=str(manifest_path),
            bbox_path=str(bbox_path),
            images_dir=str(images_dir),
            svm_model_path=svm_path,
            output_path=str(annotated_path),
            limit=args.limit
        )

        print("\n=== Step 4: Exporting Clean UI & Public Formats ===")
        from pipeline.export import export_clean_datasets
        export_clean_datasets(str(annotated_path), str(out_base))

        print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()

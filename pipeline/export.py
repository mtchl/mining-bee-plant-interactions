#!/usr/bin/env python3
"""
Pipeline Step 4: Export Clean Datasets for Web UI and Public Research Reuse.

Generates:
1. detections-ui.json: Lightweight client-side JSON filtered strictly to plant detections
   with confidence > threshold (default: > 0.4), with unused/redundant fields removed.
2. detections-public.json: Complete research-grade JSON with spatial coordinates, provenance,
   and audit metrics (all observations).
3. detections-public.csv: Complete flat tabular CSV ready for R, Pandas, QGIS, and Darwin Core.
"""

import argparse
import csv
import json
from pathlib import Path


def export_clean_datasets(
    input_path: str,
    output_dir: str,
    min_confidence: float = 0.4,
    round_decimals: int = 4
):
    """
    Transform raw pipeline detections into optimized UI and public release formats.

    Args:
        input_path: Path to detections-genus-annotated.json.
        output_dir: Directory where exported files will be written.
        min_confidence: Minimum plant detection confidence score (> threshold)
                        to include in detections-ui.json (default: 0.4).
        round_decimals: Number of decimal places to round confidence scores.
    """
    in_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading annotated records from {input_path}...")
    with open(in_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    ui_records = []
    public_records = []

    for item in records:
        pd = item.get('plantDetection')
        has_plant = bool(item.get('hasPlant', False))

        # Standardize clean plantDetection object
        clean_pd = None
        if pd and has_plant:
            clean_pd = {
                "genus": pd.get("genus"),
                "score": round(float(pd["score"]), round_decimals) if pd.get("score") is not None else None,
                "nativeStatus": round(float(pd["nativeStatus"]), round_decimals) if pd.get("nativeStatus") is not None else None
            }

        # 1. Web UI format:
        # Filter ONLY plant detections exceeding the confidence threshold
        if has_plant and clean_pd and clean_pd.get("score") is not None and clean_pd["score"] > min_confidence:
            ui_rec = {
                "occurrenceID": item.get("occurrenceID", ""),
                "scientificName": item.get("scientificName", ""),
                "genus": item.get("genus", ""),
                "family": item.get("family", ""),
                "imageID": item.get("imageID", ""),
                "eventDate": item.get("eventDate", ""),
                "dataResourceName": item.get("dataResourceName", ""),
                "identifiedBy": item.get("identifiedBy", ""),
                "hasPlant": True,
                "plantConfidence": round(float(item["plantConfidence"]), round_decimals) if item.get("plantConfidence") is not None else None,
                "plantDetection": clean_pd
            }
            ui_records.append(ui_rec)

        # 2. Public Research format (complete dataset with coordinates, canonical image URLs, audit metrics)
        lat = float(item["decimalLatitude"]) if item.get("decimalLatitude") and item["decimalLatitude"] != "" else None
        lon = float(item["decimalLongitude"]) if item.get("decimalLongitude") and item["decimalLongitude"] != "" else None
        img_id = item.get("imageID", "")
        img_url = f"https://images.ala.org.au/image/{img_id}" if img_id else ""

        pub_rec = {
            "occurrenceID": item.get("occurrenceID", ""),
            "scientificName": item.get("scientificName", ""),
            "beeGenus": item.get("genus", ""),
            "beeFamily": item.get("family", ""),
            "eventDate": item.get("eventDate", ""),
            "decimalLatitude": lat,
            "decimalLongitude": lon,
            "imageID": img_id,
            "imageUrl": img_url,
            "dataResourceName": item.get("dataResourceName", ""),
            "identifiedBy": item.get("identifiedBy", ""),
            "boxesRedacted": int(item.get("boxesRedacted", 0)),
            "hasPlant": has_plant,
            "plantFilterConfidence": round(float(item["plantConfidence"]), round_decimals) if item.get("plantConfidence") is not None else None,
            "plantGenus": clean_pd["genus"] if clean_pd else None,
            "plantGenusScore": clean_pd["score"] if clean_pd else None,
            "plantGenusNativeStatus": clean_pd["nativeStatus"] if clean_pd else None
        }
        public_records.append(pub_rec)

    # Save unified UI JSON (filtered to plant detections > threshold)
    ui_path = out_dir / "detections-ui.json"
    with open(ui_path, 'w', encoding='utf-8') as f:
        json.dump(ui_records, f, indent=2)

    # Clean up deprecated detections-ui-plants-only.json if present
    deprecated_ui_plants = out_dir / "detections-ui-plants-only.json"
    if deprecated_ui_plants.exists():
        deprecated_ui_plants.unlink()

    # Save Public JSON
    pub_json_path = out_dir / "detections-public.json"
    with open(pub_json_path, 'w', encoding='utf-8') as f:
        json.dump(public_records, f, indent=2)

    # Save Public CSV
    pub_csv_path = out_dir / "detections-public.csv"
    csv_fields = list(public_records[0].keys())
    with open(pub_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(public_records)

    print(f"\nExport complete from {len(records)} source records:")
    print(f"  - Web UI (Unified, plant confidence > {min_confidence}): {ui_path} ({len(ui_records)} records)")
    print(f"  - Public Research JSON (All records):             {pub_json_path} ({len(public_records)} records)")
    print(f"  - Public Research CSV (All rows):                {pub_csv_path} ({len(public_records)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export clean UI and public research datasets.")
    parser.add_argument("--input-json", default="data/SEHbees/detections-genus-annotated.json", help="Path to input annotated JSON")
    parser.add_argument("--output-dir", default="data/SEHbees", help="Output directory for generated datasets")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Minimum plant detection score for UI export (default: 0.4)")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal rounding precision for float scores (default: 4)")
    args = parser.parse_args()

    export_clean_datasets(
        args.input_json,
        args.output_dir,
        min_confidence=args.min_confidence,
        round_decimals=args.decimals
    )


if __name__ == "__main__":
    main()

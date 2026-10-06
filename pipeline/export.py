#!/usr/bin/env python3
"""
Pipeline Step 4: Export Clean Datasets for Web UI and Public Research Reuse.

Generates:
1. detections-ui.json: Lightweight client-side JSON (stripped of unused/redundant fields).
2. detections-ui-plants-only.json: Compact interaction-only JSON for high-performance web loading.
3. detections-public.json: Research-grade JSON with spatial coordinates and provenance.
4. detections-public.csv: Flat tabular CSV ready for R, Pandas, QGIS, and Darwin Core workflows.
"""

import argparse
import csv
import json
from pathlib import Path


def export_clean_datasets(input_path: str, output_dir: str, round_decimals: int = 4):
    """Transform raw pipeline detections into optimized UI and public release formats."""
    in_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading annotated records from {input_path}...")
    with open(in_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    ui_full = []
    ui_plants_only = []
    public_records = []

    for item in records:
        pd = item.get('plantDetection')
        has_plant = bool(item.get('hasPlant', False))

        # Standardize plantDetection object
        clean_pd = None
        if pd and has_plant:
            clean_pd = {
                "genus": pd.get("genus"),
                "score": round(float(pd["score"]), round_decimals) if pd.get("score") is not None else None,
                "nativeStatus": round(float(pd["nativeStatus"]), round_decimals) if pd.get("nativeStatus") is not None else None
            }

        # 1. Web UI format (no coordinates, no local paths, no redundant arrays)
        ui_rec = {
            "occurrenceID": item.get("occurrenceID", ""),
            "scientificName": item.get("scientificName", ""),
            "genus": item.get("genus", ""),
            "family": item.get("family", ""),
            "imageID": item.get("imageID", ""),
            "eventDate": item.get("eventDate", ""),
            "dataResourceName": item.get("dataResourceName", ""),
            "identifiedBy": item.get("identifiedBy", ""),
            "hasPlant": has_plant,
            "plantConfidence": round(float(item["plantConfidence"]), round_decimals) if item.get("plantConfidence") is not None else None,
            "plantDetection": clean_pd
        }
        ui_full.append(ui_rec)
        if has_plant and clean_pd:
            ui_plants_only.append(ui_rec)

        # 2. Public Research format (spatial coordinates, canonical image URLs, audit metrics)
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

    # Save UI Full
    ui_full_path = out_dir / "detections-ui.json"
    with open(ui_full_path, 'w', encoding='utf-8') as f:
        json.dump(ui_full, f, indent=2)

    # Save UI Plants-Only
    ui_plants_path = out_dir / "detections-ui-plants-only.json"
    with open(ui_plants_path, 'w', encoding='utf-8') as f:
        json.dump(ui_plants_only, f, indent=2)

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

    print(f"\nExport complete for {len(records)} records:")
    print(f"  - Web UI (Full):           {ui_full_path} ({len(ui_full)} records)")
    print(f"  - Web UI (Plant-Positive): {ui_plants_path} ({len(ui_plants_only)} records)")
    print(f"  - Public Research JSON:    {pub_json_path} ({len(public_records)} records)")
    print(f"  - Public Research CSV:     {pub_csv_path} ({len(public_records)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export clean UI and public research datasets.")
    parser.add_argument("--input-json", default="data/SEHbees/detections-genus-annotated.json", help="Path to input annotated JSON")
    parser.add_argument("--output-dir", default="data/SEHbees", help="Output directory for generated datasets")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal rounding precision for float scores")
    args = parser.parse_args()

    export_clean_datasets(args.input_json, args.output_dir, round_decimals=args.decimals)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pipeline Step 1: Harvest ALA Occurrence Images and Metadata.

Reads an Atlas of Living Australia (ALA) occurrence CSV, extracts the first image ID,
downloads images politely with retry logic, and outputs a manifest JSON.
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ALA_HEADERS = {
    'User-Agent': 'BioCLIP-miningRelations-Bot/1.0 (mitchell.whitelaw@anu.edu.au; educational research)'
}


def extract_first_image_id(images_val: str) -> str | None:
    """Extract first image ID/UUID from pipe-separated images field."""
    if not images_val or not images_val.strip():
        return None
    parts = [p.strip() for p in images_val.split('|') if p.strip()]
    if not parts:
        return None
    first = parts[0]
    if first.startswith("http"):
        return first.split('/')[-1].strip()
    return first


def download_image(image_id: str, dest_path: Path, retries: int = 3, polite_delay: float = 0.5) -> bool:
    """Download single image from ALA Images API with retry logic and backoff."""
    url = f"https://api.ala.org.au/images/image/{image_id}/large"

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    req = urllib.request.Request(url, headers=ALA_HEADERS)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                    time.sleep(polite_delay)
                    return True
        except urllib.error.HTTPError as e:
            if attempt == retries:
                print(f"  [HTTP ERROR] {image_id}: {e.code} {e.reason}")
        except Exception as e:
            if attempt == retries:
                print(f"  [ERROR] {image_id}: {e}")

        if attempt < retries:
            time.sleep(attempt * 2)

    return False


def harvest(csv_path: str, output_dir: str, manifest_path: str, limit: int | None = None, delay: float = 0.5):
    """Harvest occurrences from CSV, download images, and write manifest JSON."""
    csv_file = Path(csv_path)
    images_dir = Path(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    print(f"Reading occurrences from {csv_path}...")
    candidates = []

    with open(csv_file, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Map column indices
        col_map = {col: idx for idx, col in enumerate(header)}
        required = ["images", "recordID", "scientificName", "genus", "family"]
        for req in required:
            if req not in col_map:
                raise ValueError(f"Missing required column in CSV: {req}")

        for row in reader:
            if len(row) <= col_map["images"]:
                continue
            images_val = row[col_map["images"]].strip()
            if not images_val:
                continue

            img_id = extract_first_image_id(images_val)
            if not img_id:
                continue

            candidates.append({
                "dataResourceName": row[col_map["dataResourceName"]].strip() if "dataResourceName" in col_map and len(row) > col_map["dataResourceName"] else "",
                "identifiedBy": row[col_map["identifiedBy"]].strip() if "identifiedBy" in col_map and len(row) > col_map["identifiedBy"] else "",
                "occurrenceID": row[col_map["recordID"]].strip(),
                "scientificName": row[col_map["scientificName"]].strip(),
                "genus": row[col_map["genus"]].strip(),
                "family": row[col_map["family"]].strip(),
                "imageID": img_id,
                "decimalLatitude": row[col_map["decimalLatitude"]].strip() if "decimalLatitude" in col_map and len(row) > col_map["decimalLatitude"] else "",
                "decimalLongitude": row[col_map["decimalLongitude"]].strip() if "decimalLongitude" in col_map and len(row) > col_map["decimalLongitude"] else "",
                "eventDate": row[col_map["eventDate"]].strip() if "eventDate" in col_map and len(row) > col_map["eventDate"] else "",
            })

            if limit and len(candidates) >= limit:
                break

    print(f"Found {len(candidates)} candidate occurrences with images.")
    manifest = []
    success_count = 0

    for idx, rec in enumerate(candidates, 1):
        img_id = rec["imageID"]
        dest_file = images_dir / f"{img_id}.jpg"
        if idx % 100 == 0 or idx == len(candidates):
            print(f"[{idx}/{len(candidates)}] Processing images (downloaded: {success_count})...")

        if download_image(img_id, dest_file, polite_delay=delay):
            success_count += 1
            manifest.append(rec)

    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Harvest complete. Successfully processed {success_count}/{len(candidates)} images.")
    print(f"Manifest written to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Harvest occurrence images and manifest from ALA CSV.")
    parser.add_argument("--csv-path", default="data/SEHbees/records.csv", help="Path to ALA records CSV")
    parser.add_argument("--output-dir", default="data/SEHbees/images", help="Directory to save downloaded images")
    parser.add_argument("--manifest-path", default="data/SEHbees/manifest.json", help="Path to output manifest JSON")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on records for testing")
    parser.add_argument("--delay", type=float, default=0.5, help="Polite delay between requests in seconds")
    args = parser.parse_args()

    harvest(args.csv_path, args.output_dir, args.manifest_path, limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()

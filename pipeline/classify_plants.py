#!/usr/bin/env python3
"""
Pipeline Step 3: Redact Insect BBoxes, SVM Plant Pre-filter, BioCLIP Genus Classification,
and ALA Nativeness Annotation.

Pipeline logic:
1. Multi-target bounding box redaction with confidence threshold (>= 0.25) and max area (< 0.50).
2. BioCLIP visual embedding extraction.
3. Linear SVM pre-filter probability check (>= 0.30) to verify identifiable plant presence.
4. BioCLIP TreeOfLifeClassifier genus prediction (Rank.GENUS, k=1, Kingdom: Plantae).
5. In-memory cached ALA establishmentMeans API query for native status ratio.
"""

import argparse
import json
from pathlib import Path
import time
import cv2
import joblib
import numpy as np
from PIL import Image
import requests
import torch
from bioclip import Rank, TreeOfLifeClassifier

ALA_HEADERS = {
    'User-Agent': 'BioCLIP-miningRelations-Bot/1.0 (mitchell.whitelaw@anu.edu.au; educational research)'
}


def convert_img(img_bgr):
    """Convert OpenCV BGR image to PIL RGB Image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def robust_blur_detections(img_path: str, detections: list, min_conf: float = 0.25, max_area: float = 0.50, allowed_categories: set = {'1'}):
    """
    Robust multi-target bounding box redaction:
    - Filters out low-confidence boxes (< min_conf).
    - Filters out oversized boxes (> max_area).
    - Blurs qualified animal/bee boxes with adaptive Gaussian blur kernel.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, 0

    h, w, _ = img.shape
    if not detections:
        return convert_img(img), 0

    qualified_boxes = []
    for det in detections:
        cat = str(det.get('category', '1'))
        conf = det.get('conf', 0.0)
        bbox = det.get('bbox', [0, 0, 0, 0])
        bw, bh = bbox[2], bbox[3]
        area = bw * bh

        if cat in allowed_categories and conf >= min_conf and area <= max_area:
            qualified_boxes.append(det)

    if not qualified_boxes:
        return convert_img(img), 0

    img_blurred = img.copy()
    for det in qualified_boxes:
        x, y, bw, bh = det['bbox']
        x1, y1 = max(0, int(x * w)), max(0, int(y * h))
        x2, y2 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = img_blurred[y1:y2, x1:x2]
        bw_px = x2 - x1
        bh_px = y2 - y1

        k_size = int(min(bw_px, bh_px) * 0.3) | 1
        k_size = max(15, min(k_size, 151))

        blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
        img_blurred[y1:y2, x1:x2] = blurred_roi

    return convert_img(img_blurred), len(qualified_boxes)


def get_genus_nativeness(genus_name: str, cache: dict) -> float | None:
    """
    Query ALA API for genus establishmentMeans (native vs introduced) with local caching.
    Returns ratio: native_count / (native_count + introduced_count), or None if unavailable.
    """
    if not genus_name:
        return None

    if genus_name in cache:
        return cache[genus_name]

    url = f"https://api.ala.org.au/occurrences/occurrences/search?q=genus:{genus_name}&facets=establishmentMeans&im=false"
    try:
        time.sleep(0.05)  # Polite sleep
        response = requests.get(url, headers=ALA_HEADERS, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            facet_results = res_data.get("facetResults", [])
            native_count = 0
            introduced_count = 0

            for facet in facet_results:
                if facet.get("fieldName") == "establishmentMeans":
                    for item in facet.get("fieldResult", []):
                        label = item.get("label", "").lower()
                        count = item.get("count", 0)
                        if label == "native":
                            native_count += count
                        elif label == "introduced":
                            introduced_count += count

            total = native_count + introduced_count
            if total > 0:
                nativeness = round(native_count / total, 4)
                cache[genus_name] = nativeness
                return nativeness
    except Exception as e:
        print(f"    [ALA Nativeness Error] {genus_name}: {e}")

    cache[genus_name] = None
    return None


def classify_plants(
    manifest_path: str,
    bbox_path: str,
    images_dir: str,
    svm_model_path: str,
    output_path: str,
    svm_threshold: float = 0.30,
    min_bbox_conf: float = 0.25,
    max_bbox_area: float = 0.50,
    limit: int | None = None,
    checkpoint_every: int = 500,
    device: str | None = None
):
    """Run full classification pipeline on harvested images."""
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading SVM pre-filter from {svm_model_path} (threshold={svm_threshold})...")
    svm_filter = joblib.load(svm_model_path)

    print(f"Initializing BioCLIP TreeOfLifeClassifier on device '{device}'...")
    classifier = TreeOfLifeClassifier(device=device)
    plant_filter = classifier.create_taxa_filter(Rank.KINGDOM, ['Plantae'])
    classifier.apply_filter(plant_filter)

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    if limit:
        records = records[:limit]

    print(f"Loading MegaDetector bboxes from {bbox_path}...")
    with open(bbox_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    bbox_dict = {b['file']: b.get('detections', []) for b in bbox_data.get('images', [])}

    images_base = Path(images_dir)
    genus_nativeness_cache = {}
    plant_detection_count = 0
    total = len(records)

    print(f"\nProcessing {total} records...")
    for idx, item in enumerate(records):
        filename = f"{item['imageID']}.jpg"
        filepath = images_base / filename

        detections = bbox_dict.get(filename, [])
        blurred_img, num_blurred_boxes = robust_blur_detections(
            str(filepath),
            detections,
            min_conf=min_bbox_conf,
            max_area=max_bbox_area
        )

        item['boxesRedacted'] = num_blurred_boxes

        if blurred_img is None:
            item['hasPlant'] = False
            item['plantConfidence'] = 0.0
            item['plantDetection'] = None
            continue

        # 1. BioCLIP visual embedding
        with torch.no_grad():
            feat = classifier.create_image_features([blurred_img], normalize=True)
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()

        # 2. Linear SVM plant presence gate
        plant_prob = float(svm_filter.predict_proba(feat)[0, 1])
        has_plant = plant_prob >= svm_threshold

        item['plantConfidence'] = round(plant_prob, 4)
        item['hasPlant'] = has_plant

        # 3. BioCLIP Genus prediction if plant gate passed
        if has_plant:
            predictions = classifier.predict([blurred_img], Rank.GENUS, k=1)
            top_pred = predictions[0] if predictions else None
            top_genus = top_pred['genus'] if top_pred else None
            top_score = round(top_pred['score'], 4) if top_pred else 0.0

            if top_score > 0.2:
                plant_detection_count += 1

            native_status = get_genus_nativeness(top_genus, genus_nativeness_cache)

            item['plantDetection'] = {
                "genus": top_genus,
                "score": top_score,
                "nativeStatus": native_status
            }
        else:
            item['plantDetection'] = None

        if (idx + 1) % checkpoint_every == 0:
            print(f"  Checkpoint [{idx+1}/{total}]: Plants detected: {plant_detection_count}, Genera cached: {len(genus_nativeness_cache)}")
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print(f"\nProcessing complete for {total} records.")
    print(f"Plants detected: {plant_detection_count}")
    print(f"Unique plant genera cached: {len(genus_nativeness_cache)}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Redact bee bboxes and classify plant genera using BioCLIP and SVM pre-filter.")
    parser.add_argument("--manifest", default="data/SEHbees/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--bbox-file", default="data/SEHbees/bbox_detections.json", help="Path to bbox JSON")
    parser.add_argument("--images-dir", default="data/SEHbees/images", help="Path to images directory")
    parser.add_argument("--svm-model", default="models/plant_filter_svm.joblib", help="Path to trained SVM model")
    parser.add_argument("--output-path", default="data/SEHbees/detections-genus-annotated.json", help="Path to output JSON")
    parser.add_argument("--threshold", type=float, default=0.30, help="SVM decision threshold (default: 0.30)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed")
    parser.add_argument("--device", default=None, help="Compute device ('mps', 'cuda', or 'cpu')")
    args = parser.parse_args()

    # Fallback to outputs/ if models/ model is not found
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


if __name__ == "__main__":
    main()

"""
Pipeline step: Robust Bounding Box Redaction, SVM Plant Pre-Filter, BioCLIP Genus Classification,
and ALA Nativeness Annotation.

Enhancements:
1. Robust BBox handling:
   - Filters by category (animal/bee).
   - Only redacts boxes with confidence >= BBOX_MIN_CONF (0.25).
   - Skips oversized boxes exceeding BBOX_MAX_AREA (0.50) to prevent wiping out the scene.
   - Redacts all qualified boxes (multi-target handling) with adaptive Gaussian blur.
2. SVM pre-filtering gate:
   - Uses BioCLIP visual embedding with calibrated Linear SVM (threshold 0.30).
   - Skips genus classification on non-plant frames, preventing false positives (e.g. Tetradium/Stelis).
3. ALA Nativeness Annotation:
   - Queries ALA establishmentMeans facet for top identified plant genus with in-memory caching.
   - Bundles nativeStatus into top plant detection and top-level record.
"""

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

# Configuration paths and thresholds
MANIFEST_PATH = 'data/SEHbees/manifest.json'
BBOX_PATH = 'data/SEHbees/bbox_detections.json'
SVM_MODEL_PATH = 'models/plant_filter_svm.joblib' if Path('models/plant_filter_svm.joblib').exists() else 'outputs/plant_filter_svm.joblib'
OUTPUT_JSON_PATH = 'data/SEHbees/detections-genus-annotated.json'

SVM_DECISION_THRESHOLD = 0.30
BBOX_MIN_CONF = 0.25
BBOX_MAX_AREA = 0.50
ALLOWED_BBOX_CATEGORIES = {'1'}  # '1': animal (MegaDetector)

ALA_HEADERS = {
    'User-Agent': 'BioCLIP-miningRelations-Bot/1.0 (mitchell.whitelaw@anu.edu.au; educational research)'
}


def convert_img(img_bgr):
    """Convert OpenCV BGR image to PIL RGB Image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def robust_blur_detections(img_path, detections, min_conf=BBOX_MIN_CONF, max_area=BBOX_MAX_AREA, allowed_categories=ALLOWED_BBOX_CATEGORIES):
    """
    Robust multi-target bounding box redaction:
    - Rejects low-confidence detections (< min_conf) to avoid blurring flower centers/leaves.
    - Rejects oversized bounding boxes (> max_area) to avoid blurring the entire photograph.
    - Blurs all qualified boxes using an adaptive Gaussian blur kernel scaled to box dimensions.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, 0

    h, w, _ = img.shape
    if not detections or len(detections) == 0:
        return convert_img(img), 0

    # Filter detections
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
        # No boxes meet the confidence and area requirements; leave image unblurred
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

        # Adaptive Gaussian blur kernel: proportional to box size, odd integer >= 15
        k_size = int(min(bw_px, bh_px) * 0.3) | 1
        k_size = max(15, min(k_size, 151))

        blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
        img_blurred[y1:y2, x1:x2] = blurred_roi

    return convert_img(img_blurred), len(qualified_boxes)


def get_genus_nativeness(genus_name, cache):
    """
    Query ALA API for genus establishmentMeans (native vs introduced) with local in-memory caching.
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
                nativeness = native_count / total
                cache[genus_name] = nativeness
                print(f"    [ALA Nativeness] {genus_name}: {nativeness:.4f} (native={native_count}, introduced={introduced_count})")
                return nativeness
            else:
                print(f"    [ALA Nativeness] {genus_name}: No native/introduced status recorded")
        else:
            print(f"    [ALA Nativeness] API returned status {response.status_code} for {genus_name}")
    except Exception as e:
        print(f"    [ALA Nativeness] Error querying ALA for {genus_name}: {e}")

    cache[genus_name] = None
    return None


def main():
    print(f"Loading SVM pre-filter from {SVM_MODEL_PATH} (threshold={SVM_DECISION_THRESHOLD})...")
    svm_filter = joblib.load(SVM_MODEL_PATH)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Initializing TreeOfLifeClassifier on {device}...")
    classifier = TreeOfLifeClassifier(device=device)

    plant_filter = classifier.create_taxa_filter(Rank.KINGDOM, ['Plantae'])
    classifier.apply_filter(plant_filter)

    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'r') as f:
        data = json.load(f)

    print(f"Loading MegaDetector bboxes from {BBOX_PATH}...")
    with open(BBOX_PATH, 'r') as f:
        bbox_data = json.load(f)
    bbox_dict = {b['file']: b.get('detections', []) for b in bbox_data.get('images', [])}

    genus_nativeness_cache = {}
    plant_detection_count = 0
    total_images = len(data)

    print(f"\nStarting processing for {total_images} records...")
    for idx, item in enumerate(data):
        filename = item["imageID"] + ".jpg"
        filepath = f"data/SEHbees/images/{filename}"

        detections = bbox_dict.get(filename, [])
        blurred_img, num_blurred_boxes = robust_blur_detections(filepath, detections)

        if blurred_img is None:
            print(f"[{idx+1}/{total_images}] Warning: Could not read image at {filepath}, skipping.")
            item['hasPlant'] = False
            item['plantConfidence'] = 0.0
            item['plantDetection'] = None
            item['boxesRedacted'] = 0
            continue

        item['boxesRedacted'] = num_blurred_boxes

        # 1. Extract BioCLIP visual embedding for the pre-filter
        with torch.no_grad():
            feat = classifier.create_image_features([blurred_img], normalize=True)
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()

        # 2. Evaluate SVM pre-filter probability
        plant_prob = float(svm_filter.predict_proba(feat)[0, 1])
        has_plant = plant_prob >= SVM_DECISION_THRESHOLD

        item['plantConfidence'] = round(plant_prob, 4)
        item['hasPlant'] = has_plant

        # 3. Classify with BioCLIP only if plant is detected
        if has_plant:
            predictions = classifier.predict([blurred_img], Rank.GENUS, k=1)
            top_pred = predictions[0] if predictions else None
            top_genus = top_pred['genus'] if top_pred else None
            top_score = round(float(top_pred['score']), 4) if top_pred else 0.0

            if top_score > 0.2:
                plant_detection_count += 1
                print(f"[{idx+1}/{total_images}] Plant Detected (p={plant_prob:.3f}, redacted={num_blurred_boxes}): {top_genus} (score={top_score:.3f})")
            else:
                print(f"[{idx+1}/{total_images}] Plant Detected (p={plant_prob:.3f}) but top genus below 0.2 score ({top_genus}: {top_score:.3f})")

            # 4. Nativeness lookup for the top genus
            native_status = get_genus_nativeness(top_genus, genus_nativeness_cache)

            # Store the single top-ranked plant detection
            item['plantDetection'] = {
                "genus": top_genus,
                "score": top_score,
                "nativeStatus": native_status
            }
        else:
            item['plantDetection'] = None
            print(f"[{idx+1}/{total_images}] Pre-filtered (p={plant_prob:.3f} < {SVM_DECISION_THRESHOLD}, redacted={num_blurred_boxes}): No plant in frame.")

        # Periodic checkpoint save every 500 records
        if (idx + 1) % 500 == 0:
            print(f"  Checkpoint: processed {idx+1}/{total_images}. Plants detected: {plant_detection_count}. Cached genera: {len(genus_nativeness_cache)}.")
            with open(OUTPUT_JSON_PATH, 'w') as f:
                json.dump(data, f, indent=2)

    print(f"\nCompleted processing {total_images} records.")
    print(f"Total plants detected (>0.2 score): {plant_detection_count}")
    print(f"Total unique plant genera cached for nativeness: {len(genus_nativeness_cache)}")
    print(f"Saving final results to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    try:
        from pipeline.export import export_clean_datasets
        print("\nExporting clean Web UI and Public research datasets...")
        export_clean_datasets(OUTPUT_JSON_PATH, str(Path(OUTPUT_JSON_PATH).parent))
    except Exception as e:
        print(f"Note: Could not run automated export ({e}). Run 'python run_pipeline.py export' manually.")

    print("Done!")


if __name__ == '__main__':
    main()

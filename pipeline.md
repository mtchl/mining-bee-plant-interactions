# Pipeline

1. **Source data**: Download occurrences CSV for target query from ALA (e.g. `data/SEHbees/records.csv`).

2. **Harvest images (`pipeline/harvest.py` / `run_pipeline.py harvest`)**: Use data from CSV, download first image for each occurrence from ALA with exponential backoff and polite delay, save metadata fields and image IDs to `manifest.json`.

3. **Detect bees (`pipeline/detect_bees.py` / `run_pipeline.py detect`)**: Use MegaDetector (MDV5A) to find bounding boxes for animals/bees in all images, save results to `bbox_detections.json`.

4. **Redact, filter & classify plants (`pipeline/classify_plants.py` / `run_pipeline.py classify`)**:
   - Filter qualified MegaDetector bounding boxes (conf >= 0.25, area <= 0.50) and apply adaptive Gaussian blur.
   - Extract BioCLIP visual embedding and pre-filter with a calibrated Linear SVM (threshold 0.30) to verify an identifiable plant is in frame.
   - If plant gate passes, pass blurred image to BioCLIP with taxon filter `Kingdom:Plantae` to obtain top-ranked genus prediction (`Rank.GENUS`, `k=1`).
   - Query ALA `establishmentMeans` facet with caching to determine empirical nativeness ratio.
   - Save annotated records to `detections-genus-annotated.json`.

5. **Clean Data Export (`pipeline/export.py` / `run_pipeline.py export`)**:
   - Web UI export (`detections-ui.json` and `detections-ui-plants-only.json`): Prune unused fields (`decimalLatitude`, `decimalLongitude`, `originalImagesField`, `localPath`, `boxesRedacted`), remove duplicate `plantDetections[]` array, round floats to 4 decimals.
   - Public research export (`detections-public.json` and `detections-public.csv`): Retain spatial coordinates, ALA occurrence provenance, canonical image URLs, and flattened detection audit metrics for open science and Darwin Core reuse.
 





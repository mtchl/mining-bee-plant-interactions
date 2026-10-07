# Mining Bee-Plant Interactions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![BioCLIP 2](https://img.shields.io/badge/BioCLIP-2.0-brightgreen.svg)](https://imageomics.github.io/bioclip-2/)
[![MegaDetector](https://img.shields.io/badge/MegaDetector-v5a-orange.svg)](https://github.com/agentmorris/MegaDetector)
[![Atlas of Living Australia](https://img.shields.io/badge/Data-Atlas%20of%20Living%20Australia-yellow.svg)](https://www.ala.org.au/)

Experimental data-mining of bee-plant interactions (visitation) from biodiversity occurrence records. This repository implements an automated computer vision and data mining pipeline that processes occurrence records and photographs from the **Atlas of Living Australia (ALA)** to detect, identify and annotate plants visited by Australian native and introduced bees.

### Demo & Data
[Anthophiles](https://anthophiles.mtchl.net) demonstrates the results for a [dataset](https://doi.org/10.26197/ala.6f6eb117-96d3-4f33-b2bb-1990e6454104) of bee occurrences from the South Eastern Highlands bioregion. From c. 16,000 source images it yields around 3,300 high confidence interaction records. 

Full interaction data for this demo is included here as [JSON](seh-bees/seh-bees-plants.json) and [CSV](seh-bees/seh-bees-plants.csv)



## Pipeline Overview

1. **Harvesting (`pipeline/harvest.py`):** Downloads occurrence images from the Atlas of Living Australia (one image per record).
2. **Localization (`pipeline/detect_bees.py`):** Detects animal/bee bounding boxes using MegaDetector (MDV5A), saves bounding box data.
3. **Redaction, Filtering & Classification (`pipeline/classify_plants.py`):** Blurs insect bounding boxes to prevent visual interference, filters out images without a plant in frame, and classifies plant genus with BioCLIP. Optionally, queries the ALA's `establishmentMeans` field to estimate whether the plant genus is native.
4. **Data Export (`pipeline/export.py`):** Produces lightweight, high-confidence data for web interfaces and unfiltered tabular data for other applications.


## Lessons Learned

1. **Insect Redaction Improves Plant Identification Accuracy**  
   BioCLIP 2 is effective at identifying plants when constrained by a taxonomic filter (`Kingdom: Plantae`). But visual signals from the insect can dominate the embedding and distort classification. Localizing the bee with MegaDetector and applying a   Gaussian blur removes insect visual features and improves plant ID accuracy.

2. **Pre-Filtering Prevents Semantic Hallucinations on Non-Plant Frames**  
   When an occurrence record depicts a bee without a visible plant, constraining BioCLIP to `Kingdom: Plantae` forces the model to select the nearest plant taxon. This causes false positives due to visual or semantic collisions in BioCLIP. False IDs include the orchid genus *Stelis* (which collides with the bee genus *Stelis* as well as images of observers' hands and fingers) or *Tetradium* (the 'bee bee' tree). To filter out non-plant images a linear classifier was trained on the BioCLIP embeddings for 100 hand-labelled images.

3. **Genus-Level Classification**  
   Classifying plants to **genus level** with BioCLIP gave significantly better accuracy than attempting to identify species.

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch with Apple Silicon MPS (`torch.backends.mps.is_available()`) or CUDA GPU support

```bash
# Clone the repository
git clone https://github.com/<your-username>/miningRelations.git
cd miningRelations

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart & CLI Usage

A unified CLI runner is provided via `run_pipeline.py`.

### 1. Run the Demo Pipeline (with Sample Data)
The repository includes `sample_data/records.csv` (50 representative occurrences across 15 native and introduced bee genera). You can run a self-contained test of the entire pipeline end-to-end:

```bash
python run_pipeline.py all --csv-path sample_data/records.csv --output-dir sample_data/demo --limit 5
```

### 2. Run Full Pipeline End-to-End
```bash
python run_pipeline.py all --csv-path data/SEHbees/records.csv --output-dir data/SEHbees
```

### 3. Run Individual Steps

#### Step 1: Harvest Images & Build Manifest
```bash
python run_pipeline.py harvest \
  --csv-path data/SEHbees/records.csv \
  --output-dir data/SEHbees/images \
  --manifest-path data/SEHbees/manifest.json \
  --delay 0.5
```

#### Step 2: MegaDetector Animal Detection
```bash
python run_pipeline.py detect \
  --image-dir data/SEHbees/images \
  --output-path data/SEHbees/bbox_detections.json \
  --model MDV5A
```

#### Step 3: Redact Bees, Filter Plants & Classify Genera
```bash
python run_pipeline.py classify \
  --manifest data/SEHbees/manifest.json \
  --bbox-file data/SEHbees/bbox_detections.json \
  --images-dir data/SEHbees/images \
  --svm-model models/plant_filter_svm.joblib \
  --threshold 0.30 \
  --output-path data/SEHbees/detections-genus-annotated.json
```

#### Step 4: Export Clean Datasets (Web UI & Public Formats)
```bash
python run_pipeline.py export \
  --input-json data/SEHbees/detections-genus-annotated.json \
  --output-dir data/SEHbees \
  --min-confidence 0.4 \
  --decimals 4
```

#### Training a Custom Plant Filter (Optional)
The pre-trained model in `models/plant_filter_svm.joblib` was calibrated on Australian bee observations. If you adapt this pipeline to other insect groups (e.g. butterflies, hoverflies) or different photography environments, you can train a domain-specific Linear SVM filter using two folders:

```bash
python run_pipeline.py train-filter \
  --positive-dir path/to/images_with_plants \
  --negative-dir path/to/images_without_plants \
  --output models/custom_plant_filter.joblib
```
*(~50–100 images per folder are sufficient, as BioCLIP visual embeddings are linearly separable.)*

---

## Data Output Formats & Data Dictionary

### 1. Web UI Deliverable (`detections-ui.json`)
Engineered specifically for client-side web applications. Automatically filtered to plant detections with prediction confidence $> 0.4$ (configurable via `--min-confidence`). All local paths, redundant arrays, and unused spatial coordinates are stripped, and floating-point confidence values are rounded to 4 decimals (reducing file size from ~14 MB to **~1.6 MB**):

```json
{
  "occurrenceID": "18258d3a-219e-41a2-93ae-b4912134a805",
  "scientificName": "Apis (Apis) mellifera",
  "genus": "Apis",
  "family": "Apidae",
  "imageID": "7f858a9f-0bf2-4702-a1d5-468ac9bde2a3",
  "eventDate": "2014-09-15T04:39:00Z",
  "dataResourceName": "ClimateWatch",
  "identifiedBy": "",
  "hasPlant": true,
  "plantConfidence": 0.7528,
  "plantDetection": {
    "genus": "Prunus",
    "score": 0.9457,
    "nativeStatus": 0.0043
  }
}
```

### 2. Public Research Deliverable (`detections-public.csv` & `detections-public.json`)
Darwin Core and FAIR-aligned format suitable for ecological research, GIS mapping, and statistical analysis:

| Field | Type | Description |
| :--- | :--- | :--- |
| `occurrenceID` | string | Unique ALA occurrence UUID |
| `scientificName` | string | Recorded insect taxon name |
| `beeGenus` | string | Insect genus |
| `beeFamily` | string | Insect family |
| `eventDate` | ISO 8601 | Date and time of observation |
| `decimalLatitude` | float | WGS84 latitude coordinate |
| `decimalLongitude` | float | WGS84 longitude coordinate |
| `imageID` | string | ALA image identifier |
| `imageUrl` | string | Canonical public image URL (`https://images.ala.org.au/image/{imageID}`) |
| `dataResourceName` | string | ALA contributing dataset / citizen science provider |
| `identifiedBy` | string | Observer / identifier credit |
| `boxesRedacted` | integer | Number of MegaDetector bounding boxes redacted in image |
| `hasPlant` | boolean | Binary plant presence decision from calibrated SVM gate |
| `plantFilterConfidence` | float (0–1) | Calibrated probability of plant presence from visual embedding SVM |
| `plantGenus` | string | Top-ranked plant genus predicted by BioCLIP |
| `plantGenusScore` | float (0–1) | BioCLIP model prediction confidence |
| `plantGenusNativeStatus`| float (0–1) | Ratio of ALA Australian records flagged as native vs introduced |

---

## Repository Structure

```
├── pipeline/                   # Modular pipeline package
│   ├── __init__.py
│   ├── harvest.py              # Step 1: ALA Harvester
│   ├── detect_bees.py          # Step 2: MegaDetector batch detector
│   ├── classify_plants.py      # Step 3: Redaction, SVM gate, BioCLIP classifier
│   └── export.py               # Step 4: UI & Public dataset exporter
├── models/
│   └── plant_filter_svm.joblib # Calibrated Linear SVM model weights (537 KB)
├── sample_data/                # Minimal sample dataset (50-row records.csv + pre-harvested demo images)
├── run_pipeline.py             # Main CLI entrypoint
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions for large images, datasets, and archive/
└── README.md
```

---

## Citations & Acknowledgments

- **Atlas of Living Australia (ALA)**: Biodiversity data and images provided under Creative Commons licensing.
- **BioCLIP 2**: Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*. [Imageomics Institute](https://imageomics.github.io/bioclip-2/).
- **MegaDetector**: Beery et al., *MegaDetector for animal detection in camera trap and field imagery*. [AgentMorris / Microsoft AI for Earth](https://github.com/agentmorris/MegaDetector).

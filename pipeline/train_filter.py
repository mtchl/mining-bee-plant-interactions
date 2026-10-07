#!/usr/bin/env python3
"""
Utility: Train Custom Plant Presence SVM Pre-Filter.

Extracts BioCLIP visual embeddings for two image folders (positive: plant present,
negative: plant absent), fits a calibrated Linear SVM, and exports the joblib model.
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
import torch
from tqdm import tqdm
from bioclip import TreeOfLifeClassifier

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect_images_from_dir(directory: Path) -> list[Path]:
    """Collect all valid image file paths in a directory."""
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def extract_embeddings(image_paths: list[Path], classifier: TreeOfLifeClassifier, batch_size: int = 16) -> np.ndarray:
    """Extract normalized BioCLIP visual embeddings for a list of image paths."""
    all_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
            except Exception as e:
                print(f"Warning: could not read {p} ({e}), creating blank fallback.")
                batch_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        with torch.no_grad():
            feats = classifier.create_image_features(batch_imgs, normalize=True)
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            all_features.append(feats)

    return np.vstack(all_features)


def train_filter(
    positive_dir: str,
    negative_dir: str,
    output_model_path: str = "models/plant_filter_svm.joblib",
    c_param: float = 1.0,
    device: str | None = None
):
    """Train a calibrated Linear SVM plant presence filter from two folders."""
    pos_path = Path(positive_dir)
    neg_path = Path(negative_dir)
    out_file = Path(output_model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pos_files = collect_images_from_dir(pos_path)
    neg_files = collect_images_from_dir(neg_path)

    if not pos_files:
        raise ValueError(f"No valid images found in positive directory: {positive_dir}")
    if not neg_files:
        raise ValueError(f"No valid images found in negative directory: {negative_dir}")

    print(f"Found {len(pos_files)} positive images (plant present) in {positive_dir}")
    print(f"Found {len(neg_files)} negative images (no plant) in {negative_dir}")

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing BioCLIP on device '{device}'...")
    classifier = TreeOfLifeClassifier(device=device)

    all_files = pos_files + neg_files
    labels = np.array([1] * len(pos_files) + [0] * len(neg_files), dtype=np.int32)

    features = extract_embeddings(all_files, classifier)

    # 5-fold cross validation check
    n_splits = min(5, min(len(pos_files), len(neg_files)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_svc = SVC(kernel="linear", C=c_param, probability=True, random_state=42)
        cv_preds = cross_val_predict(cv_svc, features, labels, cv=cv)
        acc = accuracy_score(labels, cv_preds)
        print(f"\n{n_splits}-Fold Cross-Validation Accuracy: {acc * 100:.2f}%")
        print("\nCross-Validation Classification Report:")
        print(classification_report(labels, cv_preds, target_names=["No Plant", "Plant"]))

    print("Fitting final calibrated Linear SVM model...")
    model = SVC(kernel="linear", C=c_param, probability=True, random_state=42)
    model.fit(features, labels)

    joblib.dump(model, out_file)
    print(f"Model saved successfully to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a calibrated Linear SVM plant presence filter from two image folders."
    )
    parser.add_argument("--positive-dir", required=True, help="Directory containing images WITH plants")
    parser.add_argument("--negative-dir", required=True, help="Directory containing images WITHOUT plants")
    parser.add_argument("--output", default="models/plant_filter_svm.joblib", help="Path to save trained SVM model")
    parser.add_argument("--c", type=float, default=1.0, help="SVM regularization parameter C (default: 1.0)")
    parser.add_argument("--device", default=None, help="Compute device ('mps', 'cuda', 'cpu')")
    args = parser.parse_args()

    train_filter(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        output_model_path=args.output,
        c_param=args.c,
        device=args.device
    )


if __name__ == "__main__":
    main()

"""
Training Pipeline for Structural Mask Classifier.

Orchestrates data loading, feature extraction, model training, 
and saving of the baseline Random Forest model.
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

from src.data.dataset_builder import MaskDatasetBuilder
from src.features.geometric import extract_geometric_features
from src.models.rf_classifier import StructuralMaskRFClassifier
from src.analysis.dataset_dashboard import generate_dashboard
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_feature_matrix(masks: list[np.ndarray], image_areas: list[float] = None) -> tuple[np.ndarray, list[str]]:
    """Extracts geometric features for a list of masks and builds the feature matrix X."""
    features_list = []
    feature_names = []
    
    for i, mask in enumerate(masks):
        img_area = image_areas[i] if image_areas else None
        feats = extract_geometric_features(mask, image_area=img_area)
        if not feature_names:
            feature_names = list(feats.keys())
        features_list.append(list(feats.values()))
        
    return np.array(features_list), feature_names

def train_model(project_root: str, n_estimators: int, max_depth: int = None):
    """Main training routine."""
    project_root = Path(project_root)
    
    # 1. Load data
    logger.info("Initializing Dataset Builder...")
    builder = MaskDatasetBuilder(project_root)
    mask_ids, masks, labels, image_ids, image_areas = builder.load_labeled_masks()
    
    if len(labels) == 0:
        logger.error("No labeled data available. Run labeling workflow first.")
        return
        
    # 2. Extract features
    logger.info("Extracting geometric features...")
    X, feature_names = prepare_feature_matrix(masks, image_areas)
    y = np.array(labels)
    groups = np.array(image_ids)
    
    # Optional: Save interim features for inspection
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['mask_id'] = mask_ids
    df['image_id'] = groups
    df.to_csv(interim_dir / "extracted_features.csv", index=False)
    
    # 3. Split data (80/20 grouped) to PREVENT DATA LEAKAGE
    logger.info("Splitting data into train/test sets using GroupShuffleSplit...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_index, test_index = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]
        
    logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    
    # VALIDATION: Check for data leakage
    leakage = set(groups_train).intersection(set(groups_test))
    if leakage:
        logger.error(f"DATA LEAKAGE DETECTED! Overlapping image_ids: {leakage}")
        return
    else:
        logger.info("Validation passed: No data leakage detected between train and test groups.")
    
    # 4. Train model
    model = StructuralMaskRFClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)
    
    # 5. Log feature importances
    importances = model.get_feature_importances(feature_names)
    logger.info("Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        logger.info(f"  - {feat}: {imp:.4f}")
        
    # 6. Save models and test set splits for evaluation
    save_dir = project_root / "models"
    model.save(save_dir, prefix="structural_component")
    
    np.save(save_dir / "X_test.npy", X_test)
    np.save(save_dir / "y_test.npy", y_test)
    
    # Generate Dataset Progress Dashboard
    generate_dashboard(project_root)
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multiclass Structural Component Classifier")
    parser.add_argument("--project_root", type=str, default=".", help="Project Root Directory")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth of trees")
    args = parser.parse_args()
    
    train_model(args.project_root, args.n_estimators, args.max_depth)

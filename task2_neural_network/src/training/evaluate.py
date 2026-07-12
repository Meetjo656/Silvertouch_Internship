"""
Evaluation Pipeline for Structural Mask Classifier.

Loads the trained model and test splits to compute evaluation metrics 
like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.rf_classifier import StructuralMaskRFClassifier
from src.analysis.dataset_dashboard import generate_dashboard
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(project_root: str):
    """Evaluates the trained baseline model on the holdout test set."""
    project_root = Path(project_root)
    models_dir = project_root / "models"
    
    # 1. Load test data
    x_test_path = models_dir / "X_test.npy"
    y_test_path = models_dir / "y_test.npy"
    
    if not x_test_path.exists() or not y_test_path.exists():
        logger.error("Test data not found. Please run the training pipeline first.")
        return
        
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    logger.info(f"Loaded test set with {len(y_test)} samples.")
    
    # 2. Load Model
    model = StructuralMaskRFClassifier()
    try:
        model.load(models_dir, prefix="structural_component")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    # 3. Predict
    y_pred = model.predict(X_test)
    
    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    logger.info("--- Evaluation Metrics ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Macro Precision: {prec_macro:.4f}")
    logger.info(f"Macro Recall:    {rec_macro:.4f}")
    logger.info(f"Macro F1-Score:  {f1_macro:.4f}")
    logger.info("--------------------------")
    
    # 5. Confusion Matrix
    labels_unique = np.unique(np.concatenate((y_test, y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Save CM plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels_unique,
                yticklabels=labels_unique)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Multiclass Component RF)")
    plt.tight_layout()
    
    plot_path = project_root / "data" / "interim" / "confusion_matrix.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Saved confusion matrix plot to {plot_path}")
    
    # 6. Generate Classification Report
    class_report_str = classification_report(y_test, y_pred, zero_division=0)
    
    # 7. Write Markdown Report
    report_path = project_root / "data" / "interim" / "structural_component_report.md"
    with open(report_path, "w") as f:
        f.write("# Structural Component Classifier Report\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Macro Precision:** {prec_macro:.4f}\n")
        f.write(f"- **Macro Recall:** {rec_macro:.4f}\n")
        f.write(f"- **Macro F1-Score:** {f1_macro:.4f}\n\n")
        
        f.write("## Per-Class Metrics\n")
        f.write("```text\n")
        f.write(class_report_str)
        f.write("\n```\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
    logger.info(f"Generated comprehensive report at {report_path}")
    
    # 8. Generate Dataset Progress Dashboard
    generate_dashboard(project_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multiclass Structural Component Classifier")
    parser.add_argument("--project_root", type=str, default=".", help="Project Root Directory")
    args = parser.parse_args()
    
    evaluate_model(args.project_root)

"""
Feature Importance Report Generator.

Analyzes the trained Random Forest model and the extracted features
to generate a comprehensive markdown report detailing feature importance,
interpretations, and evaluation metrics (Precision, Recall, F1, CM).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.rf_classifier import StructuralMaskRFClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_report(project_root: str):
    """Generates the feature importance and evaluation report."""
    project_root = Path(project_root)
    models_dir = project_root / "models"
    interim_dir = project_root / "data" / "interim"
    
    report_path = interim_dir / "baseline_evaluation_report.md"
    
    # 1. Load test data and features
    try:
        X_test = np.load(models_dir / "X_test.npy")
        y_test = np.load(models_dir / "y_test.npy")
        df_features = pd.read_csv(interim_dir / "extracted_features.csv")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
        
    feature_names = [col for col in df_features.columns if col not in ['label', 'mask_id', 'image_id']]
        
    # 2. Load Model
    model = StructuralMaskRFClassifier()
    try:
        model.load(models_dir, prefix="baseline")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    # 3. Predict & Compute Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Feature Importance & Interpretation (Means)
    importances = model.get_feature_importances(feature_names)
    
    # Calculate means for Structural (1) and Non-Structural (0)
    means = df_features.groupby('label')[feature_names].mean().T
    if 0 not in means.columns or 1 not in means.columns:
        logger.error("Missing class in extracted features to calculate means.")
        return
        
    means.columns = ['Mean_Non_Structural', 'Mean_Structural']
    
    # Compile into a DataFrame
    imp_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    }).set_index('Feature')
    
    analysis_df = imp_df.join(means).sort_values(by='Importance', ascending=False)
    
    # 5. Determine Contribution Direction
    analysis_df['Primary_Indicator_For'] = np.where(
        analysis_df['Mean_Structural'] > analysis_df['Mean_Non_Structural'], 
        'Structural (Higher Values)', 
        'Non-Structural (Higher Values)'
    )
    
    # 6. Generate Markdown Report
    top_10 = analysis_df.head(10)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Baseline Model Evaluation & Feature Analysis Report\n\n")
        f.write("## 1. Classification Metrics (Test Set)\n")
        f.write(f"- **Accuracy**: {acc:.4f}\n")
        f.write(f"- **Precision**: {prec:.4f}\n")
        f.write(f"- **Recall**: {rec:.4f}\n")
        f.write(f"- **F1-Score**: {f1:.4f}\n\n")
        
        f.write("## 2. Confusion Matrix\n")
        f.write("| | Predicted Non-Structural | Predicted Structural |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Actual Non-Structural** | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| **Actual Structural** | {cm[1][0]} | {cm[1][1]} |\n\n")
        
        f.write("## 3. Top 10 Most Important Features\n")
        f.write("| Rank | Feature | Importance | Mean (Structural) | Mean (Non-Structural) | Strongest Indicator For |\n")
        f.write("|---|---|---|---|---|---|\n")
        for rank, (feat, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"| {rank} | `{feat}` | {row['Importance']:.4f} | {row['Mean_Structural']:.2f} | {row['Mean_Non_Structural']:.2f} | **{row['Primary_Indicator_For']}** |\n")
            
        f.write("\n## 4. Interpretation of Top Features\n")
        for feat, row in top_10.iterrows():
            direction = "higher" if row['Mean_Structural'] > row['Mean_Non_Structural'] else "lower"
            f.write(f"- **{feat}**: Represents {row['Importance']*100:.1f}% of the model's decision criteria. Structural components tend to have significantly {direction} values for this feature compared to non-structural elements.\n")
            
    logger.info(f"Report successfully generated at {report_path}")
    
    # Save Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10['Importance'], y=top_10.index, palette="viridis")
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig(interim_dir / "feature_importance.png")
    logger.info("Saved feature importance plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evaluation Report")
    parser.add_argument("--project_root", type=str, default=".", help="Project Root Directory")
    args = parser.parse_args()
    
    generate_report(args.project_root)

"""
Random Forest Classifier Module.

Provides a wrapper around scikit-learn's RandomForestClassifier for 
predicting structural vs non-structural masks based on geometric features.
"""

from typing import Tuple, Dict, Any
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StructuralMaskRFClassifier:
    """Baseline Random Forest Classifier using Geometric Features."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        """
        Initialize the Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the tree.
            random_state (int): Random seed for reproducibility.
        """
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced", # Handle class imbalance
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model and scaler.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        logger.info(f"Training Random Forest on {len(X)} samples with {X.shape[1]} features.")
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.is_trained = True
        logger.info("Training complete.")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calling predict.")
        X_scaled = self.scaler.transform(X)
        return self.rf.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calling predict_proba.")
        X_scaled = self.scaler.transform(X)
        return self.rf.predict_proba(X_scaled)
        
    def get_feature_importances(self, feature_names: list[str]) -> Dict[str, float]:
        """Returns feature importances mapped to their names."""
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        importances = self.rf.feature_importances_
        return dict(zip(feature_names, importances))
        
    def save(self, save_dir: str | Path, prefix: str = "model") -> None:
        """Save the model and scaler to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"{prefix}_rf.joblib"
        scaler_path = save_dir / f"{prefix}_scaler.joblib"
        
        joblib.dump(self.rf, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")
        
    def load(self, load_dir: str | Path, prefix: str = "model") -> None:
        """Load the model and scaler from disk."""
        load_dir = Path(load_dir)
        model_path = load_dir / f"{prefix}_rf.joblib"
        scaler_path = load_dir / f"{prefix}_scaler.joblib"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model or scaler not found in {load_dir}")
            
        self.rf = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        logger.info("Loaded model and scaler successfully.")

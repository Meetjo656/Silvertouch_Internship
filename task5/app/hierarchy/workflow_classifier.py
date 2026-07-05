import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, cast

class WorkflowClassifier:
    """
    Classifies a user utterance (represented as an embedding) into a Workflow.
    Purpose: Sentence -> Workflow
    """
    def __init__(self):
        self.clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, X: np.ndarray, y: List[str]) -> None:
        """
        Trains the workflow classifier.
        X: 2D numpy array of sentence embeddings.
        y: list of workflow string labels.
        """
        y_encoded = self.label_encoder.fit_transform(y)
        self.clf.fit(cast(Any, X), y_encoded)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> str:
        """
        Predicts the top workflow for a given embedding X.
        X: 1D or 2D numpy array.
        Returns the predicted workflow label.
        """
        if not self.is_trained:
            raise RuntimeError("WorkflowClassifier must be trained before predicting.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        y_pred_encoded = self.clf.predict(cast(Any, X))
        return self.label_encoder.inverse_transform(y_pred_encoded)[0]

    def predict_top_k(self, X: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Predicts the top k workflows and their confidences.
        X: 1D or 2D numpy array representing a single sentence embedding.
        """
        if not self.is_trained:
            raise RuntimeError("WorkflowClassifier must be trained before predicting.")
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        probs = self.clf.predict_proba(cast(Any, X))[0]
        top_k_indices = np.argsort(probs)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx])
            results.append({
                "workflow": label, 
                "confidence": confidence
            })
            
        return results

    def save(self, filepath: str) -> None:
        """Saves the classifier state to disk."""
        state = {
            "clf": self.clf,
            "label_encoder": self.label_encoder,
            "is_trained": self.is_trained
        }
        joblib.dump(state, filepath)

    def load(self, filepath: str) -> None:
        """Loads the classifier state from disk."""
        state = joblib.load(filepath)
        self.clf = state["clf"]
        self.label_encoder = state["label_encoder"]
        self.is_trained = state["is_trained"]
import numpy as np
from collections import defaultdict
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
except ImportError:
    # Fallback to simple calculations if sklearn is not installed
    accuracy_score = None

class ExperimentEvaluator:
    def __init__(self, y_true, y_pred, y_pred_proba=None, classes=None):
        """
        y_true: list of actual intents
        y_pred: list of predicted intents (top 1)
        y_pred_proba: list of dictionaries mapping intent -> probability
        classes: list of all possible intents
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba or []
        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = list(set(y_true + y_pred))

    def accuracy(self):
        if accuracy_score:
            return accuracy_score(self.y_true, self.y_pred)
        return sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == yp) / max(1, len(self.y_true))

    def precision(self):
        if precision_score:
            return precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        return 0.0 # Placeholder without sklearn

    def recall(self):
        if recall_score:
            return recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        return 0.0 # Placeholder without sklearn

    def f1(self):
        if f1_score:
            return f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        return 0.0 # Placeholder without sklearn

    def top1(self):
        return self.accuracy()

    def top3(self):
        if not self.y_pred_proba:
            return self.accuracy() # Fallback to top1 if no probabilities
        
        correct = 0
        for yt, probas in zip(self.y_true, self.y_pred_proba):
            # Sort intents by probability
            top3_intents = sorted(probas, key=probas.get, reverse=True)[:3]
            if yt in top3_intents:
                correct += 1
        return correct / max(1, len(self.y_true))

    def calculate_confusion_matrix(self):
        if confusion_matrix:
            cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
            return cm.tolist()
        
        # Manual calculation
        cm = [[0]*len(self.classes) for _ in range(len(self.classes))]
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for yt, yp in zip(self.y_true, self.y_pred):
            if yt in class_to_idx and yp in class_to_idx:
                cm[class_to_idx[yt]][class_to_idx[yp]] += 1
        return cm

    def workflow_accuracy(self, y_true_wf, y_pred_wf):
        return sum(1 for yt, yp in zip(y_true_wf, y_pred_wf) if yt == yp) / max(1, len(y_true_wf))

    def entity_accuracy(self, y_true_ent, y_pred_ent):
        # A simple exact match metric
        return sum(1 for yt, yp in zip(y_true_ent, y_pred_ent) if yt == yp) / max(1, len(y_true_ent))

    def evaluate_all(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "top1": self.top1(),
            "top3": self.top3()
        }

if __name__ == "__main__":
    # Test execution
    evaluator = ExperimentEvaluator(
        y_true=["A", "B", "A", "C"],
        y_pred=["A", "B", "B", "C"],
        y_pred_proba=[{"A": 0.8, "B": 0.1}, {"A": 0.2, "B": 0.7}, {"A": 0.4, "B": 0.5}, {"C": 0.9}]
    )
    print(evaluator.evaluate_all())
    print("Confusion Matrix:", evaluator.calculate_confusion_matrix())

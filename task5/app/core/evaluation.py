"""
app/core/evaluation.py — 80/20 train/test split evaluation for intent classifiers.

Provides a realistic held-out accuracy estimate. The production classifier is
always fitted on 100% of data; this module only runs an evaluation pass.
"""

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("mybot-nlu")


def evaluate_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    workflow_id: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Split embeddings 80/20, train an evaluation-only LogisticRegression on 80%,
    and report accuracy on the held-out 20%.

    Returns a dict with keys:
        testAccuracy, trainAccuracy, testSamples, trainSamples,
        perClassReport, splitSkipped, splitSkipReason,
        overfitWarning, overfitGap
    """
    class_counts    = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0

    if min_class_count < 2:
        logger.info(
            f"  [SPLIT] Skipped for '{workflow_id}' — "
            f"min samples per class = {min_class_count} (need ≥ 2). "
            f"Add more examples per intent to enable held-out evaluation."
        )
        return {
            "testAccuracy":    None,
            "trainAccuracy":   None,
            "testSamples":     0,
            "trainSamples":    len(y),
            "perClassReport":  None,
            "splitSkipped":    True,
            "splitSkipReason": f"min samples per class = {min_class_count} (need ≥ 2)",
            "overfitWarning":  False,
            "overfitGap":      0.0,
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        eval_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=random_state,
        )
        eval_clf.fit(X_train, y_train)

        y_pred        = eval_clf.predict(X_test)
        test_accuracy = round(float(accuracy_score(y_test, y_pred)), 4)

        y_pred_train   = eval_clf.predict(X_train)
        train_accuracy = round(float(accuracy_score(y_train, y_pred_train)), 4)

        report = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        overfit_gap = train_accuracy - test_accuracy
        if overfit_gap > 0.15:
            logger.warning(
                f"  ⚠️  Possible overfitting for '{workflow_id}': "
                f"train={train_accuracy:.2%}, test={test_accuracy:.2%}, "
                f"gap={overfit_gap:.2%}. Add more diverse examples."
            )

        logger.info(
            f"  [SPLIT] '{workflow_id}': "
            f"train={len(X_train)} ({train_accuracy:.2%}), "
            f"test={len(X_test)} ({test_accuracy:.2%})"
        )

        return {
            "testAccuracy":    test_accuracy,
            "trainAccuracy":   train_accuracy,
            "testSamples":     len(X_test),
            "trainSamples":    len(X_train),
            "perClassReport":  report,
            "splitSkipped":    False,
            "splitSkipReason": None,
            "overfitWarning":  overfit_gap > 0.15,
            "overfitGap":      round(overfit_gap, 4),
        }

    except Exception as e:
        logger.warning(f"  [SPLIT] Evaluation failed for '{workflow_id}': {e}")
        return {
            "testAccuracy":    None,
            "trainAccuracy":   None,
            "testSamples":     0,
            "trainSamples":    len(y),
            "perClassReport":  None,
            "splitSkipped":    True,
            "splitSkipReason": str(e),
            "overfitWarning":  False,
            "overfitGap":      0.0,
        }

import json
import os
from datetime import datetime

class FeedbackCollector:
    def __init__(self, db_path="feedback_data.json"):
        self.db_path = db_path
        self.feedback_data = self._load_feedback_database()

    def _load_feedback_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_feedback_database(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f, indent=4)

    def collect(self, user_message, predicted_workflow, predicted_intent, predicted_subintent, predicted_entities, confidence, actual_workflow, actual_intent, actual_subintent, actual_entities, feedback_type="correction"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "prediction": {
                "workflow": predicted_workflow,
                "intent": predicted_intent,
                "subintent": predicted_subintent,
                "entities": predicted_entities,
                "confidence": confidence
            },
            "actual": {
                "workflow": actual_workflow,
                "intent": actual_intent,
                "subintent": actual_subintent,
                "entities": actual_entities
            },
            "feedback_type": feedback_type
        }
        self.feedback_data.append(entry)
        self._save_feedback_database()
        return entry

    def collect_positive(self, user_message, prediction, confidence):
        return self.collect(
            user_message,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            confidence,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            feedback_type="positive"
        )

    def collect_negative(self, user_message, prediction, confidence):
        # We know it's negative but we don't have the actual correction yet
        return self.collect(
            user_message,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            confidence,
            None, None, None, None,
            feedback_type="negative"
        )

    def collect_correction(self, user_message, prediction, confidence, actual_correction):
        return self.collect(
            user_message,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            confidence,
            actual_correction.get("workflow"), actual_correction.get("intent"), actual_correction.get("subintent"), actual_correction.get("entities"),
            feedback_type="correction"
        )

    def collect_missing_entity(self, user_message, prediction, confidence, actual_entities):
        return self.collect(
            user_message,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            confidence,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), actual_entities,
            feedback_type="missing_entity"
        )

    def collect_fallback(self, user_message, prediction, confidence):
        return self.collect(
            user_message,
            prediction.get("workflow"), prediction.get("intent"), prediction.get("subintent"), prediction.get("entities"),
            confidence,
            "fallback", "fallback", None, None,
            feedback_type="fallback"
        )

    def get_feedback(self):
        return self.feedback_data

    def get_recent_feedback(self, limit=10):
        return self.feedback_data[-limit:]

    def export_feedback(self, export_path="feedback_export.json"):
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f, indent=4)
        return export_path

    def clear_feedback(self):
        self.feedback_data = []
        self._save_feedback_database()
import json
import os
from .collector import FeedbackCollector

class DatasetBuilder:
    def __init__(self, feedback_source="feedback_data.json", training_data_path="training_data.json"):
        self.feedback_source = feedback_source
        self.training_data_path = training_data_path
        self.raw_feedback = []
        self.cleaned_feedback = []

    def load_feedback(self):
        collector = FeedbackCollector(db_path=self.feedback_source)
        self.raw_feedback = collector.get_feedback()
        return self.raw_feedback

    def clean_feedback(self):
        self.cleaned_feedback = []
        for entry in self.raw_feedback:
            # Only keep feedback that has actual corrections (skip raw negatives without corrections)
            if entry.get("feedback_type") in ["correction", "missing_entity"]:
                if entry.get("actual") and entry["actual"].get("intent"):
                    self.cleaned_feedback.append(entry)
        return self.cleaned_feedback

    def remove_duplicates(self):
        # Keep the latest correction for a given user message
        unique_feedback = {}
        for entry in self.cleaned_feedback:
            msg = entry.get("user_message", "").strip().lower()
            if msg:
                # Later entries overwrite earlier ones for the same message
                unique_feedback[msg] = entry
        
        self.cleaned_feedback = list(unique_feedback.values())
        return self.cleaned_feedback

    def validate_feedback(self):
        validated = []
        for entry in self.cleaned_feedback:
            actual = entry.get("actual", {})
            if actual.get("workflow") and actual.get("intent"):
                validated.append(entry)
        self.cleaned_feedback = validated
        return self.cleaned_feedback

    def merge_with_training_data(self):
        existing_data = []
        if os.path.exists(self.training_data_path):
            try:
                with open(self.training_data_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

        # Convert feedback entries to training data format
        new_samples = []
        for entry in self.cleaned_feedback:
            new_samples.append({
                "text": entry["user_message"],
                "workflow": entry["actual"]["workflow"],
                "intent": entry["actual"]["intent"],
                "subintent": entry["actual"]["subintent"],
                "entities": entry["actual"]["entities"],
                "source": "active_learning"
            })

        # Merge and deduplicate against existing training data
        existing_texts = {sample["text"].strip().lower() for sample in existing_data if "text" in sample}
        
        for sample in new_samples:
            if sample["text"].strip().lower() not in existing_texts:
                existing_data.append(sample)
                existing_texts.add(sample["text"].strip().lower())

        return existing_data

    def create_incremental_dataset(self, output_path="incremental_training_data.json"):
        merged_data = self.merge_with_training_data()
        self.export_json(merged_data, output_path)
        return output_path

    def export_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def statistics(self):
        return {
            "total_raw": len(self.raw_feedback),
            "total_cleaned": len(self.cleaned_feedback)
        }

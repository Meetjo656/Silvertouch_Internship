import os
import json
import glob

class ExperimentDatasetBuilder:
    def __init__(self, original_dir, merged_output_path):
        self.original_dir = original_dir
        self.merged_output_path = merged_output_path
        self.raw_samples = []
        self.merged_samples = []

    def load_all_training_files(self):
        file_paths = glob.glob(os.path.join(self.original_dir, "*.json"))
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict) and "workflowId" in data and "subIntents" in data:
                        workflow = data["workflowId"]
                        for sub_intent in data["subIntents"]:
                            intent = sub_intent.get("name", "unknown")
                            for example in sub_intent.get("examples", []):
                                self.raw_samples.append({
                                    "text": example,
                                    "intent": intent,
                                    "workflow": workflow
                                })
                    elif isinstance(data, list):
                        self.raw_samples.extend(data)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        return len(self.raw_samples)

    def validate_schema(self):
        valid_samples = []
        for sample in self.raw_samples:
            if "text" in sample and "intent" in sample and "workflow" in sample:
                valid_samples.append(sample)
        self.raw_samples = valid_samples
        return len(self.raw_samples)

    def merge_workflows(self):
        # Already merged in self.raw_samples through loading
        self.merged_samples = self.raw_samples

    def remove_duplicates(self):
        unique_samples = {}
        for sample in self.merged_samples:
            text = sample.get("text", "").strip().lower()
            if text:
                # Keep the last seen version of the text
                unique_samples[text] = sample
        self.merged_samples = list(unique_samples.values())
        return len(self.merged_samples)

    def generate_statistics(self):
        stats = {
            "total_raw": len(self.raw_samples),
            "total_merged_unique": len(self.merged_samples),
            "workflows": set(),
            "intents": set()
        }
        for sample in self.merged_samples:
            stats["workflows"].add(sample.get("workflow"))
            stats["intents"].add(sample.get("intent"))
        
        stats["workflows"] = list(stats["workflows"])
        stats["intents"] = list(stats["intents"])
        return stats

    def export(self):
        os.makedirs(os.path.dirname(self.merged_output_path), exist_ok=True)
        with open(self.merged_output_path, "w", encoding="utf-8") as f:
            json.dump(self.merged_samples, f, indent=4)
        return self.merged_output_path

if __name__ == "__main__":
    builder = ExperimentDatasetBuilder(
        original_dir=os.path.join(os.path.dirname(__file__), "..", "training_data"),
        merged_output_path=os.path.join(os.path.dirname(__file__), "datasets", "merged", "merged_training.json")
    )
    builder.load_all_training_files()
    builder.validate_schema()
    builder.merge_workflows()
    builder.remove_duplicates()
    stats = builder.generate_statistics()
    print("Dataset stats:", stats)
    builder.export()
    print("Dataset exported to", builder.merged_output_path)

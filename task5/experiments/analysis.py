import json
from collections import Counter

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self._load_data()

    def _load_data(self):
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def number_of_workflows(self):
        return len(set(sample.get("workflow") for sample in self.data if "workflow" in sample))

    def number_of_intents(self):
        return len(set(sample.get("intent") for sample in self.data if "intent" in sample))

    def number_of_subintents(self):
        return len(set(sample.get("subintent") for sample in self.data if sample.get("subintent")))

    def samples_per_intent(self):
        counts = Counter(sample.get("intent") for sample in self.data if "intent" in sample)
        return dict(counts)

    def duplicate_intents(self):
        # Intents with exactly the same samples? Or just highly similar ones?
        # A simple check: do we have intent names that are almost identical?
        intents = set(sample.get("intent") for sample in self.data if "intent" in sample)
        # Not easily computable without string distance. For now, returning None.
        return None

    def duplicate_examples(self):
        texts = [sample.get("text", "").strip().lower() for sample in self.data if "text" in sample]
        counts = Counter(texts)
        duplicates = {text: count for text, count in counts.items() if count > 1}
        return len(duplicates)

    def class_balance(self):
        counts = self.samples_per_intent()
        if not counts:
            return {}
        total = sum(counts.values())
        return {intent: count / total for intent, count in counts.items()}

    def entity_statistics(self):
        entity_keys = []
        for sample in self.data:
            entities = sample.get("entities", {})
            if isinstance(entities, dict):
                entity_keys.extend(entities.keys())
        counts = Counter(entity_keys)
        return dict(counts)

    def generate_report(self, output_path="dataset_report.txt"):
        report = []
        report.append("="*40)
        report.append("          DATASET REPORT")
        report.append("="*40)
        report.append(f"Total Examples: {len(self.data)}")
        report.append(f"Total Workflows: {self.number_of_workflows()}")
        report.append(f"Total Intents: {self.number_of_intents()}")
        report.append(f"Total Subintents: {self.number_of_subintents()}")
        report.append(f"Duplicate Example Texts: {self.duplicate_examples()}")
        
        report.append("\n--- Class Balance ---")
        balance = self.class_balance()
        for intent, ratio in sorted(balance.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {intent}: {ratio:.2%}")
            
        report.append("\n--- Entity Statistics ---")
        ent_stats = self.entity_statistics()
        for ent, count in sorted(ent_stats.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {ent}: {count} occurrences")
            
        report_text = "\n".join(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text

if __name__ == "__main__":
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "merged", "merged_training.json")
    if os.path.exists(dataset_path):
        analyzer = DatasetAnalyzer(dataset_path)
        report = analyzer.generate_report(os.path.join(os.path.dirname(__file__), "dataset_report.txt"))
        print(report)
    else:
        print(f"Dataset not found at {dataset_path}. Run dataset_builder.py first.")

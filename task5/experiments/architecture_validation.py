import json

class ArchitectureValidator:
    def __init__(self, baseline_results_path, new_results_path):
        self.baseline = self._load_results(baseline_results_path)
        self.new_results = self._load_results(new_results_path)
        
    def _load_results(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _calc_improvement(self, old_val, new_val, lower_is_better=False):
        if not old_val or not new_val:
            return "N/A"
        
        diff = new_val - old_val
        pct = (diff / old_val) * 100
        
        if lower_is_better:
            # e.g., latency drops from 100 to 50 -> -50% (improvement)
            return f"{pct:+.1f}%"
        else:
            # e.g., accuracy goes 90 to 95 -> +5.5% (improvement)
            return f"{pct:+.1f}%"

    def compare_model_count(self):
        old = self.baseline.get("pkl_count", 0)
        new = self.new_results.get("pkl_count", 0)
        return old, new, self._calc_improvement(old, new, lower_is_better=True)

    def compare_memory(self):
        old = self.baseline.get("memory_mb", 0)
        new = self.new_results.get("memory_mb", 0)
        return old, new, self._calc_improvement(old, new, lower_is_better=True)

    def compare_training(self):
        old = self.baseline.get("training_time_sec", 0)
        new = self.new_results.get("training_time_sec", 0)
        return old, new, self._calc_improvement(old, new, lower_is_better=True)

    def compare_accuracy(self):
        old = self.baseline.get("accuracy", 0.0)
        new = self.new_results.get("accuracy", 0.0)
        
        if old and new:
            # for accuracy diff we usually just show absolute pct point difference
            diff = (new - old) * 100
            imp = f"{diff:+.1f}%"
        else:
            imp = "N/A"
        return f"{old*100:.1f}%", f"{new*100:.1f}%", imp

    def compare_latency(self):
        old = self.baseline.get("inference_time_ms", 0)
        new = self.new_results.get("inference_time_ms", 0)
        return old, new, self._calc_improvement(old, new, lower_is_better=True)

    def summary(self, output_path="architecture_comparison.md"):
        md = []
        md.append("## Architecture Validation Summary")
        md.append("")
        md.append("| Metric | Old Architecture | New Global Model | Improvement |")
        md.append("|---|---:|---:|---:|")
        
        old, new, imp = self.compare_model_count()
        md.append(f"| PKL Files | {old} | {new} | {imp} |")
        
        old, new, imp = self.compare_training()
        md.append(f"| Training Time | {old:.1f}s | {new:.1f}s | {imp} |")
        
        old, new, imp = self.compare_latency()
        md.append(f"| Inference | {old:.1f}ms | {new:.1f}ms | {imp} |")
        
        old, new, imp = self.compare_accuracy()
        md.append(f"| Accuracy | {old} | {new} | {imp} |")
        
        old, new, imp = self.compare_memory()
        md.append(f"| Memory (MB) | {old:.1f} | {new:.1f} | {imp} |")
        
        md_text = "\n".join(md)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
        return md_text

if __name__ == "__main__":
    # Test script
    # Mock data for baseline and new
    with open("dummy_base.json", "w") as f:
        json.dump({"pkl_count": 17, "training_time_sec": 720, "inference_time_ms": 135, "accuracy": 0.921, "memory_mb": 500}, f)
    with open("dummy_new.json", "w") as f:
        json.dump({"pkl_count": 1, "training_time_sec": 420, "inference_time_ms": 82, "accuracy": 0.940, "memory_mb": 150}, f)
        
    validator = ArchitectureValidator("dummy_base.json", "dummy_new.json")
    print(validator.summary(None))

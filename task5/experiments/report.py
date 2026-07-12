import os

class ResearchReport:
    def __init__(self, dataset_report_path, error_report_path, arch_comparison_path):
        self.dataset_report_path = dataset_report_path
        self.error_report_path = error_report_path
        self.arch_comparison_path = arch_comparison_path

    def _read_file(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return f"[Content from {path} missing]"

    def methodology(self):
        return """
## 1. Methodology
We hypothesized that replacing per-workflow models with a shared global classifier would reduce model duplication without harming accuracy. 
To validate this, we merged the individual workflow datasets into a unified dataset. We evaluated the baseline system (multiple PKL files per workflow) against a new global Logistic Regression model using sentence embeddings as features. 
We measured training time, inference latency, memory footprint, and standard classification metrics.
"""

    def dataset(self):
        content = self._read_file(self.dataset_report_path)
        return f"## 2. Dataset\n\n```text\n{content}\n```\n"

    def experiments(self):
        return """
## 3. Experiments
- **Baseline**: Existing architecture using separate classifiers for each workflow.
- **Global Model**: A single unified Logistic Regression model trained on all intents simultaneously without hierarchical ranking.
"""

    def results(self):
        content = self._read_file(self.arch_comparison_path)
        return f"## 4. Results\n\n{content}\n"

    def discussion(self):
        error_content = self._read_file(self.error_report_path)
        return f"## 5. Discussion & Error Analysis\n\nThe following errors were captured during evaluation:\n```text\n{error_content}\n```\n"

    def limitations(self):
        return """
## 6. Limitations
- Sentence embeddings may struggle with highly domain-specific vocabulary.
- The unified model does not enforce workflow boundaries, which may lead to cross-workflow confusion if intents are highly similar.
"""

    def future_work(self):
        return """
## 7. Future Work
- Implement active learning retraining explicitly based on the error analysis.
- Add an IntentRanker to combine workflow context with global model predictions, ensuring we maintain high precision without sacrificing the architectural simplicity of the global model.
"""

    def generate(self, output_path="research_report.md"):
        report_sections = [
            "# NLU Architecture Redesign: Experimental Validation Report",
            self.methodology(),
            self.dataset(),
            self.experiments(),
            self.results(),
            self.discussion(),
            self.limitations(),
            self.future_work()
        ]
        
        full_report = "\n".join(report_sections)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_report)
        return full_report

if __name__ == "__main__":
    report = ResearchReport("dataset_report.txt", "error_report.txt", "architecture_comparison.md")
    print(report.generate())

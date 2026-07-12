from collections import defaultdict, Counter

class ErrorAnalyzer:
    def __init__(self, y_true, y_pred, y_pred_proba, texts, confidence_threshold=0.6):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.texts = texts
        self.confidence_threshold = confidence_threshold
        self.errors = []

    def collect_errors(self):
        self.errors = []
        for yt, yp, probas, text in zip(self.y_true, self.y_pred, self.y_pred_proba, self.texts):
            if yt != yp:
                conf = probas.get(yp, 0.0)
                self.errors.append({
                    "text": text,
                    "true_intent": yt,
                    "pred_intent": yp,
                    "confidence": conf
                })
        return self.errors

    def group_by_workflow(self, workflow_mapping):
        # workflow_mapping is a dict mapping intent -> workflow
        groups = defaultdict(list)
        for err in self.errors:
            wf = workflow_mapping.get(err["true_intent"], "Unknown")
            groups[wf].append(err)
        return dict(groups)

    def group_by_intent(self):
        groups = defaultdict(list)
        for err in self.errors:
            groups[err["true_intent"]].append(err)
        return dict(groups)

    def low_confidence(self):
        # Even if correct, track low confidence
        low_conf = []
        for yt, yp, probas, text in zip(self.y_true, self.y_pred, self.y_pred_proba, self.texts):
            conf = probas.get(yp, 0.0)
            if conf < self.confidence_threshold:
                low_conf.append({
                    "text": text,
                    "true_intent": yt,
                    "pred_intent": yp,
                    "confidence": conf,
                    "is_correct": (yt == yp)
                })
        return low_conf

    def confusion_pairs(self):
        pairs = Counter()
        for err in self.errors:
            pair = (err["true_intent"], err["pred_intent"])
            pairs[pair] += 1
        return pairs

    def generate_report(self, output_path="error_report.txt"):
        if not self.errors:
            self.collect_errors()
        
        report = []
        report.append("="*40)
        report.append("          ERROR ANALYSIS REPORT")
        report.append("="*40)
        report.append(f"Total Errors: {len(self.errors)}")
        
        report.append("\n--- Top Confusion Pairs ---")
        pairs = self.confusion_pairs()
        for (true_i, pred_i), count in pairs.most_common(10):
            report.append(f"{true_i}  ->  Confused with  ->  {pred_i} : {count} times")
            
        report.append("\n--- Low Confidence Predictions ---")
        low_conf = self.low_confidence()
        report.append(f"Found {len(low_conf)} predictions below {self.confidence_threshold} confidence.")
        for lc in low_conf[:5]: # just show top 5
            report.append(f"  [{'CORRECT' if lc['is_correct'] else 'WRONG'}] {lc['true_intent']} pred as {lc['pred_intent']} ({lc['confidence']:.2f}) - Text: {lc['text']}")

        report_text = "\n".join(report)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
        return report_text

if __name__ == "__main__":
    analyzer = ErrorAnalyzer(
        y_true=["Track Application", "Application Status", "Greeting"],
        y_pred=["Application Status", "Application Status", "Greeting"],
        y_pred_proba=[{"Application Status": 0.8}, {"Application Status": 0.9}, {"Greeting": 0.5}],
        texts=["where is my app", "status of my app", "hello"]
    )
    print(analyzer.generate_report())

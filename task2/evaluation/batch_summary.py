import os
import json

class BatchSummary:
    def __init__(self, output_root):
        self.output_root = output_root
        self.results = {}
        self.summary = None
    
    def add_result(self, image_name, rects):
        self.results[image_name] = rects

    def generate_report(self):
        self.summary = {
            "total": len(self.results),
            "segment_counts": {},
            "averages": {
                "total_segments": 0,
                "average_segments": 0
            }
        }

        for image_name, rects in self.results.items():
            count = len(rects)
            self.summary["segment_counts"][image_name] = count
            self.summary["averages"]["total_segments"] += count

        if self.summary["total"] > 0:
            self.summary["averages"]["average_segments"] = (
                self.summary["averages"]["total_segments"] / self.summary["total"]
            )

    def save_report(self):
        if self.summary is None:
            self.generate_report()
        report_path = os.path.join(self.output_root, "report.txt")
        with open(report_path, "w") as f:
            f.write(json.dumps(self.summary, indent=4))
        print(f"[INFO] Batch summary saved to {report_path}")

    def print_console_report(self):
        if self.summary is None:
            self.generate_report()
        print("""
        Batch Segmentation Report
        Total Images: {}
        Average Segments: {:.2f}
        """.format(self.summary["total"], self.summary["averages"]["average_segments"]))
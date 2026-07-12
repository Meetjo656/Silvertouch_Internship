from typing import Dict, Any, List
import json
import os

class ReportGenerator:
    """Generates benchmark reports in various formats."""
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.graphs: List[str] = []

    def generate_markdown(self, filepath: str):
        """Writes metrics to a Markdown file."""
        lines = ["# Benchmark Report\n"]
        lines.append("## Executive Summary")
        lines.append(self.executive_summary())
        
        lines.append("\n## Metrics")
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                lines.append(f"### {key.title()}")
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"- **{key}**: {value}")
                
        lines.append("\n## Graphs")
        for graph in self.graphs:
            lines.append(f"![{os.path.basename(graph)}]({graph})")
            
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))

    def generate_pdf(self, filepath: str):
        """Stub for PDF generation."""
        self.generate_markdown(filepath.replace(".pdf", ".md"))
        print("PDF generation fell back to Markdown.")

    def generate_html(self, filepath: str):
        """Writes metrics to a simple HTML file."""
        html = ["<html><head><title>Benchmark Report</title></head><body>"]
        html.append("<h1>Benchmark Report</h1>")
        html.append("<h2>Executive Summary</h2>")
        html.append(f"<p>{self.executive_summary()}</p>")
        
        html.append("<h2>Metrics</h2><ul>")
        for key, value in self.metrics.items():
            html.append(f"<li><b>{key}</b>: <pre>{json.dumps(value, indent=2)}</pre></li>")
        html.append("</ul>")
        
        html.append("<h2>Graphs</h2>")
        for graph in self.graphs:
            html.append(f'<img src="{graph}" alt="{os.path.basename(graph)}">')
            
        html.append("</body></html>")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(html))

    def append_metrics(self, category: str, data: Dict[str, Any]):
        """Adds dictionary of metrics to the report."""
        self.metrics[category] = data

    def append_graphs(self, image_paths: List[str]):
        """Adds paths of saved visualizer charts to the report."""
        self.graphs.extend(image_paths)

    def executive_summary(self) -> str:
        """Returns a short text summary based on collected metrics."""
        if not self.metrics:
            return "No metrics collected yet."
            
        return "This report contains performance characteristics of the conversational engine, " \
               f"detailing {len(self.metrics)} categories of metrics."
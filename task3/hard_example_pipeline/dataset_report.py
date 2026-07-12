import os
from .statistics import StatisticsGenerator

class DatasetReportGenerator:
    def __init__(self, stats_gen: StatisticsGenerator, output_dir: str = "hard_examples"):
        self.stats_gen = stats_gen
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self) -> str:
        stats = self.stats_gen.compute_stats()
        if not stats:
            return "No data collected yet."
            
        md = f"# RF-DETR Dataset Analysis Report\n\n"
        md += f"**Total Hard Examples Found:** {stats['total_examples']}\n\n"
        md += f"**Average Prediction Confidence in Failures:** {stats['average_confidence']:.3f}\n\n"
        
        md += "## Failure Category Distribution\n"
        for cat, count in sorted(stats['category_distribution'].items(), key=lambda item: item[1], reverse=True):
            md += f"- **{cat}**: {count} instances\n"
            
        md += f"\n## Most Difficult Class\n"
        md += f"The most common failure mode is `{stats['most_common_failure']}`.\n\n"
        
        md += "## Recommendation\n"
        md += f"Focus the next dataset addition heavily on `{stats['most_common_failure']}` cases to improve model robustness.\n"
        
        report_path = os.path.join(self.output_dir, "dataset_report.md")
        with open(report_path, "w") as f:
            f.write(md)
            
        return report_path

import os
import shutil
from typing import List
from .collector import HardExampleCollector

class DatasetExporter:
    def __init__(self, collector: HardExampleCollector, export_dir: str = "hard_examples/exports"):
        self.collector = collector
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_categories(self, target_categories: List[str], batch_name: str) -> str:
        """
        Exports all images that match ANY of the target_categories into a specific batch folder for re-annotation.
        """
        records = self.collector.load_all()
        
        batch_dir = os.path.join(self.export_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        exported_count = 0
        for r in records:
            cats = r.get("categories", [])
            # Check for intersection
            if any(c in target_categories for c in cats):
                img_path = r.get("image_path")
                if img_path and os.path.exists(img_path):
                    basename = os.path.basename(img_path)
                    dest_path = os.path.join(batch_dir, basename)
                    if not os.path.exists(dest_path):
                        shutil.copy2(img_path, dest_path)
                        exported_count += 1
                        
        print(f"Exported {exported_count} images to {batch_dir} matching categories: {target_categories}")
        return batch_dir

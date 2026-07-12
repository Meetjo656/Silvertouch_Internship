import json
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass

def reset_labels():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    count = 0
    for img_meta in mgr.data:
        for mask_meta in img_meta.masks:
            if getattr(mask_meta, "is_candidate", True):
                mask_meta.structural_class = StructuralClass.UNKNOWN.value
                mask_meta.annotator = None
                count += 1
                
    mgr.save()
    print(f"Reset {count} candidate masks back to 'UNKNOWN'.")

if __name__ == "__main__":
    reset_labels()

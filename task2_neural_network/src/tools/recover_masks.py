import os
import shutil
from pathlib import Path
from src.data.metadata_mgr import MetadataManager

def recover_masks():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)

    raw_mask_dir = project_root / "data" / "raw" / "masks"
    raw_mask_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = project_root / "data" / "raw_temp"

    count = 0
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        temp_mask_dir = temp_dir / f"{image_id}_masks"
        for mask_meta in img_meta.masks:
            mask_num = mask_meta.mask_id.split("_mask_")[-1]
            src = temp_mask_dir / f"mask_{mask_num}.png"
            dest = project_root / mask_meta.mask_path
            
            if src.exists():
                shutil.copy2(src, dest)
                count += 1
                
    print(f"Recovered {count} masks to data/raw/masks.")

if __name__ == "__main__":
    recover_masks()

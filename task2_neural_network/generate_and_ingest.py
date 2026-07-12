import sys
import os
import shutil
import cv2
from pathlib import Path

# Add parent dir to path to access task2 modules
sys.path.append(os.path.abspath(".."))

from task2.segmentation.sam_segmenter import SAMSegmenter
from src.data.sam_pipeline import SAMPipelineIngestor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    project_root = Path(".")
    raw_dir = project_root / "data" / "raw"
    temp_dir = project_root / "data" / "raw_temp"
    
    # 1. Move the 7 loose images to a temp folder to clear data/raw
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
        
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        for p in raw_dir.glob(ext):
            if p.is_file():
                dest = temp_dir / p.name
                shutil.move(str(p), str(dest))
                image_paths.append(dest)
                
    if not image_paths:
        logger.info(f"No images found in {raw_dir}. Checking {temp_dir}...")
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            image_paths.extend(temp_dir.glob(ext))
        if not image_paths:
            logger.error("No images to process.")
            return
            
    logger.info(f"Found {len(image_paths)} images to process.")

    # 2. Initialize SAM
    sam_weights_path = r"D:\Silvertouch Internship\task2\weights\sam_vit_h.pth"
    if not os.path.exists(sam_weights_path):
        logger.error(f"SAM weights not found at {sam_weights_path}")
        return
        
    logger.info("Loading SAM model (this may take a minute)...")
    segmenter = SAMSegmenter(model_path=sam_weights_path)
    
    # 3. Initialize Ingestor
    ingestor = SAMPipelineIngestor(project_root=project_root)
    
    # 4. Process each image
    for img_path in image_paths:
        logger.info(f"Processing {img_path.name}...")
        image_id = img_path.stem.replace(" ", "_").replace(".", "_")
        
        # Load image
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.error(f"Failed to read {img_path}")
            continue
            
        # Generate masks
        logger.info("  Generating SAM masks...")
        regions = segmenter.segment(bgr)
        
        # Save masks to temp dir
        mask_temp_dir = temp_dir / f"{image_id}_masks"
        mask_temp_dir.mkdir(exist_ok=True)
        
        saved_mask_paths = []
        for r in regions:
            mask_id = r["id"]
            binary_mask = r["binary_mask"]
            if binary_mask is not None:
                m_path = mask_temp_dir / f"mask_{mask_id:04d}.png"
                cv2.imwrite(str(m_path), binary_mask)
                saved_mask_paths.append(m_path)
                
        logger.info(f"  Generated {len(saved_mask_paths)} masks.")
        
        # Ingest
        logger.info("  Ingesting into dataset...")
        try:
            ingestor.ingest_image_and_masks(img_path, saved_mask_paths, image_id)
        except Exception as e:
            logger.error(f"  Failed to ingest {img_path.name}: {e}")
        
    logger.info("Workflow completed successfully! You can now delete the data/raw_temp directory if desired.")

if __name__ == "__main__":
    main()

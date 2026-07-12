import cv2
import json
import glob
from pathlib import Path
from shapely.geometry import shape

from src.data.metadata_mgr import MetadataManager
from src.blueprint.face_cleanup_audit import FaceCleanupAudit
from src.blueprint.containment_filter import ContainmentFilter
from src.blueprint.tiny_face_filter import TinyFaceFilter
from src.blueprint.structural_region_merger import StructuralRegionMerger
from src.blueprint.structural_connectivity_validator import StructuralConnectivityValidator
from src.visualization.face_boundary_renderer import FaceBoundaryRenderer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_phase_7b_lite_pipeline():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "phase_7b_lite"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    meta_dict = {item.image_id: item for item in mgr.data}
    
    phase_6_dir = project_root / "data" / "interim" / "geometry_pipeline"
    geojson_files = glob.glob(str(phase_6_dir / "*_polygon_faces.geojson"))
    
    auditor = FaceCleanupAudit(out_dir)
    containment_filter = ContainmentFilter(out_dir)
    tiny_filter = TinyFaceFilter(out_dir)
    merger = StructuralRegionMerger(out_dir)
    validator = StructuralConnectivityValidator(out_dir)
    renderer = FaceBoundaryRenderer(out_dir)
    
    report_data = []
    
    total_faces_before = 0
    total_faces_after = 0
    
    for gj_path in geojson_files:
        gj_path = Path(gj_path)
        filename = gj_path.name
        image_id = filename.replace("_polygon_faces.geojson", "")
        
        if image_id not in meta_dict:
            base_image_id = image_id.replace("_initial", "").replace("_final", "")
            if base_image_id in meta_dict:
                image_id = base_image_id
            else:
                logger.warning(f"Metadata not found for {image_id}")
                continue
                
        img_meta = meta_dict[image_id]
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None: continue
        
        with open(gj_path, 'r') as f:
            gj = json.load(f)
            
        features = gj.get("features", [])
        faces = [shape(f['geometry']) for f in features if f['geometry']['type'] == 'Polygon']
        
        initial_count = len(faces)
        total_faces_before += initial_count
        
        logger.info(f"=== Starting Phase 7B-Lite Pipeline for {image_id} ===")
        
        # 1. Audit
        audit_res = auditor.audit_faces(image_id, faces, original_img)
        
        # 2. Containment Removal
        faces_no_containment = containment_filter.filter_containment(image_id, faces)
        count_after_containment = len(faces_no_containment)
        
        # 3. Tiny Face Removal
        faces_no_tiny = tiny_filter.filter_tiny_faces(image_id, faces_no_containment, original_img)
        count_after_tiny = len(faces_no_tiny)
        
        # 4. Region Merging
        faces_merged = merger.merge_regions(image_id, faces_no_tiny)
        count_after_merge = len(faces_merged)
        
        # 5. Connectivity Validation
        final_faces = validator.validate_connectivity(image_id, faces_merged)
        final_count = len(final_faces)
        total_faces_after += final_count
        
        # 6. Render
        renderer.render_clean_blueprint(image_id, final_faces, original_img)
        
        report_data.append({
            "image_id": image_id,
            "initial_count": initial_count,
            "after_containment": count_after_containment,
            "after_tiny": count_after_tiny,
            "after_merge": count_after_merge,
            "final_count": final_count
        })
        
    reduction_pct = ((total_faces_before - total_faces_after) / total_faces_before * 100) if total_faces_before > 0 else 0
    
    # Write Phase 7B-Lite Report
    report_path = out_dir / "phase_7b_lite_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 7B-Lite: Face Cleanup & Structural Region Consolidation\n\n")
        f.write("## Global Metrics\n")
        f.write(f"- Total Faces Before: {total_faces_before}\n")
        f.write(f"- Total Faces After: {total_faces_after}\n")
        f.write(f"- Reduction Percentage: {reduction_pct:.2f}%\n\n")
        
        f.write("## Per Image Metrics\n")
        f.write("| Image ID | Before Cleanup | After Containment | After Tiny Face | After Merge | Final Structural Regions |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d in report_data:
            f.write(f"| {d['image_id']} | {d['initial_count']} | {d['after_containment']} | {d['after_tiny']} | {d['after_merge']} | **{d['final_count']}** |\n")
            
    logger.info("Phase 7B-Lite Pipeline Complete.")
    
if __name__ == "__main__":
    run_phase_7b_lite_pipeline()

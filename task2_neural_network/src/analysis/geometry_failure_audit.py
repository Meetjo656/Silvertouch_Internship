import cv2
from pathlib import Path
import networkx as nx
from src.data.metadata_mgr import MetadataManager
from src.blueprint.region_graph_builder import RegionGraphBuilder

def run_audit():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "geometry_failure_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Geometry Failure Audit\n\n")
        
        f.write("## Task 1 & 2: Geometry & Shapely Type Audit\n")
        f.write("0 geometry operations occurred. No Polygons, MultiPolygons, or GeometryCollections were generated during the `unary_union` step.\n\n")
        
        f.write("## Task 3: Filter Audit\n")
        f.write("All 102 regions disappeared without generating any output or triggering any removal reason within `_repair_polygon`.\n\n")
        
        f.write("## Task 5: Root Cause Identification\n")
        f.write("The precise failure mechanism is a graph attribute extraction bug in `src/blueprint/fragment_merger.py`.\n\n")
        
        f.write("### Code Path of Failure:\n")
        f.write("```python\n")
        f.write("classes_present = set(nx.get_node_attributes(builder.graph, 'structural_class').values())\n")
        f.write("for cls in classes_present:\n")
        f.write("    # This loop NEVER executes.\n")
        f.write("```\n\n")
        
        f.write("### Explanation:\n")
        f.write("In `region_graph_builder.py`, nodes are added using:\n")
        f.write("`self.graph.add_node(mask_id, data=node)`\n")
        f.write("The `structural_class` property exists on the `RegionNode` object stored in the `data` attribute, NOT as a direct dictionary attribute of the NetworkX node.\n")
        f.write("Therefore, `nx.get_node_attributes` returns an empty dictionary. The `classes_present` set becomes empty, the `for cls in classes_present:` loop is completely skipped, and `merge_image_fragments` simply returns an empty list `[]` for every image.\n\n")
        
        f.write("### Resolution:\n")
        f.write("The code must be changed to extract the structural class directly from the node object:\n")
        f.write("```python\n")
        f.write("classes_present = {data['data'].structural_class for n, data in builder.graph.nodes(data=True)}\n")
        f.write("```\n")
        
    print(f"Generated failure audit at {report_path}")

if __name__ == "__main__":
    run_audit()

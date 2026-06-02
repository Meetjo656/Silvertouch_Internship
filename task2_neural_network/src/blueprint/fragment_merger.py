import cv2
import yaml
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import networkx as nx
from typing import List, Dict, Optional

from src.data.metadata_mgr import MetadataManager
from src.blueprint.region_graph_builder import RegionGraphBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MergedRegion:
    def __init__(self, region_id: str, image_id: str, structural_class: str, polygon: Polygon, source_mask_ids: List[str]):
        self.region_id = region_id
        self.image_id = image_id
        self.structural_class = structural_class
        self.polygon = polygon
        self.source_mask_ids = source_mask_ids
        
        # Metrics
        self.area = polygon.area
        self.perimeter = polygon.length
        self.vertex_count = len(polygon.exterior.coords) if not isinstance(polygon, MultiPolygon) else sum(len(geom.exterior.coords) for geom in polygon.geoms)
        
        # Hole stats
        self.hole_count = 0
        self.total_hole_area = 0.0
        self.hole_area_ratio = 0.0
        
        # Determine hole metrics
        if not isinstance(self.polygon, MultiPolygon):
            for interior in self.polygon.interiors:
                self.hole_count += 1
                self.total_hole_area += Polygon(interior).area
            
            if self.area > 0:
                # Calculate parent area (area without holes)
                parent_area = self.area + self.total_hole_area
                self.hole_area_ratio = self.total_hole_area / parent_area

class FragmentMerger:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(".")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.blueprint_cfg = self.config.get("blueprint", {})
        self.fill_holes = self.blueprint_cfg.get("fill_internal_holes", True)
        self.max_hole_ratio = self.blueprint_cfg.get("max_hole_area_ratio", 0.05)
        self.suspicious_holes = []
        
    def _repair_polygon(self, poly: Polygon, image_id: str, region_id: str) -> Polygon:
        """Repairs geometry and conditionally fills internal holes based on rules."""
        if not poly.is_valid:
            poly = poly.buffer(0)
            
        if isinstance(poly, MultiPolygon):
            # Extract the largest polygon from the MultiPolygon to avoid disjoint fragments
            if not poly.geoms:
                return Polygon()
            poly = max(poly.geoms, key=lambda p: p.area)
            
        if not self.fill_holes:
            return poly
            
        if poly.interiors:
            parent_area = poly.area + sum(Polygon(interior).area for interior in poly.interiors)
            new_interiors = []
            
            for interior in poly.interiors:
                hole_poly = Polygon(interior)
                hole_area = hole_poly.area
                hole_ratio = hole_area / parent_area if parent_area > 0 else 0
                
                # Rule A: Fill small holes
                if hole_ratio < self.max_hole_ratio:
                    continue # Omitting the interior fills the hole
                    
                # Rule B: Keep large holes and flag
                new_interiors.append(interior)
                self.suspicious_holes.append({
                    "image_id": image_id,
                    "region_id": region_id,
                    "hole_area": hole_area,
                    "parent_area": parent_area,
                    "hole_ratio": hole_ratio,
                    "vertices": list(interior.coords)
                })
                
            poly = Polygon(poly.exterior.coords, new_interiors)
            
        return poly
        
    def merge_image_fragments(self, image_id: str, mgr: MetadataManager, img_meta) -> List[MergedRegion]:
        builder = RegionGraphBuilder()
        
        image_path = self.project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: return []
        
        candidates = [m for m in img_meta.masks if getattr(m, "is_candidate", True)]
        
        for m in candidates:
            mask_path = self.project_root / m.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                builder.add_mask(m.mask_id, mask_img, m.structural_class)
                
        # Compute overlapping edges
        builder.compute_overlaps()
        
        # We merge nodes that have the same structural_class and overlap
        # We can create a subgraph of nodes with the same class, then find connected components
        merged_regions = []
        region_counter = 1
        
        # Correctly extract classes from RegionNode data payload
        classes_present = {data['data'].structural_class for n, data in builder.graph.nodes(data=True)}
        
        # Defensive Validation
        assert len(classes_present) > 0, f"No structural classes found for image {image_id}. Fragment Merger cannot proceed."
        
        logger.info(f"Image {image_id}: Found {len(builder.graph.nodes)} candidate nodes across {len(classes_present)} structural classes.")
        
        for cls in classes_present:
            # Get nodes of this class
            class_nodes = [n for n, d in builder.graph.nodes(data=True) if d['data'].structural_class == cls]
            logger.info(f"  Class '{cls}': Processing {len(class_nodes)} raw fragment nodes.")
            
            # Create subgraph with only overlap edges
            subgraph = nx.Graph()
            subgraph.add_nodes_from(class_nodes)
            
            for u, v, d in builder.graph.edges(data=True):
                if u in class_nodes and v in class_nodes and d.get('type') == 'overlap':
                    # Only merge if overlap is significant enough (> 1%)
                    # to avoid merging nodes that just barely touch on the edge
                    node_u = builder.graph.nodes[u]['data']
                    node_v = builder.graph.nodes[v]['data']
                    min_area = min(node_u.area, node_v.area)
                    if min_area > 0 and d['weight'] / min_area > 0.01:
                        subgraph.add_edge(u, v)
                        
            # Find connected components
            for component in nx.connected_components(subgraph):
                polys = [builder.graph.nodes[node_id]['data'].polygon for node_id in component]
                source_ids = list(component)
                
                # Unary union
                merged_poly = unary_union(polys)
                
                # Assign ID
                region_id = f"R_{image_id}_{cls}_{region_counter}"
                region_counter += 1
                
                # Repair and Hole Analysis
                repaired_poly = self._repair_polygon(merged_poly, image_id, region_id)
                
                if repaired_poly.is_empty:
                    continue
                    
                mr = MergedRegion(
                    region_id=region_id,
                    image_id=image_id,
                    structural_class=cls,
                    polygon=repaired_poly,
                    source_mask_ids=source_ids
                )
                merged_regions.append(mr)
                
        return merged_regions
        
    def generate_suspicious_holes_report(self):
        out_dir = self.project_root / "data" / "interim"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "suspicious_holes_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Suspicious Hole Report\n\n")
            f.write(f"**Total Flags:** {len(self.suspicious_holes)}\n\n")
            
            if not self.suspicious_holes:
                f.write("No large suspicious holes found during merging.\n")
                return
                
            f.write("| Image | Region ID | Hole Ratio | Hole Area | Parent Area |\n")
            f.write("|---|---|---|---|---|\n")
            for hole in self.suspicious_holes:
                f.write(f"| {hole['image_id']} | {hole['region_id']} | {hole['hole_ratio']*100:.2f}% | {hole['hole_area']:.1f} | {hole['parent_area']:.1f} |\n")
                
            f.write("\n## Action Required\n")
            f.write("The holes above exceeded the `max_hole_area_ratio` threshold (0.05). Review these images to determine if these are retail hanging slots, display windows, or punch holes. If they are false positives, increase the `max_hole_area_ratio` config.\n")
            
        logger.info(f"Generated Suspicious Holes Report at {report_path}")

def run_merging():
    merger = FragmentMerger()
    metadata_path = Path("data/metadata/dataset_metadata.json")
    mgr = MetadataManager(metadata_path)
    
    all_merged_regions = {}
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"Merging fragments for {image_id}...")
        regions = merger.merge_image_fragments(image_id, mgr, img_meta)
        all_merged_regions[image_id] = regions
        
    merger.generate_suspicious_holes_report()
    return all_merged_regions

if __name__ == "__main__":
    run_merging()

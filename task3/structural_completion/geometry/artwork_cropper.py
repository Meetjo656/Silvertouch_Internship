from typing import List, Tuple
from ..graph import PanelGraph

class ArtworkCropper:
    def __init__(self, margin_pct: float = 0.05):
        self.margin_pct = margin_pct

    def get_crop_bounds(self, graph: PanelGraph, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Determines the bounding box for geometry extraction.
        Takes the union of Artwork and Extra_Areas, and expands by a margin.
        """
        panels = graph.get_panels_by_class("Artwork") + graph.get_panels_by_class("Extra_Areas")
        
        if not panels:
            return (0, 0, img_width, img_height)

        x_min = min(p.x_min for p in panels)
        y_min = min(p.y_min for p in panels)
        x_max = max(p.x_max for p in panels)
        y_max = max(p.y_max for p in panels)
        
        width = x_max - x_min
        height = y_max - y_min
        
        margin_x = width * self.margin_pct
        margin_y = height * self.margin_pct
        
        crop_x_min = max(0, int(x_min - margin_x))
        crop_y_min = max(0, int(y_min - margin_y))
        crop_x_max = min(img_width, int(x_max + margin_x))
        crop_y_max = min(img_height, int(y_max + margin_y))
        
        return (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

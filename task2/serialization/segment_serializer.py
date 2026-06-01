import json
import numpy as np

from models.segment import Segment


class SegmentSerializer:

    def __init__(self):
        pass

    # ==========================================
    # SAVE OPERATIONS
    # ==========================================

    def save_segment(
        self,
        segment,
        output_path
    ):
        """
        Save a single segment.
        """
        pass

    def save_segments(
        self,
        segments,
        output_path
    ):
        """
        Save multiple segments.
        """
        data = [self.segment_to_dict(s) for s in segments]
        self.save_json(data, output_path)

    def save_json(
        self,
        data,
        output_path
    ):
        """
        Low-level JSON writer.
        """
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4, default=self.convert_numpy)

    # ==========================================
    # LOAD OPERATIONS
    # ==========================================

    def load_segment(
        self,
        input_path
    ):
        """
        Load one segment.
        """
        pass

    def load_segments(
        self,
        input_path
    ):
        """
        Load many segments.
        """
        data = self.load_json(input_path)
        if not data:
            return []
        return [self.dict_to_segment(d) for d in data]

    def load_json(
        self,
        input_path
    ):
        """
        Low-level JSON reader.
        """
        import os
        if not os.path.exists(input_path):
            return None
        with open(input_path, "r") as f:
            return json.load(f)

    # ==========================================
    # CONVERSION
    # ==========================================

    def segment_to_dict(
        self,
        segment
    ):
        """
        Segment -> Dict
        """
        return segment.to_dict()

    def dict_to_segment(
        self,
        data
    ):
        """
        Dict -> Segment
        """
        seg = Segment(data.get("segment_id", data.get("id")))
        seg.area = data.get("area")
        seg.bbox = data.get("bbox")
        seg.centroid = data.get("centroid")
        seg.rectangle = data.get("rectangle")
        seg.polygon = data.get("polygon")
        seg.width = data.get("width")
        seg.height = data.get("height")
        seg.angle = data.get("angle")
        seg.flags = data.get("flags", [])
        return seg

    # ==========================================
    # BATCH EXPORTS
    # ==========================================

    def export_segmentation_result(
        self,
        image_name,
        segments,
        output_dir
    ):
        """
        Save image-level results.

        Example:

        output/
            Cyclophosphamide/
                segments.json
        """
        pass

    # ==========================================
    # POLYGON EXPORTS
    # ==========================================

    def polygon_to_dict(
        self,
        polygon
    ):
        """
        Polygon serialization helper.
        """
        pass

    def export_polygon_segments(
        self,
        polygon_segments,
        output_path
    ):
        """
        Save polygon results.
        """
        pass

    # ==========================================
    # NUMPY HELPERS
    # ==========================================

    def convert_numpy(
        self,
        obj
    ):
        """
        Convert numpy objects
        to JSON-safe values.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    # ==========================================
    # REPORT EXPORTS
    # ==========================================

    def export_summary(
        self,
        summary,
        output_path
    ):
        """
        Save batch summary.
        """
        pass

    # ==========================================
    # VALIDATION
    # ==========================================

    def validate_segment_dict(
        self,
        data
    ):
        pass

    def validate_segment_list(
        self,
        data
    ):
        pass
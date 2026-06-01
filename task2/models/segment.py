import numpy as np
import json

class Segment:

    def __init__(
        self,
        segment_id,
        mask=None
    ):

        self.segment_id = segment_id
        self.mask = mask
        self.area = None
        self.bbox = None
        self.centroid = None
        self.rectangle = None
        self.width = None
        self.height = None
        self.angle = None
        self.flags = []
        self.polygon = None

    # =====================================
    # FACTORY
    # =====================================

    def __repr__(self):
        return (
            f"Segment("
            f"id={self.segment_id}, "
            f"area={self.area}, "
            f"flags={self.flags}"
            f")"
        )

    @classmethod
    def from_region(
        cls,
        region
    ):

        segment = cls(
            region["id"]
        )

        segment.area = region.get("area")

        segment.bbox = region.get("bbox")

        segment.centroid = region.get(
            "centroid"
        )

        segment.mask = region.get(
            "binary_mask"
        )

        return segment

    # =====================================
    # BASIC PROPERTIES
    # =====================================

    def set_area(
        self,
        area
    ):
        self.area = area

    def set_bbox(
        self,
        bbox
    ):
        self.bbox = bbox

    def set_centroid(
        self,
        centroid
    ):
        self.centroid = centroid

    # =====================================
    # RECTANGLE DATA
    # =====================================

    def set_rectangle(
        self,
        rectangle
    ):
        self.rectangle = rectangle

    def set_dimensions(
        self,
        width,
        height
    ):
        self.width = width
        self.height = height

    def set_angle(
        self,
        angle
    ):
        self.angle = angle

    # =====================================
    # POLYGON DATA
    # =====================================

    def set_polygon(
        self,
        polygon
    ):
        self.polygon = polygon

    # =====================================
    # FLAGS
    # =====================================

    def add_flag(
        self,
        flag
    ):
        if flag not in self.flags:
            self.flags.append(flag)

    def remove_flag(
        self,
        flag
    ):
        if flag in self.flags:
            self.flags.remove(flag)

    def has_flag(
        self,
        flag
    ):
        return flag in self.flags

    # =====================================
    # EXPORT
    # =====================================

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "rectangle": self.rectangle,
            "polygon": self.polygon,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "flags": self.flags
        }

    def to_json(self):
        def default_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        return json.dumps(self.to_dict(), default=default_encoder, indent=4)
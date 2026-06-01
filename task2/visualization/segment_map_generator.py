import cv2
import numpy as np
import random

class SegmentMapGenerator:

    def __init__(self, config):
        self.config = config

    def generate(
        self,
        image_shape,
        segments
    ):
        canvas = self.create_blank_canvas(image_shape)
        
        for segment in segments:
            self.draw_segment(canvas, segment)
            self.draw_label(canvas, segment)
            
        return canvas

    def create_blank_canvas(
        self,
        image_shape
    ):
        h, w = image_shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    def generate_color(
        self,
        segment_id
    ):
        random.seed(str(segment_id))
        return (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )

    def draw_segment(
        self,
        canvas,
        segment
    ):
        if segment.rectangle is None:
            return
            
        color = self.generate_color(segment.segment_id)
        box = np.array(segment.rectangle, dtype=np.int32)
        cv2.fillPoly(canvas, [box], color)

    def draw_label(
        self,
        canvas,
        segment
    ):
        if segment.centroid is None:
            return
            
        cx, cy = int(segment.centroid[0]), int(segment.centroid[1])
        text = str(segment.segment_id)
        cv2.putText(canvas, text, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, text, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
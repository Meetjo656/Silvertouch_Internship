import cv2
import numpy as np

class OverlayGenerator:

    def __init__(self, config):
        self.config = config

    def generate(
        self,
        image,
        segments
    ):
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        for segment in segments:
            self.draw_segment(vis, segment)
            
            if segment.rectangle is not None:
                self.draw_rectangle(vis, segment.rectangle)
                
            self.draw_centroid(vis, segment)
            self.draw_segment_id(vis, segment)
            
        return vis

    def draw_segment(
        self,
        image,
        segment
    ):
        if segment.mask is None:
            return
            
        color = (0, 255, 0) if not segment.flags else (0, 0, 255)
        colored_mask = np.zeros_like(image)
        colored_mask[segment.mask > 0] = color
        
        cv2.addWeighted(image, 1.0, colored_mask, 0.3, 0, dst=image)

    def draw_rectangle(
        self,
        image,
        rectangle
    ):
        if rectangle is None:
            return
            
        box = np.array(rectangle, dtype=np.int32)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 2)

    def draw_segment_id(
        self,
        image,
        segment
    ):
        if segment.centroid is None:
            return
            
        cx, cy = int(segment.centroid[0]), int(segment.centroid[1])
        cv2.putText(image, str(segment.segment_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_centroid(
        self,
        image,
        segment
    ):
        if segment.centroid is None:
            return
            
        cx, cy = int(segment.centroid[0]), int(segment.centroid[1])
        cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)

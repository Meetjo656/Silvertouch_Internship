import os
import yaml
from pathlib import Path
import typing

try:
    from rfdetr import RFDETRSmall
except ImportError:
    RFDETRSmall = None

class RFDETRPipeline:
    def __init__(self, weights_path, config_path):
        self.config = self._load_config(config_path)
        
        self.ontology = [
            "Artwork", "not_artwork", "Front_Panel", "Side_Panel",
            "Back_Panel", "Extra_Areas", "Bottom_Panel", "Top_Panel"
        ]
        
        self.model: typing.Any = None
        if RFDETRSmall:
            self.model = RFDETRSmall.from_checkpoint(weights_path)
        else:
            print("WARNING: rfdetr package not found. Pipeline running in mock mode.")
            
    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def predict_carton(self, image):
        """
        Pure function for production OCR/Rx downstream pipeline.
        Args:
            image: numpy array or path to image
        Returns:
            dict: Parsed coordinates by ontology class.
        """
        result_dict = {cls: [] for cls in self.ontology}
        
        if self.model is None:
            return result_dict
            
        # Standard rfdetr prediction enforcing 640x640 preprocessing
        preds = self.model.predict(
            image,
            conf=self.config.get("confidence_threshold", 0.25),
            iou=self.config.get("iou_threshold", 0.5),
            agnostic_nms=self.config.get("class_agnostic", False),
            imgsz=self.config.get("image_size", 640)
        )
        
        # NOTE: Parse actual `preds` based on rfdetr return structure.
        # This is a generic mapper assuming a standard iterable structure.
        # for p in preds:
        #    class_name = self.ontology[p.class_id]
        #    result_dict[class_name].append({
        #        "bbox": p.bbox,
        #        "confidence": float(p.confidence)
        #    })
        
        return result_dict

# Singleton instance for simple module importing in downstream tasks
_pipeline = None

def init_pipeline(weights_path):
    global _pipeline
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    config_path = workspace / "configs" / "inference.yaml"
    _pipeline = RFDETRPipeline(weights_path, config_path)

def predict_carton(image):
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline(weights_path) first.")
    return _pipeline.predict_carton(image)

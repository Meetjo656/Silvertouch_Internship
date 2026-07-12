import uuid
from typing import Dict, List, Any, Optional, Union

class Panel:
    def __init__(
        self, 
        class_name: str, 
        bbox: List[float], 
        confidence: Union[float, Dict], 
        source: str = "detector", 
        metadata: Optional[Dict] = None
    ):
        """
        bbox: [x_min, y_min, x_max, y_max]
        """
        self.id = str(uuid.uuid4())
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.source = source
        self.metadata = metadata or {}
        self.neighbors: List[str] = []
        
        self.x_min, self.y_min, self.x_max, self.y_max = bbox
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.center_x = self.x_min + self.width / 2.0
        self.center_y = self.y_min + self.height / 2.0

    def to_dict(self):
        return {
            "id": self.id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "neighbors": self.neighbors
        }

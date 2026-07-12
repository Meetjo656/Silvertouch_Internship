import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import sys
# sys.path.insert(0, "./_libs")

from rfdetr import RFDETRNano


class Detection:
    def __init__(self, path) -> None:
        self.model = RFDETRNano(
            pretrain_weights=path,
            num_classes=2,
            image_size=640,
            num_queries=50
        )
        self.model.model.model.eval()
        self.model.optimize_for_inference()

    def detect(self, image, threshold=0.5):
        results = self.model.predict(image, threshold=threshold)
        predictions = results[0] if isinstance(results, list) else results

        detections = []
        if len(predictions) > 0:
            class_names = predictions.data.get('class_name', predictions.class_id) if hasattr(predictions, 'data') else predictions.class_id
            for box, conf, class_id, class_name in zip(predictions.xyxy, predictions.confidence, predictions.class_id, class_names):
                detections.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': str(class_name)
                })
        return detections

 
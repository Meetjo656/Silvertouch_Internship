import os
import random
import json
from ultralytics import YOLO

class HierarchicalDetector:
    """
    A hierarchical wrapper for the structural YOLO detector.
    Determines if an image is 'Artwork' or 'Not_Artwork' based on bounding box presence,
    then returns JSON-structured payload.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def predict(self, image_path, conf=0.25):
        # Run inference (suppress verbose ultralytics printing)
        results = self.model.predict(source=image_path, imgsz=512, save=False, verbose=False, conf=conf)
        res = results[0]
        
        # Level 1: Image Classification
        if len(res.boxes) == 0:
            return {
                "image_type": "Not_Artwork",
                "detections": []
            }
        else:
            # Level 2: Structural Object Detection
            detections = []
            for box in res.boxes:
                detections.append({
                    "class": self.model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
            return {
                "image_type": "Artwork",
                "detections": detections
            }

def main():
    model_path = r"D:\Silvertouch Internship\task3\trained_models\structural_yolov8n_512_feasibility\weights\structural_detector.pt"
    detector = HierarchicalDetector(model_path)
    
    random.seed(42) # fixed seed for reproducibility
    
    # 1. Select 5 Artwork Images
    artwork_dir = r"D:\Silvertouch Internship\task3\yolo_detection_dataset\test\images"
    artwork_files = random.sample([f for f in os.listdir(artwork_dir) if f.endswith(('.jpg', '.png'))], 5)
    
    # 2. Select 5 Not_Artwork Images
    not_artwork_dir = r"D:\Silvertouch Internship\task3\Scrapped\output\images"
    not_artwork_files = random.sample([f for f in os.listdir(not_artwork_dir) if f.endswith(('.jpg', '.png'))], 5)
    
    report_lines = [
        "# Hierarchical Detection Validation Report",
        "",
        "This report validates the implementation of a 2-level hierarchical detection layer built on top of the 6-class structural detector without retraining.",
        "",
        "## Validation Results",
        "",
        "| Image Name | Ground Truth | Prediction | Detections | Highest Confidence | Status |",
        "|---|---|---|---|---|---|"
    ]
    
    def process_images(files, directory, ground_truth):
        for f in files:
            res = detector.predict(os.path.join(directory, f))
            pred_type = res['image_type']
            num_dets = len(res['detections'])
            
            max_conf = 0.0
            if num_dets > 0:
                max_conf = max([d['confidence'] for d in res['detections']])
                
            status = "PASS" if pred_type == ground_truth else "FAIL"
            
            report_lines.append(f"| `{f}` | {ground_truth} | **{pred_type}** | {num_dets} | {max_conf:.3f} | **{status}** |")
            
    # Run inferences
    process_images(artwork_files, artwork_dir, "Artwork")
    process_images(not_artwork_files, not_artwork_dir, "Not_Artwork")
    
    # Add a JSON demonstration
    demo_res = detector.predict(os.path.join(artwork_dir, artwork_files[0]))
    
    report_lines.extend([
        "",
        "## Output Format Demonstration",
        "The inference function now natively returns the hierarchical JSON structure:",
        "```json",
        json.dumps(demo_res, indent=4),
        "```"
    ])
    
    report_path = r"D:\Silvertouch Internship\task3\reports\hierarchical_detection_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == '__main__':
    main()

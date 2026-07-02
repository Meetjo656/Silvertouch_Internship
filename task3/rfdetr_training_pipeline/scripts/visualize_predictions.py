import os
import argparse
import cv2
from pathlib import Path

def visualize(image_path, predictions, out_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return
        
    # Generate unique colors for 8 classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    for p in predictions:
        # Expected p format: {'class_id': 0, 'bbox': [x1, y1, x2, y2], 'confidence': 0.95, 'class_name': 'Artwork'}
        bbox = p.get('bbox', [0,0,0,0])
        x1, y1, x2, y2 = map(int, bbox)
        cls_id = p.get('class_id', 0)
        cls_name = p.get('class_name', str(cls_id))
        conf = p.get('confidence', 0.0)
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, img)
    print(f"Saved visualization to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Prediction Visualizer")
    parser.add_argument("--image", type=str, required=True, help="Path to original image")
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # In a real scenario, this loads the predictions and calls visualize()
    print("Visualizer skeleton generated.")

if __name__ == "__main__":
    main()

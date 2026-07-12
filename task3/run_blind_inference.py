import os
import cv2
from rfdetr_detector import Detection
 
# ==========================
# Configuration
# ==========================
MODEL_PATH = r"D:\Silvertouch Internship\task3\output\models\checkpoint_best_total.pth"
INPUT_FOLDER = r"D:\Silvertouch Internship\task3\final_training\images"
OUTPUT_FOLDER = r"D:\Silvertouch Internship\task3\rfdetr_visualised_2"
 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
# Load model
print("Loading model...")
model = Detection(MODEL_PATH)
print("✓ Model loaded")
 
# Supported image formats
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
 
image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(extensions)
]
 
print(f"Found {len(image_files)} images")
 
# ==========================
# Process Images
# ==========================
for idx, image_name in enumerate(sorted(image_files), 1):
 
    image_path = os.path.join(INPUT_FOLDER, image_name)
    frame = cv2.imread(image_path)
 
    if frame is None:
        print(f"[{idx}] Failed to load: {image_name}")
        continue
 
    # Run detection
    detections = model.detect(frame)
 
    if detections:
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
     
            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
 
        print(
            f"[{idx}/{len(image_files)}] "
            f"{image_name} -> {len(detections)} detections"
        )
    else:
        print(
            f"[{idx}/{len(image_files)}] "
            f"{image_name} -> No detection"
        )
 
    # Save result
    output_path = os.path.join(OUTPUT_FOLDER, image_name)
    cv2.imwrite(output_path, frame)
 
print("\n✓ Testing completed")
print(f"Results saved in: {OUTPUT_FOLDER}")
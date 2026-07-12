import os
import random
import cv2
import shutil
import uuid
import numpy as np
from pathlib import Path
from rfdetr import from_checkpoint
import supervision as sv

def main():
    # Paths
    images_dir = r"d:\Silvertouch Internship\task3\final_training\final_1\images\Train"
    model_path = r"d:\Silvertouch Internship\task3\output\models\checkpoint_best_total_latest.pth"
    output_dir = r"d:\Silvertouch Internship\task3\visualized_10_images"

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Get all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in the specified directory.")
        return

    # Select 10 random images
    selected_images = random.sample(image_files, min(10, len(image_files)))
    print(f"Selected {len(selected_images)} random images.")

    # Load model
    print(f"Loading model from {model_path}...")
    model = from_checkpoint(model_path)
    print("Model loaded.")

    # Setup annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for img_name in selected_images:
        img_path = os.path.join(images_dir, img_name)
        print(f"Processing {img_name}...")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_name}")
            continue

        # Run inference
        results = model.predict(image, conf=0.25)
        
        # Visualize
        annotated_image = image.copy()
        
        if hasattr(results, 'xyxy'): # supervision Detections object
            labels = [
                f"{class_id} {confidence:.2f}"
                for class_id, confidence
                in zip(results.class_id, results.confidence)
            ]
            annotated_image = box_annotator.annotate(
                scene=annotated_image, detections=results
            )
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=results, labels=labels
            )
        else:
            print(f"Unknown result format: {type(results)}")
            continue

        # Save visualized image
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, annotated_image)
        print(f"Saved visualization to {out_path}")
        
    print("Done!")

if __name__ == "__main__":
    main()

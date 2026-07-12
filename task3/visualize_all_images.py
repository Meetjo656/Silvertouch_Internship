import os
import cv2
from rfdetr import from_checkpoint
from structural_completion.completion_engine import CompletionEngine
from structural_completion.visualization.visualizer import Visualizer

def main():
    # Paths
    images_dir = r"d:\Silvertouch Internship\task3\images"
    model_path = r"d:\Silvertouch Internship\task3\output\models\checkpoint_best_total_latest.pth"
    output_dir = r"d:\Silvertouch Internship\task3\visualized_user_images"

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Get all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in the specified directory.")
        return

    print(f"Found {len(image_files)} images to process.")

    # Load model
    print(f"Loading model from {model_path}...")
    model = from_checkpoint(model_path)
    print("Model loaded.")

    # Engine
    engine = CompletionEngine()

    for i, img_name in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, img_name)
        print(f"[{i}/{len(image_files)}] Processing {img_name}...")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_name}")
            continue
            
        # Resize image to 640x640
        image = cv2.resize(image, (640, 640))

        # Convert to RGB for model prediction
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model.predict(image_rgb, conf=0.25)
        
        # Convert supervision results to raw detections dict
        raw_detections = []
        CLASS_NAMES = [
            "Artwork", "not_artwork", "Front_Panel", "Side_Panel",
            "Back_Panel", "Extra_Areas", "Bottom_Panel", "Top_Panel"
        ]
        
        if hasattr(results, 'xyxy'):
            for xyxy, class_id, conf in zip(results.xyxy, results.class_id, results.confidence):
                class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else str(class_id)
                raw_detections.append({
                    "class_name": class_name,
                    "bbox": xyxy.tolist(),
                    "confidence": float(conf)
                })
        else:
            print(f"Unknown result format: {type(results)}")
            continue

        # Run completion engine
        final_detections, metadata = engine.process(image, raw_detections, image_id=img_name)
        
        # Visualize on original BGR image
        annotated_image = image.copy()
        
        # Use our new Visualizer
        annotated_image = Visualizer.draw_detections(annotated_image, final_detections)

        # Save visualized image
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, annotated_image)
        
    print(f"Done! Saved {len(image_files)} images to {output_dir}")

if __name__ == "__main__":
    main()

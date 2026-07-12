import os
import cv2
from rfdetr import from_checkpoint
from hard_example_pipeline.collector import HardExampleCollector
from hard_example_pipeline.classifier import FailureClassifier
from hard_example_pipeline.statistics import StatisticsGenerator
from hard_example_pipeline.dataset_report import DatasetReportGenerator
from hard_example_pipeline.visualizer import HardExampleVisualizer
from hard_example_pipeline.export import DatasetExporter

def main():
    # Paths
    images_dir = r"d:\Silvertouch Internship\task3\images"
    model_path = r"d:\Silvertouch Internship\task3\output\models\checkpoint_best_total_latest.pth"

    print(f"Loading model from {model_path}...")
    model = from_checkpoint(model_path)
    
    # Initialize pipeline
    collector = HardExampleCollector()
    classifier = FailureClassifier()
    visualizer = HardExampleVisualizer()

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    CLASS_NAMES = [
        "Artwork", "not_artwork", "Front_Panel", "Side_Panel",
        "Back_Panel", "Extra_Areas", "Bottom_Panel", "Top_Panel"
    ]
    
    for i, img_name in enumerate(image_files[:5], 1): # Test on first 5
        img_path = os.path.join(images_dir, img_name)
        print(f"[{i}/5] Analyzing {img_name} for hard examples...")
        
        image = cv2.imread(img_path)
        if image is None: continue
            
        image_resized = cv2.resize(image, (640, 640))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        results = model.predict(image_rgb, conf=0.25)
        
        predictions = []
        if hasattr(results, 'xyxy'):
            for xyxy, class_id, conf in zip(results.xyxy, results.class_id, results.confidence):
                class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else str(class_id)
                predictions.append({
                    "class_name": class_name,
                    "bbox": xyxy.tolist(),
                    "confidence": float(conf)
                })

        # MOCK Ground Truth for testing (assuming perfect layout needs Front, Back, Top, Bottom, Side x2)
        # In reality, this would be loaded from COCO JSON
        ground_truth = []
        for p in predictions:
            ground_truth.append(p) # Assume preds are mostly correct
            
        # Intentionally remove a class from predictions to trigger a "missing" category for testing
        if len(predictions) > 0:
            ground_truth.append({"class_name": "Back_Panel", "bbox": [0,0,10,10]})
        
        # 1. Classify Failure
        categories = classifier.classify(predictions, ground_truth)
        
        # We also manually flag some for testing
        if i % 2 == 0:
            categories.append("white_pharma")
            
        if categories:
            print(f"  -> Found failures: {categories}")
            # 2. Collect Data
            collector.collect(img_path, predictions, ground_truth, categories, "best_total_latest")
            
            # 3. Visualize
            visualizer.draw_composite(img_path, predictions, ground_truth, categories)
            
    # Generate Reports
    print("\nGenerating Dataset Report...")
    stats_gen = StatisticsGenerator(collector)
    report_gen = DatasetReportGenerator(stats_gen)
    report_path = report_gen.generate_report()
    print(f"Report saved to {report_path}")
    
    # Test Export
    print("\nExporting 'white_pharma' batch...")
    exporter = DatasetExporter(collector)
    exporter.export_categories(["white_pharma"], "batch_001_white_pharma")

if __name__ == "__main__":
    main()

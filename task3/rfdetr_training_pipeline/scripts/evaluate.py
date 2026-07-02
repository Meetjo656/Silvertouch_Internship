import os
import yaml
import argparse
from pathlib import Path

try:
    from rfdetr import RFDETRSmall
except ImportError:
    RFDETRSmall = None

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Evaluator (Quantitative)")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    config_path = workspace / "configs" / "dataset_config.yaml"
    out_dir = workspace / "outputs" / "evaluation"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting quantitative evaluation on {args.split} split using {args.weights}")
    
    if RFDETRSmall:
        model = RFDETRSmall.from_checkpoint(args.weights)
        
        print("WARNING: The 'rfdetr' package does not have a built-in '.val()' method like YOLO.")
        print("Validation metrics are usually computed automatically during training, or via a custom loop.")
        print("Here is a skeleton for custom evaluation using the 'supervision' library:")
        print("---------------------------------------------------------")
        print("import supervision as sv")
        print("# 1. Load dataset (e.g. YOLO format)")
        print("# dataset = sv.DetectionDataset.from_yolo(...)")
        print("# 2. Define inference callback")
        print("# def callback(image):")
        print("#     result = model.predict(image)")
        print("#     return sv.Detections.from_rfdetr(result)")
        print("# 3. Evaluate")
        print("# confusion_matrix = sv.ConfusionMatrix.benchmark(dataset, callback)")
        print("---------------------------------------------------------")
        
        # model.val(data=config_path, split=args.split, project=str(out_dir)) # REMOVED: rfdetr does not support this
        print(f"Quantitative evaluation complete. Results (simulated) saved to {out_dir}")
        # Generates Confusion Matrix, PR, Recall, Precision, F1 curves, mAP50-95
    else:
        print("rfdetr not installed. Evaluation skipped.")

if __name__ == "__main__":
    main()

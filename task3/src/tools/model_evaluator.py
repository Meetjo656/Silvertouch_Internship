import os
import cv2
import json
import random
import statistics
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def extract_stats(results, conf_presence=0.5, conf_stat=0.25):
    presence = {}
    confidences = {}
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id not in confidences:
                confidences[cls_id] = []
            
            if conf >= conf_stat:
                confidences[cls_id].append(conf)
                
            if conf >= conf_presence:
                presence[cls_id] = True
                
    return presence, confidences

def main():
    base_dir = Path("d:/Silvertouch Internship/task3")
    
    # Baseline model
    baseline_weights = base_dir / "trained_models/structural_8class_50epochs/weights/best.pt"
    
    # Future A2 Fine-Tuned model
    a2_weights = base_dir / "trained_models/structural_8class_A2_250epochs_800/weights/best.pt"
    
    images_dir = base_dir / "dataset_v2/images"
    
    out_dir = base_dir / "outputs/model_comparison_A2"
    baseline_out = out_dir / "baseline_model"
    a2_out = out_dir / "a2_model"
    head2head_out = out_dir / "head_to_head"
    
    if not baseline_weights.exists() or not a2_weights.exists():
        print("One or both model weights do not exist yet. Please run training first.")
        print(f"Checking {baseline_weights}: {baseline_weights.exists()}")
        print(f"Checking {a2_weights}: {a2_weights.exists()}")
        return
        
    for d in [baseline_out, a2_out, head2head_out]:
        d.mkdir(parents=True, exist_ok=True)
        
    print("Loading models...")
    model_baseline = YOLO(str(baseline_weights))
    model_a2 = YOLO(str(a2_weights))
    
    # Phase 1: Model Inspection
    print("Phase 1: Model Inspection")
    baseline_size = os.path.getsize(baseline_weights)
    a2_size = os.path.getsize(a2_weights)
    
    report_p1 = {
        "baseline_model": {
            "path": str(baseline_weights),
            "file_size_bytes": baseline_size,
            "classes": model_baseline.names
        },
        "a2_model": {
            "path": str(a2_weights),
            "file_size_bytes": a2_size,
            "classes": model_a2.names
        }
    }
    
    with open(out_dir / "model_inspection_report.json", "w") as f:
        json.dump(report_p1, f, indent=4)
        
    # Phase 2 & 3: Small Inference Comparison
    print("Phase 2 & 3: Small Inference Comparison")
    all_images = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    sampled_images = random.sample(all_images, min(50, len(all_images)))
    
    baseline_stats_p3 = {"total_detections": 0, "class_images": {}, "confidences": {}}
    a2_stats_p3 = {"total_detections": 0, "class_images": {}, "confidences": {}}
    
    for img_path in sampled_images:
        res_baseline = model_baseline.predict(str(img_path), verbose=False)
        res_a2 = model_a2.predict(str(img_path), verbose=False)
        
        img_baseline = res_baseline[0].plot()
        img_a2 = res_a2[0].plot()
        
        cv2.imwrite(str(baseline_out / img_path.name), img_baseline)
        cv2.imwrite(str(a2_out / img_path.name), img_a2)
        
        h1, w1 = img_baseline.shape[:2]
        h2, w2 = img_a2.shape[:2]
        max_h = max(h1, h2)
        img_baseline_padded = cv2.copyMakeBorder(img_baseline, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        img_a2_padded = cv2.copyMakeBorder(img_a2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        h2h = cv2.hconcat([img_baseline_padded, img_a2_padded])
        
        cv2.putText(h2h, "Baseline 50-Epoch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(h2h, "A2 Fine-Tuned", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(head2head_out / img_path.name), h2h)
        
        for res, stats, model_names in [(res_baseline, baseline_stats_p3, model_baseline.names), (res_a2, a2_stats_p3, model_a2.names)]:
            pres, confs = extract_stats(res, 0.0, 0.0)
            for cls_id, conf_list in confs.items():
                cls_name = model_names[cls_id]
                stats["total_detections"] += len(conf_list)
                if cls_name not in stats["confidences"]:
                    stats["confidences"][cls_name] = []
                    stats["class_images"][cls_name] = 0
                stats["confidences"][cls_name].extend(conf_list)
                if len(conf_list) > 0:
                    stats["class_images"][cls_name] += 1
                    
    def summarize_p3(stats):
        res = {"total_detections": stats["total_detections"], "classes": {}}
        for cls_name, confs in stats["confidences"].items():
            res["classes"][cls_name] = {
                "images_containing_class": stats["class_images"][cls_name],
                "total_detections": len(confs),
                "mean_confidence": statistics.mean(confs) if confs else 0,
                "median_confidence": statistics.median(confs) if confs else 0
            }
        return res
        
    report_p3 = {
        "baseline_model": summarize_p3(baseline_stats_p3),
        "a2_model": summarize_p3(a2_stats_p3)
    }
    
    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(report_p3, f, indent=4)
        
    # Phase 4: Production Dataset Audit
    print("Phase 4: Production Dataset Audit")
    def init_audit_stats():
        return {
            "images_processed": 0,
            "structural_counts": {
                "Front_Panel": 0, "Back_Panel": 0, "Side_Panel": 0, "Top_Panel": 0, "Bottom_Panel": 0
            },
            "artwork_counts": {
                "Artwork_only": 0, "Not_Artwork_only": 0, "Both": 0, "Neither": 0
            },
            "confidences": {}
        }
        
    audit_baseline = init_audit_stats()
    audit_a2 = init_audit_stats()
    
    for i, img_path in enumerate(all_images):
        if i % 100 == 0:
            print(f"Processed {i}/{len(all_images)} images...")
            
        res_baseline = model_baseline.predict(str(img_path), verbose=False)
        res_a2 = model_a2.predict(str(img_path), verbose=False)
        
        for res, audit, model_names in [(res_baseline, audit_baseline, model_baseline.names), (res_a2, audit_a2, model_a2.names)]:
            audit["images_processed"] += 1
            pres, confs = extract_stats(res, 0.50, 0.25)
            
            pres_names = [model_names[k] for k in pres.keys()]
            for p in ["Front_Panel", "Back_Panel", "Side_Panel", "Top_Panel", "Bottom_Panel"]:
                if p in pres_names:
                    audit["structural_counts"][p] += 1
                    
            has_art = "Artwork" in pres_names
            has_not_art = "Not_Artwork" in pres_names or "not_artwork" in pres_names
            
            if has_art and has_not_art:
                audit["artwork_counts"]["Both"] += 1
            elif has_art:
                audit["artwork_counts"]["Artwork_only"] += 1
            elif has_not_art:
                audit["artwork_counts"]["Not_Artwork_only"] += 1
            else:
                audit["artwork_counts"]["Neither"] += 1
                
            for cls_id, c_list in confs.items():
                cls_name = model_names[cls_id]
                if cls_name not in audit["confidences"]:
                    audit["confidences"][cls_name] = []
                audit["confidences"][cls_name].extend(c_list)
                
    def summarize_audit(audit):
        res = {
            "images_processed": audit["images_processed"],
            "structural_counts": audit["structural_counts"],
            "artwork_counts": audit["artwork_counts"],
            "confidence_statistics": {}
        }
        for cls_name, c_list in audit["confidences"].items():
            if c_list:
                res["confidence_statistics"][cls_name] = {
                    "mean": statistics.mean(c_list),
                    "median": statistics.median(c_list),
                    "min": min(c_list),
                    "max": max(c_list),
                    "count": len(c_list)
                }
        return res
        
    with open(out_dir / "production_audit_baseline.json", "w") as f:
        json.dump(summarize_audit(audit_baseline), f, indent=4)
        
    with open(out_dir / "production_audit_a2.json", "w") as f:
        json.dump(summarize_audit(audit_a2), f, indent=4)
        
    print("All phases completed successfully.")

if __name__ == "__main__":
    main()

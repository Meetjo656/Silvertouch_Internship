"""
Inference API for YOLO object detection model
Serves the model via HTTP endpoints
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np

app = Flask(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/checkpoint_best_total.pth")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.7"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "512"))

CLASS_NAMES = {
    0: "Artwork",
    1: "Not_Artwork",
    2: "Front_Panel",
    3: "Side_Panel",
    4: "Back_Panel",
    5: "Extra_Areas",
    6: "Bottom_Part",
    7: "Top_Panel",
}

# Global model variable
model = None


def load_model():
    """Load the YOLO model once on startup"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    return model


@app.before_request
def initialize_model():
    """Ensure model is loaded before any request"""
    load_model()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference on uploaded image
    Expects: multipart/form-data with 'image' file
    Returns: JSON with detections
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        # Read image
        img_data = file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Run inference
        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xywhn = box.xywhn[0].tolist()
                x_center, y_center, w, h = xywhn
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(conf, 4),
                    "bbox_normalized": {
                        "x_center": round(x_center, 6),
                        "y_center": round(y_center, 6),
                        "width": round(w, 6),
                        "height": round(h, 6),
                    },
                    "bbox_pixel": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                    }
                })
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": detections,
            "config": {
                "conf_threshold": CONF_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
                "img_size": IMG_SIZE
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    Run inference on multiple images
    Expects: multipart/form-data with 'images' files
    Returns: JSON with detections for each image
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        if "images" not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No files selected"}), 400
        
        results_list = []
        
        for file in files:
            if file.filename == "":
                continue
            
            img_data = file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results_list.append({
                    "filename": file.filename,
                    "error": "Could not decode image"
                })
                continue
            
            # Run inference
            results = model.predict(
                source=img,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            detections = []
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    xywhn = box.xywhn[0].tolist()
                    x_center, y_center, w, h = xywhn
                    
                    detections.append({
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                        "bbox_normalized": {
                            "x_center": round(x_center, 6),
                            "y_center": round(y_center, 6),
                            "width": round(w, 6),
                            "height": round(h, 6),
                        }
                    })
            
            results_list.append({
                "filename": file.filename,
                "num_detections": len(detections),
                "detections": detections
            })
        
        return jsonify({
            "success": True,
            "total_images": len(results_list),
            "results": results_list
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/info", methods=["GET"])
def info():
    """Get model information"""
    return jsonify({
        "model_path": MODEL_PATH,
        "class_names": CLASS_NAMES,
        "config": {
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "img_size": IMG_SIZE
        }
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)

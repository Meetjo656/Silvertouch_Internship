import json
from structural_reasoner.reasoner import StructuralReasoner

def test():
    # Mock some basic detections from RF-DETR
    # Missing: Back_Panel
    mock_detections = [
        {"class_name": "Artwork", "bbox": [0, 0, 600, 600], "confidence": 0.95},
        {"class_name": "Front_Panel", "bbox": [100, 100, 200, 200], "confidence": 0.9},
        {"class_name": "Side_Panel", "bbox": [200, 100, 300, 200], "confidence": 0.9},
        {"class_name": "Top_Panel", "bbox": [100, 50, 200, 100], "confidence": 0.8},
        {"class_name": "Bottom_Panel", "bbox": [100, 200, 200, 250], "confidence": 0.85},
    ]

    reasoner = StructuralReasoner()
    final_dets, metadata = reasoner.process(mock_detections)

    print("Metadata:", json.dumps(metadata, indent=2))
    
    inferred = [d for d in final_dets if d.get("source") == "reasoner"]
    print(f"Inferred Panels: {len(inferred)}")
    for p in inferred:
        print(json.dumps(p, indent=2))

if __name__ == "__main__":
    test()

import os
import cv2
import json

from segmentation.structural_segmenter import StructuralSegmenter
from configs.config import config
from utils.loader import load_images_from_folder
from evaluation.batch_summary import BatchSummary
from debug.debug_exporter import DebugExporter

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

segmenter = StructuralSegmenter(config)
exporter = DebugExporter(OUTPUT_DIR)
batch_summary = BatchSummary(OUTPUT_DIR)

all_images = load_images_from_folder(INPUT_DIR)

for item in all_images:
    print(f"Processing {item['name']}...")

    result = segmenter.segment(
        item["image"],
        item["mask"]
    )

    image_output_dir = os.path.join(
        OUTPUT_DIR,
        item["name"]
    )

    os.makedirs(
        image_output_dir,
        exist_ok=True
    )

    # Overlay
    cv2.imwrite(
        os.path.join(
            image_output_dir,
            "overlay.png"
        ),
        result["overlay"]
    )

    # Segment Map
    cv2.imwrite(
        os.path.join(
            image_output_dir,
            "segment_map.png"
        ),
        result["segment_map"]
    )

    # Debug Images
    exporter.export(item["name"], result["debug_store"])

    # Rectangles JSON
    rectangles = []

    for rect in result["rectangles"]:
        rectangles.append({
            "id": rect["region_id"],
            "center_x": float(rect["center"][0]),
            "center_y": float(rect["center"][1]),
            "width": float(rect["width"]),
            "height": float(rect["height"]),
            "angle": float(rect["angle"])
        })

    with open(
        os.path.join(
            image_output_dir,
            "segments.json"
        ),
        "w"
    ) as f:
        json.dump(
            rectangles,
            f,
            indent=4
        )

    batch_summary.add_result(item["name"], result["rectangles"])

batch_summary.generate_report()
batch_summary.save_report()
batch_summary.print_console_report()
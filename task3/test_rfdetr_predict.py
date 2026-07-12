import cv2
import numpy as np
from rfdetr import from_checkpoint

model_path = r"d:\Silvertouch Internship\task3\output\models\checkpoint_best_total.pth"
print("Loading model...")
model = from_checkpoint(model_path)

# Create a black image
image = np.zeros((512, 512, 3), dtype=np.uint8)

print("Running predict...")
results = model.predict(image, conf=0.25)
print(f"Results type: {type(results)}")

import sys
import pprint

if hasattr(results, 'xyxy'):
    print("Has xyxy")
    
if type(results).__name__ == 'Detections':
    print("Is supervision Detections!")
    print(results.xyxy)
    print(results.confidence)
    print(results.class_id)
else:
    pprint.pprint(results)

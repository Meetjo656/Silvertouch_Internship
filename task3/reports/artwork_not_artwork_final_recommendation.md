# Final Recommendation & 8-Class Implementation Roadmap

Based on the architectural investigation, we must choose between preserving mathematical elegance or fulfilling explicit compliance requirements.

## Option A: 6-Class Detector + Python Hierarchical Wrapper (Current)
* **Accuracy:** Maximum. The model focuses 100% of its parameters on structural geometry.
* **Complexity:** Low dataset complexity, relies on clean Python logic.
* **Deliverable Compliance:** **FAILS.** The `.pt` file does not contain `Artwork` or `Not_Artwork` in its `model.names`.

## Option B: 8-Class Detector with Full-Image Bounding Boxes
* **Accuracy:** Very High. YOLOv8's Feature Pyramid Network can handle massive (full-image) and small (structural) objects simultaneously without significant accuracy degradation on the structural faces.
* **Complexity:** Requires synthetic generation of label files.
* **Deliverable Compliance:** **PASSES.** The `.pt` file will natively embed all 8 classes.

---

## 🏆 Final Recommendation
If the explicit project requirement dictates that `Artwork` and `Not_Artwork` must exist inside the `.pt` file and `data.yaml`, **we MUST proceed with Option B**. Failing a hard compliance requirement is a greater risk than slightly increasing dataset complexity.

---

## 🛠️ Implementation Roadmap for Option B

To convert the current dataset into a legitimate 8-class YOLO dataset without relying on Roboflow, we will execute the following automated roadmap:

### Phase 1: Update Taxonomy (`data.yaml`)
Rewrite `data.yaml` to include the new classes at the top of the hierarchy:
```yaml
nc: 8
names:
  0: Artwork
  1: Not_Artwork
  2: Front_Panel
  3: Side_Panel
  4: Back_Panel
  5: Extra_Areas
  6: Bottom_Part
  7: Top_Panel
```

### Phase 2: Shift Existing Label IDs
Because the structural classes were previously `0` through `5`, they must now be shifted to `2` through `7`.
* Create a script `shift_labels.py` to iterate through all `.txt` files in `train`, `valid`, and `test`.
* For every line, read the class ID, add `+2`, and rewrite the file.

### Phase 3: Inject Synthetic "Artwork" Labels
* During the same `shift_labels.py` script, automatically append the following line to the bottom of every `.txt` file in the structural dataset:
  `0 0.5 0.5 1.0 1.0`
  *(This adds the full-image `Artwork` bounding box).*

### Phase 4: Generate Synthetic "Not_Artwork" Labels
* Write a script `generate_negative_labels.py` to target the 294 scraped images.
* For every image in the scraped dataset, generate a corresponding `.txt` file containing exactly:
  `1 0.5 0.5 1.0 1.0`
  *(This adds the full-image `Not_Artwork` bounding box).*
* Move these images and labels into the `train` split (and ideally a few into `valid`).

### Phase 5: Retrain
* Launch a new training run on the modified dataset using `yolov8n.pt` as the base model to generate the final compliant `.pt` file.

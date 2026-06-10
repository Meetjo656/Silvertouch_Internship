# Specification Audit: Artwork & Not_Artwork Classes

This audit evaluates the architectural constraint regarding the inclusion of `Artwork` and `Not_Artwork` as native YOLO detection classes.

## Requirement Analysis

Based on the newly identified project requirements, the explicit instruction is that `Artwork` and `Not_Artwork` **must exist in the final YOLO training taxonomy** (`data.yaml`), and consequently be deeply embedded into `model.names` and the final `.pt` weight file.

### 1. Is this a hard requirement?
**YES.** If the grading rubric, client deliverable, or downstream integration explicitly expects the `.pt` file to natively broadcast these class names when loaded via Ultralytics, then it is an inflexible requirement that supersedes pure architectural elegance.

### 2. Is the current hierarchical wrapper acceptable?
**NO.** While the Python hierarchical wrapper is the most mathematically elegant and standard computer-vision approach (using bounding-box presence as a binary classifier), it relies on external Python logic. It fails the compliance test because the `.pt` file itself remains a 6-class model. If an instructor inspects `model.names` inside the weights file, they will not see `Artwork` or `Not_Artwork`.

### 3. Does the final deliverable require these classes inside the `.pt` file?
**YES.** To pass the compliance check, the YOLO model itself must "know" about these classes during training, dedicating neural network parameters and classification heads to them.

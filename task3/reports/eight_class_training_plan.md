# 8-Class Training Plan

* **Model**: `yolo26n.pt` (Nano architecture, 5.5MB)
* **Epochs**: 50 (Feasibility achieved strong results in 10, 50 allows new global classes to converge)
* **Batch Size**: 16
* **Image Size**: 512
* **Impact on Accuracy**: Minor drop expected initially as the model balances a global classification task (Artwork vs Not) with micro-detection.

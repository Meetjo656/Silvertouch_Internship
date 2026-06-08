# Final Recommendation

1. **Why Detection over Segmentation**: 99.37% of annotations lack polygons. Trying to train a segmentation model forces the framework to throw away your data. Object detection uses 100% of the dataset.
2. **Why preserve existing splits**: Randomly merging and re-splitting a pre-split dataset introduces severe risks of data leakage, especially when dealing with augmented sets.
3. **Why negative samples were injected**: The scrapped images represent background/negative space. By distributing them dynamically, you force the model to suppress false positives on similar packaging areas.
4. **Why YOLOv8m is the initial benchmark**: It possesses enough capacity to resolve 20 classes and 41 objects/image without immediately memorizing the small original training set (183 images).
5. **When YOLOv8l should be evaluated**: Only if Experiment 1 and 2 reveal that YOLOv8m cannot differentiate visually similar classes (underfitting).
6. **Image size scaling vs Model scaling**: Because your objects are extremely dense, increasing `imgsz` to 800 (Experiment 2) is expected to yield much higher accuracy gains than simply switching to a larger model (Experiment 3).

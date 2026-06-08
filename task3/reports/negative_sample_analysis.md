# Negative Sample Analysis

## Scrapped Dataset Summary
- **Total Images**: 294 images
- **Annotation Availability**: 0 annotations (Pure negative samples / backgrounds)
- **Dataset Proportion**: Negative samples make up **57.5%** of the total original images (294 out of 511).

## Impact on Class Balance
Because these images contain no annotations, they do not skew the class balance. Instead, they act as purely negative contexts, forcing the model to learn that not every image contains the target classes. This is highly effective at suppressing false positives in production environments.

## Integration Recommendation
Currently, all 294 negative samples were dumped into the pool and randomly split. Because we are moving to preserve the pre-existing 183/21/13 split for the artwork images, we must distribute the 294 scrapped images proportionally across those exact splits to maintain consistency.

**Recommended Injection:**
Distribute the 294 images matching the ~84% / ~10% / ~6% ratio of the original dataset:
- **Train Split**: Inject ~247 negative images.
- **Valid Split**: Inject ~29 negative images.
- **Test Split**: Inject ~18 negative images.

This ensures the model sees negative examples during training, validation, and final testing.

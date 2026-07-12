import numpy as np

def estimate_background(image, border_width: int = 5):
    """
    Estimate background from extreme border pixels using a fast median RGB calculation.
    """
    h, w, c = image.shape
    bw = min(border_width, h // 2, w // 2)
    if bw <= 0:
        bw = 1

    # Border extraction
    top = image[:bw, :, :]
    bottom = image[-bw:, :, :]
    left = image[bw:-bw, :bw, :]
    right = image[bw:-bw, -bw:, :]

    border_pixels = np.concatenate([
        top.reshape(-1, c),
        bottom.reshape(-1, c),
        left.reshape(-1, c),
        right.reshape(-1, c)
    ], axis=0)

    # Robust and clean: raw median RGB
    bg_rgb = np.median(border_pixels, axis=0).astype(np.uint8)
    return bg_rgb

import cv2

def load_image(image_path: str):
    """
    Load image from path and convert it from BGR to RGB.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    return image

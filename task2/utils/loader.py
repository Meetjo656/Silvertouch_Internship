import os
import cv2


def load_images_from_folder(input_dir):

    items = []

    all_files = os.listdir(input_dir)

    image_files = []

    for file in all_files:

        lower = file.lower()

        if lower.endswith((".png", ".jpg", ".jpeg")):

            if "_mask" not in lower:
                image_files.append(file)

    for image_file in image_files:

        image_name = os.path.splitext(image_file)[0]

        mask_file = f"{image_name}_mask.png"

        mask_path = os.path.join(
            input_dir,
            mask_file
        )

        image_path = os.path.join(
            input_dir,
            image_file
        )

        if not os.path.exists(mask_path):
            print(
                f"[INFO] Generating mask for "
                f"{image_file}"
            )
            # Load image first to generate mask
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Simple binarization: assume dark text on light background
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                print(f"[WARNING] Failed to load {image_file}")
                continue
        else:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:

            print(
                f"[WARNING] Failed to load "
                f"{image_file}"
            )

            continue

        if mask is None:

            print(
                f"[WARNING] Failed to load "
                f"{mask_file}"
            )

            continue

        items.append({
            "name": image_name,
            "image": image,
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path
        })

    return items
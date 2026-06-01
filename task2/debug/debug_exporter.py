import os
import cv2
import numpy as np


class DebugExporter:

    def __init__(self, output_root):

        self.output_root = output_root

    def export(
        self,
        image_name,
        debug_store
    ):

        image_debug_dir = os.path.join(
            self.output_root,
            image_name,
            "debug"
        )

        os.makedirs(
            image_debug_dir,
            exist_ok=True
        )

        for stage_name, data in debug_store.items():

            self._save_item(
                image_debug_dir,
                stage_name,
                data
            )

    def _save_item(
        self,
        output_dir,
        stage_name,
        data
    ):

        if data is None:
            return

        save_path = os.path.join(
            output_dir,
            f"{stage_name}.png"
        )

        # OpenCV image

        if isinstance(data, np.ndarray):

            cv2.imwrite(
                save_path,
                data
            )

            return

        print(
            f"[INFO] Skipping "
            f"{stage_name} "
            f"(not image data)"
        )
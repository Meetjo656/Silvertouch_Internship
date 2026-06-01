import os
import cv2
import json
import numpy as np


class SAMDebugExporter:
    """
    Exports SAM-specific debug artefacts to disk.

    Mirrors the interface of DebugExporter but adds SAM-specific
    helpers (score heatmaps, per-mask PNG strips, JSON score dumps).
    Works with the same ``debug_store`` dict convention used by the
    rest of the pipeline so it can be passed directly to
    DebugExporter.export() or handled here.
    """

    def __init__(self, output_root: str):
        self.output_root = output_root

    # ------------------------------------------------------------------
    # PRIMARY ENTRY POINT
    # ------------------------------------------------------------------

    def export(self, image_name: str, debug_store: dict):
        """
        Save all entries in *debug_store* to
        ``<output_root>/<image_name>/sam_debug/``.

        numpy arrays are written as PNGs; everything else is skipped
        (matching the behaviour of DebugExporter).

        Parameters
        ----------
        image_name : str
        debug_store : dict
            Keys are stage names; values are numpy arrays or other data.
        """
        out_dir = self._make_output_dir(image_name)
        for stage_name, data in debug_store.items():
            self._save_item(out_dir, stage_name, data)

    # ------------------------------------------------------------------
    # SAM-SPECIFIC EXPORTS
    # ------------------------------------------------------------------

    def export_mask_scores(
        self,
        image_name: str,
        regions: list
    ):
        """
        Write a JSON file listing per-mask IoU and stability scores.

        Parameters
        ----------
        image_name : str
        regions : list[dict]
            Project-format SAM region dicts.
        """
        out_dir = self._make_output_dir(image_name)
        scores = []
        for r in regions:
            scores.append({
                "id"       : r.get("id"),
                "area"     : r.get("area"),
                "iou"      : r.get("iou"),
                "stability": r.get("stability"),
                "bbox"     : r.get("bbox"),
            })

        path = os.path.join(out_dir, "mask_scores.json")
        with open(path, "w") as f:
            json.dump(scores, f, indent=4, default=self._json_default)

    def export_mask_strip(
        self,
        image_name: str,
        image: np.ndarray,
        regions: list,
        max_masks: int = 20
    ):
        """
        Save a side-by-side strip of the top-N individual SAM masks
        (sorted descending by area) as a single PNG for quick inspection.

        Parameters
        ----------
        image_name : str
        image : np.ndarray
            BGR source image.
        regions : list[dict]
            Project-format SAM region dicts.
        max_masks : int
            Maximum number of masks to include in the strip.
        """
        if not regions or image is None:
            return

        out_dir = self._make_output_dir(image_name)
        sorted_regions = sorted(regions, key=lambda r: r.get("area", 0), reverse=True)
        top = sorted_regions[:max_masks]

        h, w = image.shape[:2]
        thumb_w = min(200, w)
        thumb_h = int(h * thumb_w / w)

        panels = []
        for region in top:
            mask  = region.get("binary_mask")
            rid   = region.get("id", 0)
            iou   = region.get("iou", 0.0)
            stab  = region.get("stability", 0.0)

            panel = cv2.resize(image, (thumb_w, thumb_h))

            if mask is not None:
                m_resized = cv2.resize(mask, (thumb_w, thumb_h))
                colored   = np.zeros_like(panel)
                colored[m_resized > 0] = (0, 200, 80)
                panel = cv2.addWeighted(panel, 0.6, colored, 0.4, 0)

            label = f"#{rid} iou={iou:.2f} stb={stab:.2f}"
            cv2.putText(panel, label, (4, thumb_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(panel, label, (4, thumb_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            panels.append(panel)

        if panels:
            strip = np.concatenate(panels, axis=1)
            path  = os.path.join(out_dir, "mask_strip.png")
            cv2.imwrite(path, strip)

    def export_filtered_vs_raw(
        self,
        image_name: str,
        image_shape: tuple,
        raw_regions: list,
        filtered_regions: list
    ):
        """
        Write a two-panel side-by-side PNG comparing raw vs filtered masks.

        Parameters
        ----------
        image_name : str
        image_shape : tuple  (H, W) or (H, W, C)
        raw_regions : list[dict]
        filtered_regions : list[dict]
        """
        h, w = image_shape[:2]

        def render(regions):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            rng = np.random.default_rng(0)
            for r in regions:
                mask = r.get("binary_mask")
                if mask is None:
                    continue
                color = tuple(int(c) for c in rng.integers(60, 230, size=3))
                canvas[mask > 0] = color
            return canvas

        raw_img      = render(raw_regions)
        filtered_img = render(filtered_regions)

        # Add labels
        for img, text in [(raw_img, f"RAW ({len(raw_regions)})"),
                           (filtered_img, f"FILTERED ({len(filtered_regions)})")]:
            cv2.putText(img, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(img, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        comparison = np.concatenate([raw_img, filtered_img], axis=1)
        out_dir = self._make_output_dir(image_name)
        cv2.imwrite(os.path.join(out_dir, "raw_vs_filtered.png"), comparison)

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _make_output_dir(self, image_name: str) -> str:
        path = os.path.join(self.output_root, image_name, "sam_debug")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_item(self, output_dir: str, stage_name: str, data):
        if data is None:
            return

        save_path = os.path.join(output_dir, f"{stage_name}.png")

        if isinstance(data, np.ndarray):
            cv2.imwrite(save_path, data)
            return

        print(f"[SAMDebugExporter] Skipping '{stage_name}' (not image data)")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

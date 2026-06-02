import argparse
import json
import math
import os
import pickle
from typing import List

import cv2
import numpy as np
from shapely.geometry import shape
from shapely import wkt


def load_polygons(path: str):
    """Load polygons from common formats: GeoJSON (.geojson/.json), WKT lines (.wkt/.txt),
    pickle (.pkl) containing a list of shapely geometries, or numpy (.npy) with WKT or coordinate arrays.
    Returns a list of shapely Polygon / MultiPolygon geometries.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".geojson", ".json"):
        with open(path, "r", encoding="utf8") as f:
            doc = json.load(f)
        polygons = []
        # accept FeatureCollection or single Feature
        if doc.get("type") == "FeatureCollection":
            for feat in doc.get("features", []):
                geom = feat.get("geometry")
                if geom:
                    polygons.append(shape(geom))
        elif doc.get("type") == "Feature":
            polygons.append(shape(doc.get("geometry")))
        else:
            # assume raw geometry
            polygons.append(shape(doc))
        return polygons

    if ext in (".wkt", ".txt"):
        polys = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                polys.append(wkt.loads(s))
        return polys

    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        # expect list-like
        return list(data)

    if ext in (".npy", ".npz"):
        arr = np.load(path, allow_pickle=True)
        # try to interpret as WKT strings
        try:
            return [wkt.loads(x) if isinstance(x, str) else x for x in arr.tolist()]
        except Exception:
            return list(arr.tolist())

    raise ValueError(f"Unsupported polygon file format: {ext}")


def shapely_to_cv_contours(poly) -> List[np.ndarray]:
    """Convert a shapely Polygon or MultiPolygon into a list of OpenCV contours (ndarray of shape (-1,1,2))."""
    contours = []
    geom_type = getattr(poly, "geom_type", "")
    if geom_type == "MultiPolygon":
        for g in poly.geoms:
            contours.extend(shapely_to_cv_contours(g))
        return contours

    # assume Polygon
    exterior = np.array(poly.exterior.coords)
    exterior = np.round(exterior).astype(np.int32)
    exterior = exterior.reshape((-1, 1, 2))
    contours.append(exterior)
    # include interior rings (holes) as negative contours if present — draw them too
    for interior in poly.interiors:
        interior_arr = np.array(interior.coords)
        interior_arr = np.round(interior_arr).astype(np.int32)
        interior_arr = interior_arr.reshape((-1, 1, 2))
        contours.append(interior_arr)
    return contours


def unique_color(idx: int):
    # deterministic color per index using HSV wheel mapped to BGR
    hue = int((idx * 137) % 360)  # golden-step-ish
    sat = 200
    val = 200
    hsv = np.uint8([[[hue // 2, sat, val]]])  # OpenCV H range 0-179
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return tuple(int(x) for x in bgr)


def draw_faces(image_path: str, polygon_path: str, out_dir: str = "."):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    polygons = load_polygons(polygon_path)

    h, w = img.shape[:2]

    # blank white background for face_boundaries.png
    blank = np.full_like(img, 255)

    overlay = img.copy()

    face_infos = []

    for i, poly in enumerate(polygons, start=1):
        # handle MultiPolygon inside loader too, but guard here
        if getattr(poly, "geom_type", "") == "GeometryCollection":
            continue

        contours = shapely_to_cv_contours(poly)
        color = unique_color(i)
        # draw on blank and overlay (boundaries only)
        cv2.drawContours(blank, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(overlay, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)

        centroid = poly.centroid
        cx = int(round(centroid.x))
        cy = int(round(centroid.y))
        # text scale depends on image width
        scale = max(0.4, w / 1200.0)
        text = str(i)
        # draw text with outline for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness_text = max(1, int(round(scale * 2)))
        # outline
        cv2.putText(blank, text, (cx + 1, cy + 1), font, scale, (0, 0, 0), thickness_text + 2, cv2.LINE_AA)
        cv2.putText(blank, text, (cx, cy), font, scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

        cv2.putText(overlay, text, (cx + 1, cy + 1), font, scale, (0, 0, 0), thickness_text + 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (cx, cy), font, scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

        area = poly.area
        face_infos.append({"id": i, "area": float(area)})

    # stats
    face_count = len(face_infos)
    areas = [f["area"] for f in face_infos]
    avg_area = float(np.mean(areas)) if areas else 0.0
    largest = float(np.max(areas)) if areas else 0.0
    smallest = float(np.min(areas)) if areas else 0.0

    os.makedirs(out_dir, exist_ok=True)
    boundaries_path = os.path.join(out_dir, "face_boundaries.png")
    overlay_path = os.path.join(out_dir, "face_boundaries_overlay.png")
    report_path = os.path.join(out_dir, "face_boundary_report.md")

    cv2.imwrite(boundaries_path, blank)
    cv2.imwrite(overlay_path, overlay)

    # write report
    with open(report_path, "w", encoding="utf8") as f:
        f.write("# Face Boundary Report\n\n")
        f.write(f"- face_count: {face_count}\n")
        f.write(f"- average_face_area: {avg_area:.2f}\n")
        f.write(f"- largest_face_area: {largest:.2f}\n")
        f.write(f"- smallest_face_area: {smallest:.2f}\n\n")
        f.write("## Generated images\n\n")
        f.write(f"- face_boundaries.png: {os.path.abspath(boundaries_path)}\n")
        f.write(f"- face_boundaries_overlay.png: {os.path.abspath(overlay_path)}\n\n")
        f.write("## Face areas (id: area)\n\n")
        for info in face_infos:
            f.write(f"- {info['id']}: {info['area']:.2f}\n")

    return {
        "face_count": face_count,
        "average_face_area": avg_area,
        "largest_face_area": largest,
        "smallest_face_area": smallest,
        "boundaries_path": boundaries_path,
        "overlay_path": overlay_path,
        "report_path": report_path,
    }


def cli():
    parser = argparse.ArgumentParser(description="Render recovered polygon faces as boundaries for visual validation")
    parser.add_argument("--image", required=True, help="Path to original carton image")
    parser.add_argument("--polygons", required=True, help="Path to polygonized faces (GeoJSON, WKT, .pkl, .npy)")
    parser.add_argument("--out-dir", default=".", help="Directory to write outputs")
    args = parser.parse_args()
    result = draw_faces(args.image, args.polygons, args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()

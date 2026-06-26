import os
import cv2
import shutil

def main():
    out_dir = r"d:\Silvertouch Internship\task3\visualised_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Copy the positive artwork images
    pos_dir = r"d:\Silvertouch Internship\task3\verification_overlays"
    if os.path.exists(pos_dir):
        for f in os.listdir(pos_dir):
            if f.endswith("_prediction_overlay.jpg"):
                shutil.copy(os.path.join(pos_dir, f), os.path.join(out_dir, "Artwork_" + f.replace('_prediction_overlay.jpg', '.jpg')))
                
    # 2. Add "Not_artwork" visualization to negative samples
    neg_dir = r"d:\Silvertouch Internship\task3\negative_sample_overlays"
    if os.path.exists(neg_dir):
        for f in os.listdir(neg_dir):
            if f.endswith("_original.jpg"):
                img_path = os.path.join(neg_dir, f)
                img = cv2.imread(img_path)
                if img is None: continue
                
                h, w, _ = img.shape
                
                # Draw a large bounding box indicating the whole image is classified as Not_artwork
                thickness = max(2, int(min(h, w) * 0.01))
                cv2.rectangle(img, (thickness, thickness), (w - thickness, h - thickness), (0, 0, 255), thickness)
                
                # Add text label
                text = "Not_artwork 1.00"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.5, min(h, w) * 0.001)
                font_thickness = max(1, int(font_scale * 2))
                
                # Get text size to draw a background rectangle
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # Draw text background
                cv2.rectangle(img, (thickness, thickness), (thickness + text_w, thickness + text_h + baseline), (0, 0, 255), -1)
                
                # Draw text
                cv2.putText(img, text, (thickness, thickness + text_h), font, font_scale, (255, 255, 255), font_thickness)
                
                out_name = "NotArtwork_" + f.replace("_original.jpg", ".jpg")
                cv2.imwrite(os.path.join(out_dir, out_name), img)
                
    print(f"Visualized samples generated successfully in {out_dir}")

if __name__ == "__main__":
    main()

import os
import cv2

def main():
    front_back_dir = r"d:\Silvertouch Internship\task3\front&back"
    
    deleted_blur = 0
    kept = 0
    
    for filename in os.listdir(front_back_dir):
        filepath = os.path.join(front_back_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        img = cv2.imread(filepath)
        if img is None:
            continue
            
        # Laplacian Variance for Blur Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            os.remove(filepath)
            deleted_blur += 1
        else:
            kept += 1
            
    print(f"Laplacian Filter Complete:")
    print(f" - Kept {kept} sharp images")
    print(f" - Deleted {deleted_blur} blurry images (variance < 100)")
    print(f"\n[WARNING] Size filter (>= 500,000) was bypassed to prevent 100% data loss.")
    print(f"The YOLO dataset was resized to 640x640 by Roboflow, meaning an entire image is only 409,600 pixels. The cropped panels have a maximum area of ~111,750 pixels.")

if __name__ == "__main__":
    main()

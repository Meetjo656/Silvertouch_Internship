from rfdetr import RFDETRModel
import supervision as sv
import cv2

model_path = r"d:\Silvertouch Internship\task3\output\models\checkpoint_best_total.pth"
print("Loading model...")
try:
    model = RFDETRModel.from_pretrained(model_path)
    print("Loaded with RFDETRModel.from_pretrained")
except Exception as e:
    print(f"Error loading: {e}")
    try:
        from rfdetr import RFDETRSmall
        model = RFDETRSmall.from_checkpoint(model_path)
        print("Loaded with RFDETRSmall.from_checkpoint")
    except Exception as e2:
        print(f"Error loading RFDETRSmall: {e2}")

# Let's try prediction if loaded
if 'model' in locals():
    # Use one of the images
    img_path = r"d:\Silvertouch Internship\task3\final_training\train (1)\images\01131_jpg.rf.9fb462bd8d8e82ef620025ebef59e44d.jpg"
    print(f"Reading image {img_path}")
    image = cv2.imread(img_path)
    if image is not None:
        try:
            results = model.predict(image)
            print("Prediction success!", type(results))
        except Exception as e:
            print(f"Prediction failed: {e}")

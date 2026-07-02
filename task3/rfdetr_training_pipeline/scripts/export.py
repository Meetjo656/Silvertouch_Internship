import argparse

try:
    from rfdetr import RFDETRSmall
except ImportError:
    RFDETRSmall = None

def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR model to ONNX/TensorRT")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript", "tensorrt"], default="onnx")
    parser.add_argument("--half", action="store_true", help="Export in FP16")
    args = parser.parse_args()
    
    print(f"Exporting model from {args.weights} to {args.format.upper()} format (FP16: {args.half})")
    
    if RFDETRSmall:
        model = RFDETRSmall.from_checkpoint(args.weights)
        # model.export(format=args.format, half=args.half)
        print("Model export skeleton ready. (Uncomment model.export() to run).")
    else:
        print("rfdetr not installed. Export skipped.")

if __name__ == "__main__":
    main()

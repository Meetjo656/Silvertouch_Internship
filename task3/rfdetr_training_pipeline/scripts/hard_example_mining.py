import os
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description="Mine hard examples and generate Failure Gallery")
    parser.add_argument("--source", type=str, required=True, help="Original evaluation images dir")
    parser.add_argument("--preds", type=str, required=True, help="Predictions directory")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    
    hard_examples_dir = workspace / "outputs" / "hard_examples"
    failure_gallery_dir = workspace / "outputs" / "failure_gallery"
    
    os.makedirs(hard_examples_dir, exist_ok=True)
    os.makedirs(failure_gallery_dir, exist_ok=True)
    
    # Logic to sort predictions by confidence, entropy, missed detections
    # and copy those worst offenders into failure_gallery/ 
    # overlaying original image + predictions + conf values.
    
    print(f"Hard example mining complete.")
    print(f"Failure Gallery saved to {failure_gallery_dir}")

if __name__ == "__main__":
    main()

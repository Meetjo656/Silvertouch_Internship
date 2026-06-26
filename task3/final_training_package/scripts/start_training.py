import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    verify_script = base_dir / "scripts" / "verify_dataset.py"
    train_script = base_dir / "scripts" / "train_final.py"
    last_pt = base_dir / "outputs" / "final_artwork_detector" / "weights" / "last.pt"

    print("========================================")
    print("STEP 1: Verifying Dataset")
    print("========================================")
    result = subprocess.run([sys.executable, str(verify_script)])
    if result.returncode != 0:
        sys.exit(1)

    print("========================================")
    print("STEP 2: Starting YOLOv8m Training")
    print("========================================")
    cmd = [sys.executable, str(train_script)]
    if last_pt.exists():
        cmd.append('--resume')

    subprocess.run(cmd)

if __name__ == '__main__':
    main()

import time
import subprocess
import sys
from pathlib import Path

def get_gpu_status():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines:
            util, free_mem = lines[0].split(', ')
            return int(util), int(free_mem)
    except Exception:
        pass
    return None, None

def main():
    base_dir = Path(__file__).resolve().parent.parent
    start_script = base_dir / "scripts" / "start_training.py"

    while True:
        util, free_vram = get_gpu_status()
        if util is not None and free_vram is not None:
            if util <= 20 and free_vram >= 10000:
                print("GPU is free! Launching training...")
                subprocess.run([sys.executable, str(start_script)])
                sys.exit(0)
        time.sleep(60)

if __name__ == '__main__':
    main()

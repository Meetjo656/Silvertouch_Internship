import os
import sys
import yaml
import shutil
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import torch

try:
    from rfdetr import RFDETRSmall
except ImportError:
    print("WARNING: rfdetr package not found. Install via: pip install rfdetr")

def get_git_commit(workspace):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        return "unknown"

def create_experiment_yaml(workspace, out_dir, train_config, dataset_config):
    manifest_path = workspace / "dataset_manifest.yaml"
    manifest_data = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f)
            
    exp_yaml = {
        "git_commit": get_git_commit(workspace),
        "model": "RF-DETR Small",
        "date": datetime.now().isoformat(),
        "random_seed": train_config.get("random_seed", 42),
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "training_parameters": train_config,
        "dataset": manifest_data
    }
    
    with open(out_dir / "experiment.yaml", "w") as f:
        yaml.dump(exp_yaml, f, sort_keys=False)

def main():
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    
    train_config_path = workspace / "configs" / "train.yaml"
    data_config_path = workspace / "configs" / "dataset_config.yaml"
    manifest_path = workspace / "dataset_manifest.yaml"
    
    if not manifest_path.exists():
        print("Dataset manifest not found. Run create_dataset_manifest.py first.")
        sys.exit(1)
        
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = workspace / "outputs" / f"train_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reproducible experiment file
    create_experiment_yaml(workspace, out_dir, train_config, data_config)
    
    # Copy dataset manifest
    shutil.copy(manifest_path, out_dir / "dataset_manifest.yaml")
    
    print(f"Starting RF-DETR training in {out_dir}")
    print("Hyperparameters:")
    print(yaml.dump(train_config, default_flow_style=False))
    
    # IMPORTANT: Enforce strict preprocessing
    # The image_size from config MUST be passed to the model initialization
    # so that RF-DETR library uses its official internal preprocessing pipeline natively.
    image_size = train_config.get("image_size", 640)
    
    # NOTE: Assuming standard rfdetr API
    if 'RFDETRSmall' in globals():
        model = RFDETRSmall() # or load from specific pretrain_weights if resuming
        
        # In a real environment, you'd map the train_config to the specific arguments 
        # required by the rfdetr `train` method (like learning rate, batch size, etc.)
        
        model.train(
            dataset_dir="d:/Silvertouch Internship/task3/final_training_v2",
            epochs=train_config.get("epochs"),
            batch_size=train_config.get("batch_size"),
            lr=train_config.get("learning_rate"),
            imgsz=image_size,
            project=str(workspace / "outputs"),
            name=out_dir.name,
            seed=train_config.get("random_seed")
        )
        
        print(f"Training started and outputs will be saved to {out_dir}")
    else:
        print("Skipping actual training (rfdetr module missing).")
        
if __name__ == "__main__":
    main()

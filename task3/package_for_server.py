import os
import shutil
from pathlib import Path
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # Create the relative path for the zip archive
            arcname = os.path.relpath(file_path, start=path)
            ziph.write(file_path, arcname)

def main():
    base_dir = Path(r"d:\Silvertouch Internship\task3")
    staging_dir = base_dir / "server_training_package_v2"
    
    # 1. Clean up old staging if it exists
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    
    # 2. Copy dataset
    src_dataset = base_dir / "rfdetr_training_pipeline" / "dataset"
    dst_dataset = staging_dir / "dataset"
    if src_dataset.exists():
        print(f"Copying dataset from {src_dataset} to {dst_dataset}...")
        shutil.copytree(src_dataset, dst_dataset)
    else:
        print(f"Dataset not found at {src_dataset}!")
        return

    # 3. Copy rfdetr_training_pipeline
    src_pipeline = base_dir / "rfdetr_training_pipeline"
    dst_pipeline = staging_dir / "rfdetr_training_pipeline"
    if src_pipeline.exists():
        print(f"Copying pipeline from {src_pipeline} to {dst_pipeline}...")
        shutil.copytree(src_pipeline, dst_pipeline)
    else:
        print(f"Pipeline not found at {src_pipeline}!")
        return
        
    # 4. Copy requirements (if it exists)
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, staging_dir / "requirements.txt")
        
    # 5. Provide a simple start script for the server
    start_script_content = """import subprocess
import sys
from pathlib import Path

def main():
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the train script
    train_script = Path("rfdetr_training_pipeline") / "scripts" / "train.py"
    
    print(f"Starting training via {train_script}...")
    subprocess.run([sys.executable, str(train_script)])

if __name__ == "__main__":
    main()
"""
    with open(staging_dir / "start_server_training.py", "w", encoding="utf-8") as f:
        f.write(start_script_content)
        
    # Modify train.py to point to the new dataset location if necessary
    # Assuming rfdetr_training_pipeline/scripts/train.py uses workspace.parent / "final_training_v2"
    train_py_path = dst_pipeline / "scripts" / "train.py"
    if train_py_path.exists():
        with open(train_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the hardcoded path with the dataset directory in our package
        content = content.replace('workspace.parent / "final_training_v2"', 'workspace.parent / "dataset"')
        
        with open(train_py_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # 6. Zip everything up
    zip_path = base_dir / "server_training_package_v2.zip"
    if zip_path.exists():
        os.remove(zip_path)
        
    print(f"Zipping the package into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(staging_dir, zipf)
        
    print("Package built successfully!")

if __name__ == "__main__":
    main()

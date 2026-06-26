import os
import sys
from pathlib import Path

def verify_ontology(dataset_dir, output_file):
    print("Verifying ontology...")
    splits = ['train', 'valid', 'test']
    
    class_counts = {i: 0 for i in range(8)}
    
    for split in splits:
        lbl_dir = os.path.join(dataset_dir, split, 'labels')
        if not os.path.exists(lbl_dir):
            continue
            
        for lbl in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        cid = int(parts[0])
                        if cid in class_counts:
                            class_counts[cid] += 1
                    except: pass
                    
    missing = [cid for cid, count in class_counts.items() if count == 0]
    
    # Generate ontology_verification.md
    names = {
        0: 'Artwork', 1: 'not_artwork', 2: 'Front_Panel', 3: 'Side_Panel', 
        4: 'Back_Panel', 5: 'Extra_Areas', 6: 'Bottom_Panel', 7: 'Top_Panel'
    }
    
    report = "============================\nONTOLOGY VERIFICATION\n============================\n\n"
    for cid in range(8):
        report += f"Class {cid} {names[cid]}\nCount: {class_counts[cid]}\n\n"
        
    with open(output_file, 'w') as f:
        f.write(report)
        
    print(f"Ontology verification generated at {output_file}")
    
    if missing:
        print(f"CRITICAL ERROR: Missing classes in merged dataset: {missing}")
        sys.exit(1)
        
    print("Ontology verification PASSED. All 8 classes are present.")

if __name__ == '__main__':
    pipeline_dir = Path(__file__).resolve().parent.parent
    dataset_dir = pipeline_dir / "dataset"
    doc_dir = pipeline_dir / "documentation"
    os.makedirs(doc_dir, exist_ok=True)
    
    verify_ontology(dataset_dir, doc_dir / "ontology_verification.md")

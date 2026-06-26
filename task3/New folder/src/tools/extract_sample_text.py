import os
import random
from paddleocr import PaddleOCR

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "front_back_clean")
    output_file = os.path.join(input_dir, "ocr_sample_texts.txt")
    
    front_dir = os.path.join(input_dir, "Front_Panel")
    back_dir = os.path.join(input_dir, "Back_Panel")
    
    front_files = [os.path.join(front_dir, f) for f in os.listdir(front_dir) if f.endswith(('.jpg', '.png'))]
    back_files = [os.path.join(back_dir, f) for f in os.listdir(back_dir) if f.endswith(('.jpg', '.png'))]
    
    sample_front = random.sample(front_files, min(10, len(front_files)))
    sample_back = random.sample(back_files, min(10, len(back_files)))
    
    all_samples = sample_front + sample_back
    
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OCR Extraction Samples (10 Front, 10 Back)\n")
        f.write("="*50 + "\n\n")
        
        for idx, filepath in enumerate(all_samples):
            filename = os.path.basename(filepath)
            print(f"Processing {idx+1}/{len(all_samples)}: {filename}")
            
            result = ocr.ocr(filepath)
            
            detected_texts = []
            if result and isinstance(result, list) and len(result) > 0:
                res_dict = result[0]
                if 'rec_texts' in res_dict:
                    detected_texts = res_dict['rec_texts']
                    
            full_text = "\n".join(detected_texts)
            
            f.write(f"Filename: {filename}\n")
            f.write(f"Folder: {os.path.basename(os.path.dirname(filepath))}\n")
            f.write("Extracted Text:\n")
            f.write("-" * 20 + "\n")
            f.write(full_text if full_text.strip() else "[NO TEXT DETECTED]")
            f.write("\n")
            f.write("="*50 + "\n\n")
            
    print(f"Extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()

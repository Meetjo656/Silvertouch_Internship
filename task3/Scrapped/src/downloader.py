import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from pathlib import Path

from .utils import (
    setup_logger, IMAGES_DIR, IMAGES_CSV, DUPLICATES_CSV, 
    DEFAULT_HEADERS, MAX_WORKERS, MAX_RETRIES, REQUEST_DELAY
)
from .filters import ImageFilter

logger = setup_logger("downloader")

class ImageDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.image_filter = ImageFilter()
        
        # Initialize CSVs if they don't exist
        if not IMAGES_CSV.exists():
            pd.DataFrame(columns=["setid", "page_url", "image_url", "image_type", "local_filename", "status", "hash"]).to_csv(IMAGES_CSV, index=False)
        if not DUPLICATES_CSV.exists():
            pd.DataFrame(columns=["setid", "image_url", "local_filename", "reason"]).to_csv(DUPLICATES_CSV, index=False)
            
    def download_image(self, image_meta: Dict[str, str], index: int) -> Tuple[Dict, str]:
        """
        Downloads a single image and applies filtering.
        Returns the updated metadata and status message.
        """
        url = image_meta["image_url"]
        setid = image_meta["setid"]
        
        # Extract extension
        ext = url.split('.')[-1].split('?')[0]
        if len(ext) > 4 or not ext.isalnum():
            ext = "jpg" # default
            
        filename = f"{setid}_image_{index:03d}.{ext}"
        local_path = IMAGES_DIR / filename
        image_meta["local_filename"] = filename
        
        if local_path.exists():
            # If we resume, we still need to validate/hash to populate duplicate filter state
            is_valid, result = self.image_filter.validate_and_hash_image(str(local_path))
            if is_valid:
                image_meta["status"] = "Success (Resumed)"
                image_meta["hash"] = result
                return image_meta, "Resumed"
            else:
                image_meta["status"] = "Rejected (Resumed)"
                self._log_duplicate(image_meta, result)
                # Cleanup invalid resumed file
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                return image_meta, f"Rejected: {result}"

        # Download with retry
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, stream=True, timeout=15)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                            
                    # Validate after download
                    is_valid, result = self.image_filter.validate_and_hash_image(str(local_path))
                    
                    if is_valid:
                        image_meta["status"] = "Success"
                        image_meta["hash"] = result
                        return image_meta, "Downloaded"
                    else:
                        image_meta["status"] = "Rejected"
                        self._log_duplicate(image_meta, result)
                        # Delete rejected image
                        os.remove(local_path)
                        return image_meta, f"Rejected: {result}"
                        
                elif response.status_code == 404:
                    image_meta["status"] = "Failed (404)"
                    return image_meta, "404 Not Found"
                    
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                
            time.sleep(REQUEST_DELAY * (2 ** (attempt - 1)))
            
        image_meta["status"] = "Failed"
        return image_meta, "Max retries reached"

    def _log_duplicate(self, meta: Dict, reason: str):
        record = pd.DataFrame([{
            "setid": meta.get("setid", ""),
            "image_url": meta.get("image_url", ""),
            "local_filename": meta.get("local_filename", ""),
            "reason": reason
        }])
        record.to_csv(DUPLICATES_CSV, mode='a', header=False, index=False)

    def process_batch(self, images_metadata: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Downloads a batch of images concurrently.
        Returns a dictionary with download statistics.
        """
        stats = {
            "downloaded": 0,
            "duplicate": 0,
            "rejected": 0,
            "failed": 0
        }
        
        # First filter by URL to save bandwidth
        valid_urls_meta = []
        for meta in images_metadata:
            if self.image_filter.is_valid_url(meta["image_url"]):
                valid_urls_meta.append(meta)
            else:
                meta["status"] = "Rejected (URL)"
                meta["local_filename"] = ""
                meta["hash"] = ""
                self._append_to_csv([meta])
                stats["rejected"] += 1
                
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_meta = {
                executor.submit(self.download_image, meta, idx): meta 
                for idx, meta in enumerate(valid_urls_meta, 1)
            }
            
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    updated_meta, msg = future.result()
                    results.append(updated_meta)
                    logger.info(f"{meta['image_url']} -> {msg}")
                    
                    status = updated_meta.get("status", "")
                    if status.startswith("Success"):
                        stats["downloaded"] += 1
                    elif "Duplicate" in msg or "hash:" in msg:
                        stats["duplicate"] += 1
                        stats["rejected"] += 1
                    elif status.startswith("Rejected"):
                        stats["rejected"] += 1
                    elif status.startswith("Failed"):
                        stats["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {meta['image_url']}: {e}")
                    meta["status"] = f"Error: {e}"
                    results.append(meta)
                    stats["failed"] += 1
                    
        self._append_to_csv(results)
        return stats
        
    def _append_to_csv(self, records: List[Dict]):
        if not records:
            return
            
        # Ensure all columns exist
        df = pd.DataFrame(records)
        for col in ["setid", "page_url", "image_url", "image_type", "local_filename", "status", "hash"]:
            if col not in df.columns:
                df[col] = ""
                
        df = df[["setid", "page_url", "image_url", "image_type", "local_filename", "status", "hash"]]
        df.to_csv(IMAGES_CSV, mode='a', header=False, index=False)

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.crawler import DailyMedCrawler
from src.parser import DailyMedParser
from src.downloader import ImageDownloader
from src.utils import setup_logger, BASE_URL, IMAGES_CSV, DUPLICATES_CSV, TARGET_IMAGE_COUNT

logger = setup_logger("main")

def print_summary(labels_visited, total_found, total_downloaded, total_rejected, total_duplicates, total_failed):
    print("\n" + "="*40)
    print("SCRAPING COMPLETE SUMMARY")
    print("="*40)
    print(f"Total Labels Visited   : {labels_visited}")
    print(f"Total Images Found     : {total_found}")
    print(f"Successfully Downloaded: {total_downloaded}")
    print(f"Rejected (Size/URL)    : {total_rejected}")
    print(f"Duplicates Detected    : {total_duplicates}")
    print(f"Failed Downloads       : {total_failed}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="DailyMed Pharmaceutical Packaging Artwork Scraper")
    parser.add_argument("--target", type=int, default=TARGET_IMAGE_COUNT, help="Target number of images to download")
    args = parser.parse_args()
    
    target_count = args.target
    logger.info(f"Starting continuous scraping pipeline. Target images: {target_count}")
    
    crawler = DailyMedCrawler()
    html_parser = DailyMedParser()
    downloader = ImageDownloader()
    
    total_labels_visited = 0
    total_images_found = 0
    
    # Existing stats if resuming
    total_images_downloaded = 0
    total_duplicates_removed = 0
    total_rejected = 0
    total_failed = 0
    
    if IMAGES_CSV.exists():
        try:
            df = pd.read_csv(IMAGES_CSV)
            total_images_downloaded = len(df[df["status"].str.startswith("Success")])
            logger.info(f"Resuming... Already have {total_images_downloaded} downloaded images.")
        except Exception:
            pass
            
    api_page = 1
    
    pbar = tqdm(total=target_count, desc="Downloading Artwork", initial=total_images_downloaded)
    
    try:
        while total_images_downloaded < target_count:
            # Fetch a batch of SETIDs from the REST API
            logger.info(f"Fetching bulk SETIDs from API page {api_page}...")
            setids = crawler.fetch_spls(page=api_page)
            
            if not setids:
                logger.error("No more SETIDs found or API failed. Stopping.")
                break
                
            for setid in setids:
                if total_images_downloaded >= target_count:
                    break
                    
                total_labels_visited += 1
                page_url = f"{BASE_URL}?setid={setid}"
                
                # Fetch Label HTML
                html_content = crawler.fetch_page(page_url)
                if not html_content:
                    continue
                    
                # Parse Images
                images_meta = html_parser.parse_images(html_content, setid, page_url)
                if not images_meta:
                    continue
                    
                total_images_found += len(images_meta)
                logger.info(f"Found {len(images_meta)} image URLs for SETID {setid}")
                
                # Download Batch
                stats = downloader.process_batch(images_meta)
                
                downloaded_in_batch = stats.get("downloaded", 0)
                total_images_downloaded += downloaded_in_batch
                total_duplicates_removed += stats.get("duplicate", 0)
                total_rejected += stats.get("rejected", 0)
                total_failed += stats.get("failed", 0)
                
                pbar.update(downloaded_in_batch)
                
            api_page += 1
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        pbar.close()
        
    print_summary(
        total_labels_visited, 
        total_images_found, 
        total_images_downloaded, 
        total_rejected, 
        total_duplicates_removed, 
        total_failed
    )
    logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()

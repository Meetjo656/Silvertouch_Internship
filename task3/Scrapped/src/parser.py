from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin
from .utils import setup_logger, IMAGE_BASE_URL

logger = setup_logger("parser")

class DailyMedParser:
    def __init__(self):
        pass
        
    def parse_images(self, html_content: str, setid: str, page_url: str) -> List[Dict[str, str]]:
        """
        Parses HTML and extracts image metadata.
        Returns a list of dicts with: setid, page_url, image_url, image_type.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        images_found = []
        
        # We can identify different types of images based on classes or context
        
        # 1. Package Photos (under <div class="mod drug-photos">)
        package_photos = soup.select(".drug-photos img")
        for img in package_photos:
            src = img.get("src")
            if src:
                images_found.append({
                    "setid": setid,
                    "page_url": page_url,
                    "image_url": self._normalize_url(src),
                    "image_type": "Package Photo"
                })
                
        # 2. Section Images (inside <div class="Section">)
        # These are usually Principal Display Panels or artwork inside the label text
        section_images = soup.select(".Section img")
        for img in section_images:
            src = img.get("src")
            if src:
                alt_text = img.get("alt", "").lower()
                
                # Skip chemical structures based on alt text
                if any(word in alt_text for word in ["structure", "formula", "chemical", "molecule"]):
                    continue
                
                # Try to guess type based on alt text or just default to Section Image
                img_type = "Section Image"
                if "front" in alt_text or "principal" in alt_text:
                    img_type = "Principal Display Panel"
                elif "carton" in alt_text:
                    img_type = "Carton Artwork"
                elif "package" in alt_text:
                    img_type = "Package Artwork"
                    
                images_found.append({
                    "setid": setid,
                    "page_url": page_url,
                    "image_url": self._normalize_url(src),
                    "image_type": img_type
                })
                
        # Remove direct duplicates found on the same page
        unique_images = {img["image_url"]: img for img in images_found}.values()
        
        return list(unique_images)
        
    def _normalize_url(self, src: str) -> str:
        """
        Ensures the image URL is absolute.
        """
        if src.startswith("http"):
            return src
        return urljoin(IMAGE_BASE_URL, src)

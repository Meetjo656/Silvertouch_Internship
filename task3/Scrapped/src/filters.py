import os
from PIL import Image
import imagehash
from typing import Set, Tuple
from pathlib import Path
from .utils import setup_logger, MIN_WIDTH, MIN_HEIGHT

logger = setup_logger("filters")

class ImageFilter:
    def __init__(self):
        # Known irrelevent image paths/patterns from DailyMed
        self.banned_substrings = [
            "/images/logo",
            "nih-toplogo",
            "fda_logo",
            "spacer.gif",
            "icon",
            "button",
            "structure",
            "formula",
            "chemical",
            "molecule"
        ]
        # In-memory storage of hashes for duplicate detection
        self.seen_hashes: Set[str] = set()
        
    def is_valid_url(self, image_url: str) -> bool:
        """
        Quick check to discard obvious logos and icons without downloading.
        """
        url_lower = image_url.lower()
        # Remove query parameters for extension check
        base_url_lower = url_lower.split('?')[0]
        
        # In DailyMed, image.cfm is the script that serves images, so we also allow it
        if not (base_url_lower.endswith(('.jpg', '.jpeg', '.png', '.gif')) or "image.cfm" in url_lower):
            return False
            
        for banned in self.banned_substrings:
            if banned in url_lower:
                return False
                
        return True

    def validate_and_hash_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Validates image dimensions and checks for duplicates.
        Returns (is_valid: bool, reason_if_invalid_or_hash_if_valid: str)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    return False, f"Too small ({width}x{height})"
                    
                # Compute perceptual hash
                # We use phash for better resistance against slight scaling/compression
                img_hash = str(imagehash.phash(img))
                
                if img_hash in self.seen_hashes:
                    return False, f"Duplicate (hash: {img_hash})"
                    
                self.seen_hashes.add(img_hash)
                return True, img_hash
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return False, f"Corrupted or invalid image: {e}"

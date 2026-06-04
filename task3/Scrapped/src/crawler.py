import time
import requests
from requests.exceptions import RequestException
from typing import Optional
from .utils import setup_logger, DEFAULT_HEADERS, MAX_RETRIES, REQUEST_DELAY, API_BASE_URL

logger = setup_logger("crawler")

class DailyMedCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        
    def fetch_spls(self, page: int = 1, pagesize: int = 100) -> list:
        """
        Fetches a page of SPLs from the DailyMed REST API and returns a list of SETIDs.
        """
        url = f"{API_BASE_URL}?page={page}&pagesize={pagesize}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Fetching API page {page} (Attempt {attempt}/{MAX_RETRIES})")
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    setids = [item.get("setid") for item in data.get("data", []) if item.get("setid")]
                    time.sleep(REQUEST_DELAY)
                    return setids
                else:
                    logger.error(f"Failed to fetch API {url}. Status: {response.status_code}")
            except RequestException as e:
                logger.error(f"API request failed: {e}")
                
            time.sleep(REQUEST_DELAY * (2 ** (attempt - 1)))
            
        return []
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetches the HTML content of a URL with retry logic.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Fetching URL: {url} (Attempt {attempt}/{MAX_RETRIES})")
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    time.sleep(REQUEST_DELAY) # Responsible crawling
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"Page not found: {url}")
                    return None
                else:
                    logger.error(f"Failed to fetch {url}. Status code: {response.status_code}")
                    
            except RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
            
            # Exponential backoff for retries
            time.sleep(REQUEST_DELAY * (2 ** (attempt - 1)))
            
        logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts.")
        return None

import os
import time
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

def scrape_shutterstock(query_url, output_dir, target_count=150):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Initializing undetected-chromedriver...")
    options = uc.ChromeOptions()
    # Adding arguments to help avoid detection
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--no-sandbox')
    
    try:
        driver = uc.Chrome(options=options)
    except Exception as e:
        print(f"Failed to start Chrome via undetected_chromedriver. Ensure Chrome is installed. Error: {e}")
        return

    downloaded_count = 0
    page = 1
    downloaded_urls = set()

    try:
        while downloaded_count < target_count:
            url = f"{query_url}&page={page}" if "?" in query_url else f"{query_url}?page={page}"
            print(f"Navigating to page {page}: {url}")
            driver.get(url)
            time.sleep(5)  # Wait for cloudflare/initial load
            
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            
            while True:
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(1.5)
                
                try:
                    images = driver.find_elements(By.TAG_NAME, "img")
                    for img in images:
                        if downloaded_count >= target_count:
                            break
                            
                        try:
                            src = img.get_attribute('src')
                        except StaleElementReferenceException:
                            continue
                            
                        if not src:
                            continue
                            
                        if "shutterstock" in src and src.endswith(".jpg") and src not in downloaded_urls:
                            print(f"Downloading image {downloaded_count + 1}/{target_count}")
                            try:
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
                                }
                                response = requests.get(src, headers=headers, timeout=10)
                                if response.status_code == 200:
                                    filename = f"pharma_label_{downloaded_count + 1:03d}.jpg"
                                    filepath = os.path.join(output_dir, filename)
                                    
                                    with open(filepath, 'wb') as f:
                                        f.write(response.content)
                                        
                                    downloaded_urls.add(src)
                                    downloaded_count += 1
                            except Exception as e:
                                print(f"Failed to download {src}: {e}")
                except Exception as e:
                    print(f"Error while finding images: {e}")
                
                if downloaded_count >= target_count:
                    break
                    
                new_height = driver.execute_script("return document.body.scrollHeight")
                current_scroll = driver.execute_script("return window.pageYOffset + window.innerHeight")
                if current_scroll >= new_height:
                    scroll_attempts += 1
                    if scroll_attempts >= 3:
                        break
                    time.sleep(2)
                else:
                    scroll_attempts = 0
                        
            page += 1

    finally:
        driver.quit()

if __name__ == "__main__":
    url = "https://www.shutterstock.com/search/pharma-label"
    output_dir = r"d:\Silvertouch Internship\task3\Scrapped\output\new_scrapped"
    print(f"Starting scraping process for: {url}")
    print(f"Output directory: {output_dir}")
    scrape_shutterstock(url, output_dir, target_count=150)
    print(f"\nSuccessfully finished! Downloaded {target_count} images.")

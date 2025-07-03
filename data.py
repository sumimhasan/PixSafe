import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def download_image(url, folder, idx):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            if len(response.content) < 1024:
                print(f"🚫 Skipped (too small <1KB): {url}")
                return False

            ext = os.path.splitext(url)[1].split("?")[0]
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = ".jpg"
            file_path = os.path.join(folder, f"img{idx}{ext}")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved img{idx}{ext}")
            return True
        else:
            print(f"⚠️ Bad status: {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def scrape_images(search_query, num_imgs, save_folder):
    create_folder(save_folder)

    # ✅ Load your Firefox profile where you're logged into Gmail
    profile_path = "/home/joy/.mozilla/firefox/fnyox7kg.default-esr"
    profile = FirefoxProfile(profile_path)

    options = Options()
    options.headless = False  # Show browser window

    # ✅ Launch Firefox with the logged-in profile
    driver = webdriver.Firefox(firefox_profile=profile, options=options)
    driver.get(f"https://www.google.com/search?q={search_query}&tbm=isch")
    time.sleep(3)

    scrolls = 0
    imgs_saved = 0
    seen_urls = set()

    while imgs_saved < num_imgs and scrolls < 15:  # Max scroll limit as backup
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
        print(f"🔍 Found {len(image_elements)} images after scroll #{scrolls + 1}")

        for img in image_elements:
            if imgs_saved >= num_imgs:
                break
            try:
                url = img.get_attribute("src")
                if url and url.startswith("http") and url not in seen_urls:
                    seen_urls.add(url)
                    if download_image(url, save_folder, imgs_saved + 1):
                        imgs_saved += 1
            except Exception as e:
                print(f"⚠️ Skipping: {e}")
                continue

        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(2)
        scrolls += 1

    driver.quit()
    print(f"\n🎉 Done. Successfully saved {imgs_saved} images in '{save_folder}'.")
    
if __name__ == "__main__":
    scrape_images("query", 100, "folder")
"""
ANPC Press Release Scraper (Selenium)

A minimal Selenium-based scraper for educational/comparison purposes.
For production use, prefer the Playwright version (anpc_scraper.py).

This demonstrates the same scraping logic using Selenium instead of Playwright.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Try to import the parser from the main scraper to reuse logic
try:
    from scrapers.anpc_scraper import parse_article_html
except ImportError:
    # Fallback if imported from outside the package
    from anpc_scraper import parse_article_html

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_URL = "https://anpc.ro"
INDEX_URL_TEMPLATE = "https://anpc.ro/category/comunicate-de-presa/bpage/{page}/"
REQUEST_DELAY_SECONDS = 1.0

# Locators
ARTICLE_ITEM_XPATH = "//div[@class='brz-posts__item']"
# We'll use these as fallbacks if parse_article_html is not available
CONTENT_XPATH = "//div[contains(@class, 'brz-wp-post-content')]"
DATE_XPATH = "//div[contains(@class, 'brz-wp__postinfo')]//li[contains(@class, 'brz-li')][2]"
TIME_XPATH = "//div[contains(@class, 'brz-wp__postinfo')]//li[contains(@class, 'brz-li')][3]"

# Output
DATA_DIR = Path(__file__).parent.parent / "data" / "selenium"
PROCESSED_DIR = DATA_DIR / "processed"


def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (compatible; NLP-Course-Bot/1.0)")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_index_page(driver: webdriver.Chrome, page_num: int) -> list[str]:
    """Scrape article URLs from an index page."""
    url = INDEX_URL_TEMPLATE.format(page=page_num)
    logger.info(f"Scraping index page {page_num}: {url}")
    
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, ARTICLE_ITEM_XPATH))
        )
    except TimeoutException:
        logger.warning(f"Timeout waiting for articles on page {page_num}")
        return []
    
    article_urls = []
    items = driver.find_elements(By.XPATH, ARTICLE_ITEM_XPATH)
    for item in items:
        try:
            # Find all links in the item and take the last one (usually the one on the title/image)
            links = item.find_elements(By.TAG_NAME, "a")
            if links:
                href = links[-1].get_attribute("href")
                if href:
                    article_urls.append(href)
        except NoSuchElementException:
            continue
    
    logger.info(f"Found {len(article_urls)} articles on page {page_num}")
    return article_urls


def scrape_article(driver: webdriver.Chrome, url: str) -> Optional[dict]:
    """Scrape a single article page."""
    logger.info(f"Scraping article: {url}")
    
    try:
        driver.get(url)
        # Wait for some content to be present
        WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.TAG_NAME, "body").text.strip() != ""
        )
    except TimeoutException:
        logger.warning(f"Timeout loading {url}")
        return None
    
    # Use the BeautifulSoup parser we already refined!
    html_content = driver.page_source
    data = parse_article_html(html_content, url)
    
    return {
        **data,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def scrape_articles(max_pages: int = 1, max_articles: Optional[int] = None) -> list[dict]:
    """
    Simple scraping function for demonstration.
    
    For production use, the Playwright version has better:
    - Async support
    - Retry logic
    - Deduplication
    - Raw HTML saving
    """
    driver = create_driver()
    articles = []
    
    try:
        for page_num in range(1, max_pages + 1):
            urls = scrape_index_page(driver, page_num)
            if max_articles:
                urls = urls[:max_articles]
            
            for url in urls:
                article = scrape_article(driver, url)
                if article:
                    articles.append(article)
                time.sleep(REQUEST_DELAY_SECONDS)
            
            time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        driver.quit()
    
    # Save results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "articles_selenium.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(articles)} articles to {output_file}")
    return articles


if __name__ == "__main__":
    logger.info("Starting ANPC scraper (Selenium version)...")
    articles = scrape_articles(max_pages=1, max_articles=2)
    logger.info(f"Finished scraping {len(articles)} articles")

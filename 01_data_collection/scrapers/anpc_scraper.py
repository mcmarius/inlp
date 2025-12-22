"""
ANPC Press Release Scraper (Playwright)

Scrapes press releases from the Romanian consumer protection authority (ANPC).
Website: https://anpc.ro/category/comunicate-de-presa/

This scraper uses Playwright for rendering dynamic JavaScript content.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_URL = "https://anpc.ro"
INDEX_URL_TEMPLATE = "https://anpc.ro/category/comunicate-de-presa/bpage/{page}/"
TOTAL_PAGES = 12

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0
# NOTE: For production scrapers, consider using random delays to avoid detection:
# import random
# delay = random.uniform(0.5, 2.0)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
# Alternative retry strategies (commented for reference):
# - Skip and log: Simply log the error and continue to next item
# - Store partial data: Save what we have and mark as incomplete

# Output directories
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
PROCESSED_DIR = DATA_DIR / "processed"
ARTICLES_JSON = PROCESSED_DIR / "articles.json"


def get_slug_from_url(url: str) -> str:
    """Extract a slug from a URL for use as a filename."""
    path = urlparse(url).path.strip("/")
    # Take the last path segment
    slug = path.split("/")[-1] if path else "index"
    # Clean up the slug
    slug = re.sub(r"[^\w\-]", "_", slug)
    return slug[:100]  # Limit length


def load_existing_articles() -> dict[str, dict]:
    """
    Load existing articles from JSON file.
    
    Returns a dict mapping URL -> article data for easy deduplication.
    
    NOTE: Deduplication strategies:
    - URL-based (current): Skip if URL already scraped. Fast but won't catch content changes.
    - Content-based: Re-scrape and compare content hash. Catches updates but slower.
    - For sites with changing content, consider re-scraping periodically.
    
    NOTE: For near-duplicate detection (e.g., articles with minor edits),
    consider using Locality Sensitive Hashing (LSH) with MinHash signatures.
    This is useful when building training datasets to avoid data leakage.
    """
    if ARTICLES_JSON.exists():
        with open(ARTICLES_JSON, "r", encoding="utf-8") as f:
            articles = json.load(f)
            return {article["url"]: article for article in articles}
    return {}


def save_articles(articles: dict[str, dict]) -> None:
    """Save articles to JSON file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    articles_list = list(articles.values())
    with open(ARTICLES_JSON, "w", encoding="utf-8") as f:
        json.dump(articles_list, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(articles_list)} articles to {ARTICLES_JSON}")


def save_raw_html(url: str, html_content: str) -> Path:
    """Save raw HTML content to file."""
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    slug = get_slug_from_url(url)
    filepath = RAW_HTML_DIR / f"{slug}.html"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filepath


async def retry_with_backoff(
    func,
    *args,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    **kwargs
):
    """
    Execute a function with exponential backoff retry logic.
    
    Alternative strategies (implement as needed):
    - Linear backoff: delay = initial_backoff * attempt
    - Jittered backoff: delay = backoff * (1 + random.uniform(-0.1, 0.1))
    - Circuit breaker: Stop all requests after N consecutive failures
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (PlaywrightTimeout, Exception) as e:
            last_exception = e
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")
    raise last_exception


async def scrape_index_page(page: Page, page_num: int) -> list[str]:
    """
    Scrape article URLs from an index page.
    
    Args:
        page: Playwright page object
        page_num: Page number (1-12)
        
    Returns:
        List of article URLs found on the page
    """
    url = INDEX_URL_TEMPLATE.format(page=page_num)
    logger.info(f"Scraping index page {page_num}: {url}")
    
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(5000)
    
    # Extract article links using the XPath pattern
    article_urls = []
    n = 1
    while True:
        try:
            # await page.get_by_text(re.compile("Program")).click()
            # await page.get_by_text(re.compile("Refuză")).click()
            # links = await page.get_by_role("link").filter(has_text="Citeste mai mult").all()
            links = await page.locator("//div[@class='brz-posts__item']").all()

            links = [(await link.get_by_role('link').all())[-1] for link in links]
            article_urls = [await link.get_attribute('href') for link in links]
            if article_urls:
                break
            if not article_urls and n > 2:
                logger.info(f"No articles found after {n} attempts")
                break
            n += 1

        except Exception as e:
            logger.error(f"Error scraping index page {page_num}: {e}")
            break
    
    # await page.wait_for_timeout(5000)
    logger.info(f"Found {len(article_urls)} articles on page {page_num}")
    return article_urls


def parse_article_html(html_content: str, url: str) -> dict:
    """
    Extract article data from HTML content using BeautifulSoup.
    
    This function decouples extraction from Playwright, allowing it to run 
    on saved raw HTML files.
    """
    soup = BeautifulSoup(html_content, "lxml")
    
    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    if not title:
        # Fallback to h1 or itemprop="name"
        name_elem = soup.find(attrs={"itemprop": "name"})
        if name_elem:
            title = name_elem.get_text(strip=True)
        else:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
                
    if title:
        # Clean up title
        title = title.split(" – ")[0].split(" | ")[0].strip()
    
    # Extract content
    # We prefer class-based selectors as they are more robust than nth-child paths
    content_text = ""
    content_div = soup.find("div", class_="brz-wp-post-content")
    
    if not content_div:
        # Fallback to the rigid path provided by user if class is missing
        content_div = soup.select_one("body > div:nth-of-type(8) > section:nth-of-type(3) > div > div > div:nth-of-type(3) > div > div")
        
    if not content_div:
        # Broader fallback
        content_div = soup.find("div", class_="brz-rich-text") or soup.find("div", class_="entry-content")
    
    if content_div:
        # Extract text from paragraphs if they exist
        paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            content_text = "\n\n".join(paragraphs)
        else:
            # Fallback to direct text if no paragraphs are found
            content_text = content_div.get_text(strip=True, separator="\n\n")

    # Extract date and time from post info
    date_text = ""
    time_text = ""
    
    # Try finding the post info container
    post_info = soup.find("div", class_="brz-wp__postinfo")
    if post_info:
        li_items = post_info.find_all("li", class_="brz-li")
        # Structure is usually: Author, Date, Time, Comments
        if len(li_items) >= 2:
            date_text = li_items[1].get_text(strip=True)
        if len(li_items) >= 3:
            time_text = li_items[2].get_text(strip=True)
            
    # Fallback to rigid paths if class-based search fails
    if not date_text:
        date_elem = soup.select_one("body > div:nth-of-type(8) > section:nth-of-type(3) > div > div > div:nth-of-type(5) > div > div > div > ul > li:nth-of-type(2)")
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            
    if not time_text:
        time_elem = soup.select_one("body > div:nth-of-type(8) > section:nth-of-type(3) > div > div > div:nth-of-type(5) > div > div > div > ul > li:nth-of-type(3)")
        if time_elem:
            time_text = time_elem.get_text(strip=True)
                
    return {
        "url": url,
        "title": title,
        "content": content_text.strip(),
        "date": date_text,
        "time": time_text,
    }


async def scrape_article(page: Page, url: str) -> Optional[dict]:
    """
    Scrape a single article page.
    
    Args:
        page: Playwright page object
        url: Article URL
        
    Returns:
        Dict with article data or None if scraping failed
    """
    logger.info(f"Scraping article: {url}")
    
    try:
        await page.goto(url, wait_until="networkidle")
    except PlaywrightTimeout:
        logger.warning(f"Timeout loading {url}")
        return None
    
    # Save raw HTML
    html_content = await page.content()
    raw_html_path = save_raw_html(url, html_content)
    
    # Extract data from HTML
    data = parse_article_html(html_content, url)
    
    return {
        **data,
        "raw_html_path": str(raw_html_path),
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


async def scrape_articles(
    max_pages: Optional[int] = None,
    max_articles_per_page: Optional[int] = None,
) -> list[dict]:
    """
    Scrape all articles from ANPC press releases.
    
    Args:
        max_pages: Maximum number of index pages to scrape (default: all 12)
        max_articles_per_page: Maximum articles per page (default: all)
        
    Returns:
        List of scraped article dictionaries
    """
    pages_to_scrape = max_pages or TOTAL_PAGES
    
    # Load existing articles for deduplication
    existing_articles = load_existing_articles()
    logger.info(f"Found {len(existing_articles)} existing articles")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (compatible; NLP-Course-Bot/1.0; Educational)"
        )
        page = await context.new_page()
        
        try:
            # Collect all article URLs
            all_urls = []
            for page_num in range(1, pages_to_scrape + 1):
                urls = await retry_with_backoff(scrape_index_page, page, page_num)
                if max_articles_per_page:
                    urls = urls[:max_articles_per_page]
                all_urls.extend(urls)
                
                # Rate limiting between index pages
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
            
            # Filter out already-scraped URLs
            new_urls = [url for url in all_urls if url not in existing_articles]
            logger.info(
                f"Found {len(all_urls)} total articles, "
                f"{len(new_urls)} new to scrape"
            )
            
            # Scrape each new article
            for url in new_urls:
                article = await retry_with_backoff(scrape_article, page, url)
                if article:
                    existing_articles[url] = article
                    # Save after each article (in case of interruption)
                    save_articles(existing_articles)
                
                # Rate limiting between articles
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                
        finally:
            await browser.close()
    
    return list(existing_articles.values())


def reprocess_saved_html() -> list[dict]:
    """
    Re-extract information from all saved raw HTML files.
    
    Useful if we want to extract other information from the raw data
    without re-scraping the web.
    """
    if not ARTICLES_JSON.exists():
        logger.warning("No articles JSON found. Reprocessing might be limited to raw files.")
        existing_articles = {}
    else:
        existing_articles = load_existing_articles()
        
    if not RAW_HTML_DIR.exists():
        logger.error(f"Raw HTML directory not found: {RAW_HTML_DIR}")
        return []
    
    # We need a mapping from slug to URL if we want to update existing entries
    # Or we can just iterate over the JSON entries that have `raw_html_path`
    
    reprocessed_count = 0
    # Map raw html file back to URL using existing data if possible
    path_to_url = {a["raw_html_path"]: a["url"] for a in existing_articles.values() if "raw_html_path" in a}
    
    for html_file in RAW_HTML_DIR.glob("*.html"):
        path_str = str(html_file)
        url = path_to_url.get(path_str)
        
        # If we don't have the URL in existing data, we might not be able to fully reprocess 
        # unless we find another way to recover the URL (maybe from the HTML itself?)
        if not url:
            # Fallback: simple heuristic or skip
            continue
            
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            data = parse_article_html(content, url)
            
            # Update the entry
            if url in existing_articles:
                existing_articles[url].update(data)
                reprocessed_count += 1
        except Exception as e:
            logger.error(f"Error reprocessing {html_file}: {e}")
            
    if reprocessed_count > 0:
        save_articles(existing_articles)
        logger.info(f"Reprocessed {reprocessed_count} articles from raw HTML.")
    
    return list(existing_articles.values())


def scrape_page(page_num: int, max_articles: Optional[int] = None) -> list[dict]:
    """
    Synchronous wrapper to scrape a single index page.
    
    Useful for testing or quick scrapes from the command line.
    """
    try:
        # Check if there is an event loop already running (Jupyter/IPython)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    async def _scrape():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                urls = await scrape_index_page(page, page_num)
                logger.info(f"Urls: {urls}")
                if max_articles:
                    urls = urls[:max_articles]
                
                articles = []
                for url in urls:
                    article = await scrape_article(page, url)
                    if article:
                        articles.append(article)
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)
                
                return articles
            finally:
                await browser.close()

    if loop and loop.is_running():
        return loop.create_task(_scrape())
    else:
        return asyncio.run(_scrape())


if __name__ == "__main__":
    # Run the full scraper
    logger.info("Starting ANPC scraper...")
    articles = asyncio.run(scrape_articles())
    logger.info(f"Finished scraping {len(articles)} articles")

import pytest
import asyncio
from playwright.async_api import async_playwright
from scrapers import anpc_scraper

# Constants for E2E tests
# Article to find on the first index page
EXPECTED_TOP_ARTICLE_URL = "https://anpc.ro/comandament-anpc-in-zonele-turistice/"
# Specific article page for content extraction test
SPECIFIC_ARTICLE_URL = "https://anpc.ro/comandament-anpc-in-zonele-turistice/"

@pytest.mark.asyncio
async def test_index_page_e2e():
    """
    End-to-end test for the index page.
    Verifies that we can find articles on the first page.
    Note: The hard-coded URL check will fail as new articles are posted.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Scrape the first page
        urls = await anpc_scraper.scrape_index_page(page, page_num=1)
        
        await browser.close()
        
        assert len(urls) > 0, "No article URLs found on index page 1"
        
        # Check for the expected top article (with helpful error message if it moved)
        if EXPECTED_TOP_ARTICLE_URL not in urls:
            found_urls_str = "\n".join(urls[:5])
            pytest.fail(
                f"Expected article '{EXPECTED_TOP_ARTICLE_URL}' not found in top results.\n"
                f"Top 5 URLs found:\n{found_urls_str}\n"
                f"Please update EXPECTED_TOP_ARTICLE_URL in tests/test_anpc_scraper_e2e.py if this is expected."
            )

@pytest.mark.asyncio
async def test_article_content_e2e():
    """
    End-to-end test for a specific article's content.
    Makes a real request to verify the extraction logic still works.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Wait for the actual content to load (handles refresh redirects)
        try:
            await page.goto(SPECIFIC_ARTICLE_URL, wait_until="networkidle")
            # Wait for either the content or the title span
            await page.wait_for_selector("div.brz-wp-post-content, span[itemprop='name']", timeout=15000)
        except Exception as e:
            print(f"Error waiting for article content: {e}")
            
        html_content = await page.content()
        
        data = anpc_scraper.parse_article_html(html_content, SPECIFIC_ARTICLE_URL)
        
        await browser.close()
        
        assert data["url"] == SPECIFIC_ARTICLE_URL
        assert "Comandament ANPC" in data["title"]
        assert len(data["content"]) > 100
        assert data["date"] != ""
        assert data["time"] != ""
        
        # Specific check for the known article content to ensure extraction quality
        assert "Autoritatea Națională pentru Protecția Consumatorilor" in data["content"]
        assert "1,6 milioane de lei" in data["content"]

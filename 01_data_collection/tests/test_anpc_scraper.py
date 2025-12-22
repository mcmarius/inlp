# Unit tests for the ANPC scraper.

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Import the module under test
from scrapers import anpc_scraper


class TestGetSlugFromUrl:
    """Tests for URL slug extraction."""
    
    def test_simple_url(self):
        url = "https://anpc.ro/controale-anpc-amenzi/"
        assert anpc_scraper.get_slug_from_url(url) == "controale-anpc-amenzi"
    
    def test_url_with_trailing_slash(self):
        url = "https://anpc.ro/some-article/"
        assert anpc_scraper.get_slug_from_url(url) == "some-article"
    
    def test_url_without_path(self):
        url = "https://anpc.ro/"
        assert anpc_scraper.get_slug_from_url(url) == "index"
    
    def test_long_slug_is_truncated(self):
        url = "https://anpc.ro/" + "a" * 150 + "/"
        slug = anpc_scraper.get_slug_from_url(url)
        assert len(slug) <= 100
    
    def test_special_characters_replaced(self):
        url = "https://anpc.ro/article?id=123&foo=bar"
        slug = anpc_scraper.get_slug_from_url(url)
        assert "?" not in slug
        assert "&" not in slug


class TestLoadExistingArticles:
    """Tests for loading existing articles."""
    
    def test_returns_empty_dict_when_no_file(self, tmp_path):
        with patch.object(anpc_scraper, "ARTICLES_JSON", tmp_path / "nonexistent.json"):
            result = anpc_scraper.load_existing_articles()
            assert result == {}
    
    def test_loads_existing_articles(self, tmp_path):
        articles_file = tmp_path / "articles.json"
        articles = [
            {"url": "https://example.com/1", "title": "Article 1"},
            {"url": "https://example.com/2", "title": "Article 2"},
        ]
        articles_file.write_text(json.dumps(articles), encoding="utf-8")
        
        with patch.object(anpc_scraper, "ARTICLES_JSON", articles_file):
            result = anpc_scraper.load_existing_articles()
            assert len(result) == 2
            assert "https://example.com/1" in result
            assert result["https://example.com/1"]["title"] == "Article 1"


class TestSaveArticles:
    """Tests for saving articles."""
    
    def test_saves_articles_to_json(self, tmp_path):
        output_file = tmp_path / "processed" / "articles.json"
        articles = {
            "https://example.com/1": {"url": "https://example.com/1", "title": "Test"},
        }
        
        with patch.object(anpc_scraper, "ARTICLES_JSON", output_file):
            with patch.object(anpc_scraper, "PROCESSED_DIR", tmp_path / "processed"):
                anpc_scraper.save_articles(articles)
        
        assert output_file.exists()
        loaded = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test"


class TestSaveRawHtml:
    """Tests for saving raw HTML."""
    
    def test_saves_html_file(self, tmp_path):
        html_dir = tmp_path / "raw_html"
        
        with patch.object(anpc_scraper, "RAW_HTML_DIR", html_dir):
            path = anpc_scraper.save_raw_html(
                "https://anpc.ro/test-article/",
                "<html><body>Test</body></html>"
            )
        
        assert path.exists()
        assert path.name == "test-article.html"
        assert path.read_text(encoding="utf-8") == "<html><body>Test</body></html>"


class TestRetryWithBackoff:
    """Tests for retry logic."""
    
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        mock_func = AsyncMock(return_value="success")
        result = await anpc_scraper.retry_with_backoff(mock_func)
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        mock_func = AsyncMock(side_effect=[Exception("fail"), "success"])
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await anpc_scraper.retry_with_backoff(
                mock_func, max_retries=2, initial_backoff=0.1
            )
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        mock_func = AsyncMock(side_effect=Exception("always fails"))
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="always fails"):
                await anpc_scraper.retry_with_backoff(
                    mock_func, max_retries=3, initial_backoff=0.1
                )
        
        assert mock_func.call_count == 3


class TestScrapeIndexPage:
    """Tests for index page scraping."""
    
    @pytest.mark.asyncio
    async def test_extracts_article_urls(self):
        # Create mock page
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        
        # Mock locator for "//div[@class='brz-posts__item']"
        mock_item1 = AsyncMock()
        mock_item2 = AsyncMock()
        
        mock_items_locator = AsyncMock()
        mock_items_locator.all = AsyncMock(return_value=[mock_item1, mock_item2])
        
        # Mock get_by_role('link') for each item
        mock_link1 = AsyncMock()
        mock_link1.get_attribute = AsyncMock(return_value="https://anpc.ro/article-1/")
        
        mock_link2 = AsyncMock()
        mock_link2.get_attribute = AsyncMock(return_value="https://anpc.ro/article-2/")
        
        # .get_by_role('link').all() returns a list of links
        mock_links_locator1 = AsyncMock()
        mock_links_locator1.all = AsyncMock(return_value=[mock_link1])
        
        mock_links_locator2 = AsyncMock()
        mock_links_locator2.all = AsyncMock(return_value=[mock_link2])
        
        mock_item1.get_by_role = MagicMock(return_value=mock_links_locator1)
        mock_item2.get_by_role = MagicMock(return_value=mock_links_locator2)
        
        # Set up the main page locator
        def mock_locator(selector):
            if selector == "//div[@class='brz-posts__item']":
                return mock_items_locator
            return AsyncMock()
            
        mock_page.locator = MagicMock(side_effect=mock_locator)
        
        urls = await anpc_scraper.scrape_index_page(mock_page, 1)
        
        assert len(urls) == 2
        assert "https://anpc.ro/article-1/" in urls
        assert "https://anpc.ro/article-2/" in urls


class TestParseArticleHtml:
    """Tests for HTML parsing using BeautifulSoup."""
    
    def test_parses_full_article(self):
        html = """
        <html>
            <head><title>Test Title – ANPC</title></head>
            <body>
                <div class="brz-wp-post-content">
                    <p>Paragraph 1</p>
                    <p>Paragraph 2</p>
                </div>
                <div class="brz-wp__postinfo">
                    <ul>
                        <li class="brz-li">admin</li>
                        <li class="brz-li">22 Decembrie 2024</li>
                        <li class="brz-li">14:30</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        data = anpc_scraper.parse_article_html(html, "https://example.com/test")
        
        assert data["title"] == "Test Title"
        assert data["content"] == "Paragraph 1\n\nParagraph 2"
        assert data["date"] == "22 Decembrie 2024"
        assert data["time"] == "14:30"
        assert data["url"] == "https://example.com/test"

    def test_fallback_selectors(self):
        # Test the fallback to nth-of-type selectors if classes are missing
        html = """
        <html>
            <head><title>Fallback Test</title></head>
            <body>
                <div>1</div><div>2</div><div>3</div><div>4</div>
                <div>5</div><div>6</div><div>7</div>
                <div id="container"> <!-- body > div:nth-of-type(8) -->
                    <section>1</section><section>2</section>
                    <section> <!-- section:nth-of-type(3) -->
                        <div>
                            <div>
                                <div>spacer</div> <!-- div:nth-of-type(1) -->
                                <div>spacer</div> <!-- div:nth-of-type(2) -->
                                <div> <!-- div:nth-of-type(3) -->
                                    <div>
                                        <div><p>Content via path</p></div>
                                    </div>
                                </div>
                                <div>spacer</div> <!-- div:nth-of-type(4) -->
                                <div> <!-- div:nth-of-type(5) -->
                                    <div>
                                        <div>
                                            <div>
                                                <ul>
                                                    <li>author</li>
                                                    <li>Date via path</li>
                                                    <li>Time via path</li>
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>
            </body>
        </html>
        """
        data = anpc_scraper.parse_article_html(html, "https://example.com/fallback")
        assert "Content via path" in data["content"]
        assert data["date"] == "Date via path"
        assert data["time"] == "Time via path"


class TestScrapeArticle:
    """Tests for article scraping integration."""
    
    @pytest.mark.asyncio
    async def test_integration(self, tmp_path):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html><head><title>Mock</title></head><body><p>Mock Content</p></body></html>")
        
        with patch.object(anpc_scraper, "RAW_HTML_DIR", tmp_path / "raw_html"):
            article = await anpc_scraper.scrape_article(
                mock_page, "https://anpc.ro/mock-article/"
            )
        
        assert article is not None
        assert article["url"] == "https://anpc.ro/mock-article/"
        assert article["scraped_at"] is not None
        assert "raw_html_path" in article
        assert (tmp_path / "raw_html" / "mock-article.html").exists()


class TestDeduplication:
    """Tests for deduplication behavior."""
    
    @pytest.mark.asyncio
    async def test_skips_existing_urls(self, tmp_path):
        """Verify that already-scraped URLs are skipped."""
        # Create existing articles file
        existing_articles = [
            {"url": "https://anpc.ro/existing/", "title": "Existing"}
        ]
        articles_file = tmp_path / "articles.json"
        articles_file.write_text(json.dumps(existing_articles), encoding="utf-8")
        
        with patch.object(anpc_scraper, "ARTICLES_JSON", articles_file):
            existing = anpc_scraper.load_existing_articles()
        
        # Simulate filtering
        all_urls = ["https://anpc.ro/existing/", "https://anpc.ro/new/"]
        new_urls = [url for url in all_urls if url not in existing]
        
        assert len(new_urls) == 1
        assert "https://anpc.ro/new/" in new_urls
        assert "https://anpc.ro/existing/" not in new_urls

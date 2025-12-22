"""
Scrapers for building NLP datasets.

This package contains web scrapers for collecting text data from various sources.
"""

from .anpc_scraper import scrape_articles, scrape_article, scrape_index_page

__all__ = ["scrape_articles", "scrape_article", "scrape_index_page"]

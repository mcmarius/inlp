import pytest
import pandas as pd
from tasks.search_engine import SearchEngine
from tasks.recommender import ArticleRecommender
from tasks.keyword_extraction import KeywordExtractor

@pytest.fixture
def sample_df():
    data = {
        'title': ['Mere Rosii', 'Pere Galbene', 'Fructe de sezon'],
        'date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'url': ['url1', 'url2', 'url3'],
        'content_tokens': [
            ['mere', 'rosii', 'proaspete'],
            ['pere', 'galbene', 'dulci'],
            ['fructe', 'sezon', 'mere', 'pere']
        ],
        'lemmatized_content': [
            'măr roşu proaspăt',
            'păr galben dulce',
            'fruct sezon măr păr'
        ]
    }
    return pd.DataFrame(data)

def test_search_engine(sample_df):
    engine = SearchEngine(sample_df)
    results = engine.search("mere", n=2)
    assert len(results) == 2
    assert 'Mere Rosii' in results['title'].values
    assert results.iloc[0]['relevance_score'] > 0

def test_recommender(sample_df):
    recommender = ArticleRecommender(sample_df)
    recommendations = recommender.get_recommendations(0, n=1)
    assert len(recommendations) == 1
    # 'Mere Rosii' should be similar to 'Fructe de sezon' because both have 'măr'
    assert recommendations.iloc[0]['title'] == 'Fructe de sezon'
    assert recommendations.iloc[0]['similarity_score'] > 0

def test_keyword_extraction(sample_df):
    extractor = KeywordExtractor(sample_df)
    keywords = extractor.get_keywords(0, n=2)
    assert len(keywords) == 2
    assert 'roşu' in keywords or 'roşu' in [w.lower() for w in keywords]
    
    df_with_keywords = extractor.add_keywords_to_df(n=2)
    assert 'keywords' in df_with_keywords.columns
    assert len(df_with_keywords['keywords'].iloc[0].split(',')) == 2

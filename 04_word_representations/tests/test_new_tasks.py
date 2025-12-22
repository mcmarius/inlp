import pytest
import pandas as pd
from tasks.authorship import AuthorshipIdentifier
from tasks.seasonal_analysis import SeasonalAnalyzer

@pytest.fixture
def sample_df():
    data = {
        'title': ['Control Iarna', 'Control Vara', 'Control Primavara', 'Control Toamna'],
        'date_iso': ['2025-01-01', '2025-07-01', '2025-04-01', '2025-10-01'],
        'lemmatized_content': [
            'zăpadă gheață și și și iarnă',
            'mărțișor floare și și și primăvară',
            'soare mare pentru pentru pentru vară',
            'școală rechizite pentru pentru pentru toamnă'
        ]
    }
    return pd.DataFrame(data)

def test_authorship_identifier(sample_df):
    docs = sample_df['lemmatized_content'].tolist()
    identifier = AuthorshipIdentifier(n_clusters=2)
    labels = identifier.cluster_articles(docs)
    assert len(labels) == 4
    assert len(set(labels)) == 2
    
    markers = identifier.get_stylistic_markers(n_words=3)
    assert len(markers) == 2
    assert len(markers[0]) == 3

def test_seasonal_analyzer(sample_df):
    analyzer = SeasonalAnalyzer(sample_df)
    assert 'season' in analyzer.df.columns
    assert analyzer.df.iloc[0]['season'] == 'Winter'
    assert analyzer.df.iloc[1]['season'] == 'Summer'
    
    keywords = analyzer.get_seasonal_keywords(n_words=2)
    assert 'Winter' in keywords
    assert len(keywords['Winter']) > 0

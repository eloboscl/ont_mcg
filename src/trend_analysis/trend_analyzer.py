# src/trend_analysis/trend_analyzer.py

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

def analyze_trends(documents: Dict[str, Dict[str, Any]], mc_terms: List[str]) -> Dict[str, Any]:
    """
    Analyze trends in management control terms over time.
    
    :param documents: Dictionary of analyzed documents
    :param mc_terms: List of management control terms to analyze
    :return: Dictionary containing trend analysis results
    """
    logger.info("Starting trend analysis...")
    
    # Convert documents to DataFrame for easier manipulation
    df = pd.DataFrame.from_dict(documents, orient='index')
    print(df.head())
    df['year'] = pd.to_numeric(df['metadata'].apply(lambda x: x['year']), errors='coerce')
    df = df.dropna(subset=['year'])
    
    # Analyze term frequency over time
    term_trends = analyze_term_frequency(df, mc_terms)
    
    # Analyze co-occurrence of terms
    co_occurrence = analyze_co_occurrence(df, mc_terms)
    
    # Analyze sentiment trends
    sentiment_trends = analyze_sentiment_trends(df, mc_terms)
    
    # Perform trend forecasting
    forecasts = forecast_trends(term_trends)
    
    return {
        "term_trends": term_trends,
        "co_occurrence": co_occurrence,
        "sentiment_trends": sentiment_trends,
        "forecasts": forecasts
    }

def analyze_term_frequency(df: pd.DataFrame, mc_terms: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze the frequency of management control terms over time."""
    term_trends = {}
    for term in mc_terms:
        yearly_counts = df.groupby('year').apply(lambda x: sum(term.lower() in doc.lower() for doc in x['content']))
        yearly_counts = yearly_counts.reset_index()
        yearly_counts.columns = ['year', 'count']
        yearly_counts['relative_frequency'] = yearly_counts['count'] / df.groupby('year').size()
        term_trends[term] = yearly_counts.to_dict('records')
    return term_trends

def analyze_co_occurrence(df: pd.DataFrame, mc_terms: List[str]) -> Dict[str, Dict[str, int]]:
    """Analyze co-occurrence of management control terms."""
    co_occurrence = {term: Counter() for term in mc_terms}
    for _, doc in df.iterrows():
        present_terms = [term for term in mc_terms if term.lower() in doc['content'].lower()]
        for term in present_terms:
            co_occurrence[term].update(present_terms)
    return {term: dict(counter) for term, counter in co_occurrence.items()}

def analyze_sentiment_trends(df: pd.DataFrame, mc_terms: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze sentiment trends for management control terms."""

    df['sentiment'] = df['nlp_analysis'].apply(lambda x: x['sentiment'])

    sentiment_trends = {}
    for term in mc_terms:
        term_docs = df[df['content'].str.contains(term, case=False)]
        yearly_sentiment = term_docs.groupby('year')['sentiment'].mean()
        yearly_sentiment = yearly_sentiment.reset_index()
        yearly_sentiment.columns = ['year', 'average_sentiment']
        sentiment_trends[term] = yearly_sentiment.to_dict('records')
    return sentiment_trends

def forecast_trends(term_trends: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Forecast future trends based on historical data."""
    forecasts = {}
    for term, trend in term_trends.items():
        years = [item['year'] for item in trend]
        frequencies = [item['relative_frequency'] for item in trend]
        if len(years) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(years, frequencies)
            forecasts[term] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "forecast_next_5_years": [slope * (max(years) + i) + intercept for i in range(1, 6)]
            }
    return forecasts
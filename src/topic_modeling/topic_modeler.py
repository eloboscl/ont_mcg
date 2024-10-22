# src/topic_modeling/topic_modeler.py

import logging
import multiprocessing
from typing import Dict, List, Tuple

import nltk
import numpy as np
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config.settings import CUSTOM_STOP_WORDS

logger = logging.getLogger(__name__)

def get_stop_words() -> set:
    """Get combined set of standard and custom stopwords."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(CUSTOM_STOP_WORDS)
    return stop_words

def preprocess_text(text: str, stop_words: set = None) -> List[str]:
    """Tokenize and clean text."""
    if stop_words is None:
        stop_words = get_stop_words()
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and short tokens
    tokens = [token for token in tokens 
             if token not in stop_words 
             and len(token) > 2 
             and token.isalnum()]
    
    return tokens

def prepare_texts_for_lda(documents: Dict[str, Dict]) -> List[List[str]]:
    """Prepare texts for LDA by converting them to lists of tokens."""
    processed_texts = []
    stop_words = get_stop_words()
    
    for doc in documents.values():
        if 'content' in doc:
            tokens = preprocess_text(doc['content'], stop_words)
            if tokens:  # Only add if we have tokens
                processed_texts.append(tokens)
    
    return processed_texts

def perform_topic_modeling(documents: Dict[str, Dict], 
                         num_topics: int = 10, 
                         workers: int = None) -> Tuple[LdaMulticore, float]:
    """
    Perform topic modeling on the documents using LDA.
    
    Args:
        documents: Dictionary of documents with their content
        num_topics: Number of topics to extract
        workers: Number of worker processes for parallel processing
    
    Returns:
        tuple: (LDA model, coherence score)
    """
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 4)
    
    logger.info("Preparing texts for topic modeling...")
    texts = prepare_texts_for_lda(documents)
    
    if not texts:
        logger.error("No valid texts found for topic modeling")
        return None, 0.0
    
    logger.info("Creating dictionary...")
    dictionary = corpora.Dictionary(texts)
    
    # Filter out extreme frequencies
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    logger.info("Creating corpus...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    logger.info(f"Training LDA model with {num_topics} topics...")
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=17,
        passes=10,
        workers=workers
    )
    
    logger.info("Computing coherence score...")
    coherence_model = CoherenceModel(
        model=lda_model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    logger.info(f"Topic modeling complete. Coherence score: {coherence_score}")
    
    return lda_model, coherence_score

def print_topics(lda_model: LdaMulticore, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    """Print and return the topics with their top words."""
    topics = {}
    for idx, topic in lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
        topics[idx] = topic
        words = [w for w, _ in topic]
        logger.info(f"Topic {idx}: {', '.join(words)}")
    return topics
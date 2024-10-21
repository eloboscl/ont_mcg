import logging
from typing import Dict, List, Tuple

import spacy
from textblob import TextBlob

logger = logging.getLogger(__name__)

# Load SpaCy model
nlp = spacy.load("en_core_web_lg") #en_core_web_sm (not too good results)

def perform_ner(text: str) -> List[Tuple[str, str]]:
    """
    Perform Named Entity Recognition on the given text.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def perform_sentiment_analysis(text: str) -> float:
    """
    Perform sentiment analysis on the given text.
    Returns a polarity score between -1 (negative) and 1 (positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    doc = nlp(text)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return sorted(set(noun_chunks), key=noun_chunks.count, reverse=True)[:top_n]

def analyze_document(doc: Dict[str, any]) -> Dict[str, any]:
    """
    Perform NER and sentiment analysis on a single document.
    """
    content = doc['content']
    entities = perform_ner(content)
    sentiment = perform_sentiment_analysis(content)
    key_phrases = extract_key_phrases(content)
    
    entity_counts = {}
    for _, label, _ in entities:
        entity_counts[label] = entity_counts.get(label, 0) + 1
    
    return {
        'entities': entities,
        'entity_counts': entity_counts,
        'sentiment': sentiment,
        'key_phrases': key_phrases
    }

def process_documents(documents: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, any]]:
    """
    Process all documents with NER and sentiment analysis.
    """
    analyzed_documents = {}
    for key, doc in documents.items():
        logger.info(f"Analyzing document: {key}")
        analysis = analyze_document(doc)
        analyzed_documents[key] = {**doc, 'nlp_analysis': analysis}
    return analyzed_documents

if __name__ == "__main__":
    # This allows you to test the module independently
    sample_text = "Apple Inc. is planning to open a new store in New York City next month. The CEO, Tim Cook, expressed excitement about the expansion."
    entities = perform_ner(sample_text)
    sentiment = perform_sentiment_analysis(sample_text)
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment}")
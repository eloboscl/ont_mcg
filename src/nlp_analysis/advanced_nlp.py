import logging
from typing import Dict, List, Tuple

import spacy
import torch
from textblob import TextBlob
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer)

logger = logging.getLogger(__name__)

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Load BERT models for NER and sentiment analysis
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def perform_ner(text: str, device: torch.device) -> List[Tuple[str, str]]:
    """
    Perform Named Entity Recognition on the given text using BERT.
    """
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = ner_model(**inputs).to(device)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    for token, prediction in zip(tokens, predictions[0]):
        if prediction != 0:  # 0 is the index for 'O' (Outside) in NER
            entity_label = ner_model.config.id2label[prediction.item()]
            entities.append((token, entity_label))
    return entities

def perform_sentiment_analysis(text: str, device: torch.device) -> float:
    """
    Perform sentiment analysis on the given text using BERT.
    Returns a polarity score between 0 (very negative) and 4 (very positive).
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = sentiment_model(**inputs).to(device)
    return torch.argmax(outputs.logits).item()

def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    doc = nlp(text)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return sorted(set(noun_chunks), key=noun_chunks.count, reverse=True)[:top_n]

def analyze_document(doc: Dict[str, any], device: torch.device) -> Dict[str, any]:
    """
    Perform NER and sentiment analysis on a single document.
    """
    content = doc['content']
    entities = perform_ner(content, device)
    sentiment = perform_sentiment_analysis(content, device)
    key_phrases = extract_key_phrases(content)
   
    entity_counts = {}
    for _, label in entities:
        entity_counts[label] = entity_counts.get(label, 0) + 1
   
    return {
        'entities': entities,
        'entity_counts': entity_counts,
        'sentiment': sentiment,
        'key_phrases': key_phrases
    }

def process_documents(documents: Dict[str, Dict[str, any]], device: torch.device) -> Dict[str, Dict[str, any]]:
    """
    Process all documents with NER and sentiment analysis.
    """
    analyzed_documents = {}
    for key, doc in documents.items():
        logger.info(f"Analyzing document: {key}")
        analysis = analyze_document(doc, device)
        analyzed_documents[key] = {**doc, 'nlp_analysis': analysis}
    return analyzed_documents

if __name__ == "__main__":
    # This allows you to test the module independently
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    sentiment_model.to(device)
    
    sample_text = "Apple Inc. is planning to open a new store in New York City next month. The CEO, Tim Cook, expressed excitement about the expansion."
    entities = perform_ner(sample_text, device)
    sentiment = perform_sentiment_analysis(sample_text, device)
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment}")
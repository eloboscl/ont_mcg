import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple

import spacy
import torch
from textblob import TextBlob
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer)

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str = None) -> logging.Logger:
    """
    Configure logging with proper Unicode handling for both console and file output.
    
    Args:
        output_dir: Optional directory for log files. If None, uses current directory.
    """
    # Create logger
    logger = logging.getLogger('ont_mcg')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler (if output_dir is provided)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'ont_mcg_{timestamp}.log')
        
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create log file: {str(e)}")
    
    return logger

def safe_unicode_logger(logger: logging.Logger, level: int, message: str):
    """
    Safely log Unicode messages with fallback to ASCII if needed.
    
    Args:
        logger: Logger instance
        level: Logging level (e.g., logging.INFO)
        message: Message to log
    """
    try:
        logger.log(level, message)
    except UnicodeEncodeError:
        # Fallback to ASCII representation if Unicode fails
        ascii_message = message.encode('ascii', 'replace').decode('ascii')
        logger.log(level, ascii_message)

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
    ner_model.to(device)  # Ensure the model is on the correct device
    with torch.no_grad():
        outputs = ner_model(**inputs)
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
    Perform sentiment analysis using a BERT model.
    Returns a sentiment score between 1 and 5.
    """
    # Move model to specified device
    sentiment_model.to(device)
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(scores, dim=1).item() + 1  # Add 1 because model outputs 0-4
        
    # Normalize to [-1, 1] range
    normalized_score = (prediction - 3) / 2  # Convert 1-5 to [-1, 1]
    
    return normalized_score

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
import logging
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config.settings import CUSTOM_STOP_WORDS

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

def get_stop_words() -> set:
    """Get combined set of standard and custom stopwords."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(CUSTOM_STOP_WORDS)
    return stop_words


class TextCleaner:
    def __init__(self, custom_stopwords=None):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        

    def clean_text(self, text):
        """
        Clean and preprocess the input text.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)

        return cleaned_text

    def process_documents(self, documents):
        """
        Process a dictionary of documents.
        """
        cleaned_documents = {}
        for key, text in documents.items():
            cleaned_documents[key] = self.clean_text(text)
            logger.info(f"Cleaned document: {key}")
        return cleaned_documents

def main(input_file, output_file):
    import json

    logger.info("Starting text cleaning process")

    # Load documents
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Initialize cleaner
    cleaner = TextCleaner()

    # Clean documents
    cleaned_documents = cleaner.process_documents(documents)

    # Save cleaned documents
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_documents, f, ensure_ascii=False, indent=4)

    logger.info(f"Cleaning complete. Cleaned texts saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python cleaner.py <input_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
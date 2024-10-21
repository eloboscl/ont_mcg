import datetime
import json
import logging
import logging.config
import os

import nltk
import numpy as np

from config import settings
from config.settings import METADATA_FILE, PDF_DIR
from src.data_ingestion import metadata_integrator, pdf_processor
from src.nlp_analysis import advanced_nlp
from src.text_processing import cleaner

logger = logging.getLogger(__name__)

def setup_run_folder():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    run_folder = os.path.join("output", timestamp)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

def setup_logging(run_folder):
    log_file = os.path.join(run_folder, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def download_nltk_data():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
    
    # Explicitly download punkt_tab
    try:
        nltk.download('punkt_tab', quiet=True)
        logger.info("Successfully downloaded NLTK resource: punkt_tab")
    except Exception as e:
        logger.error(f"Failed to download NLTK resource punkt_tab: {str(e)}")
        logger.info("Please manually download NLTK data using the following commands:")
        logger.info(">>> import nltk")
        logger.info(">>> nltk.download('punkt_tab')")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Set up logging
    logging.config.dictConfig(settings.LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    logger.info("Starting Management Control Analysis")
    logger.info(f"Using input directory: {settings.INPUT_DIR}")
    logger.info(f"Output will be saved to: {settings.OUTPUT_DIR}")
    logger.info(f"Maximum CPU usage: {settings.MAX_CPU_PERCENT}%")
    logger.info(f"Maximum memory usage: {settings.MAX_MEMORY_PERCENT}%")
    logger.info(f"Maximum GPU usage: {settings.MAX_GPU_PERCENT}%")

    run_folder = setup_run_folder()
    setup_logging(run_folder)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting new run in folder: {run_folder}")
    
    # Ensure NLTK data is downloaded
    download_nltk_data()
    
    # Process PDFs
    extracted_texts, relevance_scores, mc_terms_found, failed_files = pdf_processor.process_pdfs_in_batches(run_folder)
    
    logger.info(f"Processed {len(extracted_texts)} PDFs successfully")
    logger.info(f"Failed to process {len(failed_files)} PDFs")
    
    # Clean extracted texts
    cleaner_instance = cleaner.TextCleaner()
    cleaned_texts = cleaner_instance.process_documents(extracted_texts)
    cleaned_texts_file = os.path.join(run_folder, "cleaned_texts.json")
    with open(cleaned_texts_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_texts, f, ensure_ascii=False, indent=4)
    logger.info(f"Cleaned texts saved to: {cleaned_texts_file}")
    
    # Integrate metadata
    integrator = metadata_integrator.MetadataIntegrator(METADATA_FILE)
    integrated_documents = integrator.integrate_metadata(cleaned_texts)
    integrated_file = os.path.join(run_folder, "integrated_documents.json")
    with open(integrated_file, 'w', encoding='utf-8') as f:
        json.dump(integrated_documents, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"Integrated documents saved to: {integrated_file}")

    # Log some statistics about the integrated documents
    docs_with_metadata = sum(1 for doc in integrated_documents.values() if 'metadata' in doc)
    logger.info(f"Documents with integrated metadata: {docs_with_metadata}")
    logger.info(f"Documents without metadata: {len(integrated_documents) - docs_with_metadata}")

     # Perform advanced NLP analysis
    logger.info("Starting advanced NLP analysis...")
    analyzed_documents = advanced_nlp.process_documents(integrated_documents)
    analyzed_file = os.path.join(run_folder, "analyzed_documents.json")
    with open(analyzed_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_documents, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"NLP analysis completed. Results saved to: {analyzed_file}")

    # Log some statistics about the analyzed documents
    total_entities = sum(len(doc['nlp_analysis']['entities']) for doc in analyzed_documents.values())
    avg_sentiment = sum(doc['nlp_analysis']['sentiment'] for doc in analyzed_documents.values()) / len(analyzed_documents)
    all_key_phrases = [phrase for doc in analyzed_documents.values() for phrase in doc['nlp_analysis']['key_phrases']]
    top_key_phrases = sorted(set(all_key_phrases), key=all_key_phrases.count, reverse=True)[:10]

    logger.info(f"Total entities found: {total_entities}")
    logger.info(f"Average sentiment score: {avg_sentiment:.2f}")
    logger.info(f"Top 10 key phrases across all documents: {', '.join(top_key_phrases)}")

    logger.info("Run completed")

if __name__ == "__main__":
    main()
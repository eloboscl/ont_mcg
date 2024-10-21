import datetime
import json
import logging
import logging.config
import multiprocessing
import os

import nltk
import numpy as np
import torch

from config import settings
from config.settings import (CUSTOM_STOP_WORDS, MANAGEMENT_CONTROL_TERMS,
                             METADATA_FILE, PDF_DIR)
from src.data_ingestion import metadata_integrator, pdf_processor
from src.network_analysis import network_analyzer
from src.nlp_analysis import advanced_nlp
from src.text_processing import cleaner
from src.topic_modeling import topic_modeler
from src.trend_analysis import trend_analyzer

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

def log_summary_statistics(analyzed_documents, lda_model, author_network, trend_results):
    logger.info("Logging summary statistics...")

    # Document statistics
    total_docs = len(analyzed_documents)
    avg_length = np.mean([len(doc['cleaned_text'].split()) for doc in analyzed_documents.values()])
    logger.info(f"Total documents analyzed: {total_docs}")
    logger.info(f"Average document length: {avg_length:.2f} words")

    # Entity statistics
    all_entities = [entity for doc in analyzed_documents.values() for entity in doc['nlp_analysis']['entities']]
    entity_counts = Counter(all_entities)
    top_entities = entity_counts.most_common(10)
    logger.info(f"Top 10 named entities: {top_entities}")

    # Sentiment statistics
    sentiments = [doc['nlp_analysis']['sentiment'] for doc in analyzed_documents.values()]
    avg_sentiment = np.mean(sentiments)
    logger.info(f"Average sentiment score: {avg_sentiment:.2f}")

    # Topic modeling statistics
    logger.info("Top topics:")
    for idx, topic in lda_model.print_topics(-1):
        logger.info(f"Topic {idx}: {topic}")

    # Network analysis statistics
    logger.info(f"Author collaboration network statistics:")
    logger.info(f"  Number of authors: {author_network.number_of_nodes()}")
    logger.info(f"  Number of collaborations: {author_network.number_of_edges()}")
    degree_centrality = nx.degree_centrality(author_network)
    top_authors = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]
    logger.info(f"  Top 5 authors by degree centrality: {top_authors}")

    # Trend analysis statistics
    logger.info("Trend analysis summary:")
    for term, trend in trend_results['term_trends'].items():
        start_freq = trend[0]['relative_frequency']
        end_freq = trend[-1]['relative_frequency']
        change = (end_freq - start_freq) / start_freq * 100
        logger.info(f"  {term}: {change:.2f}% change from {trend[0]['year']} to {trend[-1]['year']}")

    # Forecast summary
    logger.info("Trend forecasts for the next 5 years:")
    for term, forecast in trend_results['forecasts'].items():
        logger.info(f"  {term}: {forecast['forecast_next_5_years']}")


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

    # Set up device for GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Process PDFs
    extracted_texts, relevance_scores, mc_terms_found, failed_files = pdf_processor.process_pdfs_in_batches(run_folder)
    
    logger.info(f"Processed {len(extracted_texts)} PDFs successfully")
    logger.info(f"Failed to process {len(failed_files)} PDFs")
    
    # Clean extracted texts
    cleaner_instance = cleaner.TextCleaner(custom_stopwords=CUSTOM_STOP_WORDS)
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
    analyzed_documents = advanced_nlp.process_documents(integrated_documents, device=device)
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


    # Perform topic modeling
    logger.info("Starting topic modeling...")
    lda_model, coherence = topic_modeler.perform_topic_modeling(
        [doc['cleaned_text'] for doc in analyzed_documents.values()],
        num_topics=10,
        workers=multiprocessing.cpu_count() - 1
    )
    topic_file = os.path.join(run_folder, "topic_model_results.json")
    with open(topic_file, 'w', encoding='utf-8') as f:
        json.dump({
            "topics": lda_model.print_topics(),
            "coherence": coherence
        }, f, ensure_ascii=False, indent=4)
    logger.info(f"Topic modeling completed. Results saved to: {topic_file}")

    # Perform network analysis
    logger.info("Starting network analysis...")
    author_network = network_analyzer.create_author_collaboration_network(analyzed_documents)
    network_file = os.path.join(run_folder, "author_network.graphml")
    network_analyzer.save_network(author_network, network_file)
    logger.info(f"Network analysis completed. Results saved to: {network_file}")

    # Perform trend analysis
    logger.info("Starting trend analysis...")
    trend_results = trend_analyzer.analyze_trends(analyzed_documents, MANAGEMENT_CONTROL_TERMS)
    trend_file = os.path.join(run_folder, "trend_analysis_results.json")
    with open(trend_file, 'w', encoding='utf-8') as f:
        json.dump(trend_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"Trend analysis completed. Results saved to: {trend_file}")

    # Log summary statistics
    log_summary_statistics(analyzed_documents, lda_model, author_network, trend_results)

    logger.info("Run completed")

if __name__ == "__main__":
    main()
import datetime
import json
import logging
import logging.config
import multiprocessing
import os
from collections import Counter

import networkx as nx
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords

from config import settings
from config.settings import (CUSTOM_STOP_WORDS, MANAGEMENT_CONTROL_TERMS,
                             METADATA_FILE, PDF_DIR)
from src.data_ingestion import metadata_integrator, pdf_processor
from src.network_analysis import network_analyzer
from src.nlp_analysis import advanced_nlp
from src.nlp_analysis.advanced_nlp import process_documents
from src.text_processing import cleaner
from src.topic_modeling import topic_modeler
from src.trend_analysis import trend_analyzer
from src.viz.sankey_generator import create_sankey_diagrams
from src.viz.wordcloud_generator import run_wordcloud_analysis

logger = logging.getLogger(__name__)

def get_stop_words() -> set:
    """Get combined set of standard and custom stopwords."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(CUSTOM_STOP_WORDS)
    return stop_words

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

def log_summary_statistics(analyzed_documents, lda_model=None, trend_results=None, author_network=None):
    """
    Log summary statistics for the analysis.
    
    Args:
        analyzed_documents: Dictionary containing analyzed documents
        lda_model: LDA model (can be None if topic modeling failed)
        trend_results: Dictionary containing trend analysis results (can be None)
        author_network: NetworkX graph of author collaborations (can be None)
    """
    logger.info("Logging summary statistics...")
    
    try:
        # [Previous document, entity, and sentiment statistics remain the same]
        
        # Network analysis statistics
        if author_network is not None:
            logger.info("\nAuthor Collaboration Network Statistics:")
            try:
                # Basic network metrics
                num_authors = author_network.number_of_nodes()
                num_collaborations = author_network.number_of_edges()
                logger.info(f"  Number of authors: {num_authors}")
                logger.info(f"  Number of collaborations: {num_collaborations}")

                if num_authors > 0:
                    # Density
                    density = nx.density(author_network)
                    logger.info(f"  Network density: {density:.3f}")

                    # Average degree
                    avg_degree = sum(dict(author_network.degree()).values()) / num_authors
                    logger.info(f"  Average collaborations per author: {avg_degree:.2f}")

                    # Largest component size
                    largest_cc = max(nx.connected_components(author_network), key=len)
                    largest_cc_size = len(largest_cc)
                    logger.info(f"  Largest connected component size: {largest_cc_size} authors "
                              f"({(largest_cc_size/num_authors)*100:.1f}% of network)")

                    # Most connected authors (by degree centrality)
                    degree_cent = nx.degree_centrality(author_network)
                    top_authors = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info("  Top 5 authors by number of collaborations:")
                    for author, centrality in top_authors:
                        num_collabs = author_network.degree(author)
                        logger.info(f"    - {author}: {num_collabs} collaborations "
                                  f"(centrality: {centrality:.3f})")

                    # Try to calculate additional metrics if the network is not too large
                    if num_authors < 1000:  # Avoid expensive computations for very large networks
                        try:
                            # Average clustering coefficient
                            avg_clustering = nx.average_clustering(author_network)
                            logger.info(f"  Average clustering coefficient: {avg_clustering:.3f}")

                            # Average shortest path length (only for largest component to ensure it's connected)
                            largest_cc_subgraph = author_network.subgraph(largest_cc)
                            avg_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
                            logger.info(f"  Average shortest path length: {avg_path_length:.2f}")
                        except Exception as e:
                            logger.debug(f"Could not compute some advanced network metrics: {e}")

                    # Communities
                    try:
                        communities = nx.community.greedy_modularity_communities(author_network)
                        num_communities = len(communities)
                        avg_community_size = np.mean([len(c) for c in communities])
                        logger.info(f"  Number of communities: {num_communities}")
                        logger.info(f"  Average community size: {avg_community_size:.1f} authors")
                    except Exception as e:
                        logger.debug(f"Could not compute community statistics: {e}")

            except Exception as e:
                logger.warning(f"Error computing network statistics: {e}")
        else:
            logger.warning("No author network available - network analysis may have failed")

        # [Previous topic modeling and trend analysis statistics remain the same]
            
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")


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
    cleaner_instance = cleaner.TextCleaner(custom_stopwords= get_stop_words())
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
    analyzed_documents = process_documents(integrated_documents, device='cuda')# advanced_nlp.process_documents(integrated_documents, device=device)
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
    try:
        lda_model, coherence = topic_modeler.perform_topic_modeling(
            analyzed_documents,
            num_topics=10,
            workers=multiprocessing.cpu_count() - 1
        )
        
        if lda_model is not None:
            topics = topic_modeler.print_topics(lda_model)
            # Save topics to file
            topics_file = os.path.join(run_folder, "topic_model_results.json")
            with open(topics_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'topics': topics,
                    'coherence_score': coherence
                }, f, ensure_ascii=False, indent=4)
            logger.info(f"Topic modeling results saved to: {topics_file}")
        else:
            logger.warning("Topic modeling failed to produce a model")
            topics = None
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        lda_model, topics = None, None

#    # Perform network analysis
#    logger.info("Starting network analysis...")
#    author_network = network_analyzer.create_author_collaboration_network(analyzed_documents)
#    network_file = os.path.join(run_folder, "author_network.graphml")
#    network_analyzer.save_network(author_network, network_file)
#    logger.info(f"Network analysis completed. Results saved to: {network_file}")

    # Perform trend analysis
    logger.info("Starting trend analysis...")
    trend_results = trend_analyzer.analyze_trends(analyzed_documents, MANAGEMENT_CONTROL_TERMS)
    trend_file = os.path.join(run_folder, "trend_analysis_results.json")
    with open(trend_file, 'w', encoding='utf-8') as f:
        json.dump(trend_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"Trend analysis completed. Results saved to: {trend_file}")

    # Viz ######################################################################
    try:
        logger.info("Generating wordclouds...")
        run_wordcloud_analysis(analyzed_documents, MANAGEMENT_CONTROL_TERMS, run_folder)
    except Exception as e:
        logger.error(f"Error generating wordclouds: {str(e)}")
 

    # Log summary statistics
    log_summary_statistics(
        analyzed_documents=analyzed_documents,
        lda_model=lda_model if lda_model is not None else None,
        trend_results=trend_results if 'trend_results' in locals() else None,
        author_network=author_network if 'author_network' in locals() else None
    )

    logger.info("Run completed")

if __name__ == "__main__":
    main()
import logging
import logging.config
from config import settings
import os
import datetime
import logging
from src.data_ingestion import pdf_processor, metadata_integrator
from src.text_processing import cleaner


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
    
    extracted_texts, relevance_scores, mc_terms_found, failed_files = pdf_processor.process_pdfs_in_batches(run_folder)
    
    logger.info(f"Processed {len(extracted_texts)} PDFs successfully")
    logger.info(f"Failed to process {len(failed_files)} PDFs")

    logger.info("Run completed")

if __name__ == "__main__":
    main()
import json
import logging
import os
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config.settings import (MANAGEMENT_CONTROL_TERMS, MAX_CPU_PERCENT,
                             PDF_BATCH_SIZE, PDF_DIR)

from ..utils.gpu_utils import get_available_gpu_memory
from ..utils.multiprocessing_utils import get_optimal_process_count

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize BERT model for text relevance scoring
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

def extract_text_and_check_terms(file_path: str, mc_terms: List[str]) -> Tuple[str, List[str]]:
    try:
        with fitz.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        
        # Check for management control terms
            term_frequencies = {}
            for term in mc_terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = len(re.findall(pattern, text.lower()))
                if matches > 0:
                    term_frequencies[term] = matches
            
            found_terms = list(term_frequencies.keys())
            
            return text, found_terms, term_frequencies
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return None, [], {}

def compute_relevance_score(text: str, terms: List[str]) -> float:
    start_time = time.time()
    # Tokenize and encode the text and terms
    text_encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    terms_encoding = tokenizer(terms, truncation=True, padding=True, return_tensors="pt").to(device)
    
    # Compute embeddings
    with torch.no_grad():
        text_embedding = model(**text_encoding).last_hidden_state.mean(dim=1)
        terms_embedding = model(**terms_encoding).last_hidden_state.mean(dim=1)
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(text_embedding, terms_embedding.mean(dim=0), dim=1)
    score = similarity.item()
    
    computation_time = time.time() - start_time
    logger.info(f"Computed relevance score in {computation_time:.2f} seconds")
    return score

def process_pdf(args: Tuple[str, str, List[str]]) -> Dict:
    pdf_path, filename, mc_terms = args
    start_time = time.time()
    
    extracted_text, found_terms, term_frequencies = extract_text_and_check_terms(pdf_path, mc_terms)
    if extracted_text:
        relevance_score = compute_relevance_score(extracted_text, mc_terms)
        processing_time = time.time() - start_time
        # Use filename without extension as the key
        key = os.path.splitext(filename)[0]
        return {
            "key": key,
            "filename": filename,
            "full_text": extracted_text,
            "mc_terms_found": found_terms,
            "relevance_score": relevance_score,
            "processing_time": processing_time,
            "mc_terms_freq": term_frequencies
        }
    return None

def process_pdfs_in_batches(output_folder: str, batch_size: int = PDF_BATCH_SIZE) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, List[str]], List[str]]:
    pdf_output_folder = os.path.join(output_folder, "pdf_processed")
    os.makedirs(pdf_output_folder, exist_ok=True)
    
    all_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    extracted_texts = {}
    relevance_scores = {}
    mc_terms_found = {}
    failed_files = []
    term_frequencies = {}

    num_processes = get_optimal_process_count(MAX_CPU_PERCENT)
    logger.info(f"Using {num_processes} processes for PDF processing")
    
    batch_size = int(batch_size)
    logger.info(f"Processing PDFs in batches of {batch_size}")
    
    total_start_time = time.time()
    batch_times = []

    with tqdm(total=len(all_pdfs), desc="Processing PDFs", ncols=100) as pbar:
        for i in range(0, len(all_pdfs), batch_size):
            batch = all_pdfs[i:i+batch_size]
            pdf_files = [(os.path.join(PDF_DIR, f), f, MANAGEMENT_CONTROL_TERMS) for f in batch]
            
            batch_start_time = time.time()
            batch_results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(process_pdf, args) for args in pdf_files]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        extracted_texts[result['key']] = result['full_text']
                        relevance_scores[result['key']] = result['relevance_score']
                        mc_terms_found[result['key']] = result['mc_terms_found']
                        term_frequencies[result['key']] = result['mc_terms_freq']
                    else:
                        failed_files.append(pdf_files[futures.index(future)][1])
                    pbar.update(1)
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Summarize batch results
            successful = len(batch_results)
            avg_time = statistics.mean([r['processing_time'] for r in batch_results]) if batch_results else 0
            logger.info(f"Batch {i//batch_size + 1}: {successful}/{len(batch)} successful, avg time: {avg_time:.2f}s")

    total_time = time.time() - total_start_time
    avg_batch_time = statistics.mean(batch_times)
    logger.info(f"Total processing time: {total_time:.2f}s, Average batch time: {avg_batch_time:.2f}s")
    logger.info(f"Processed {len(extracted_texts)}/{len(all_pdfs)} files successfully.")
    logger.info(f"Failed to process {len(failed_files)} files.")
    
    # Save results
    save_extracted_texts(extracted_texts, os.path.join(pdf_output_folder, "extracted_texts.json"))
    save_relevance_scores(relevance_scores, os.path.join(pdf_output_folder, "relevance_scores.json"))
    save_mc_terms_found(mc_terms_found, os.path.join(pdf_output_folder, "mc_terms_found.json"))
    save_failed_files(failed_files, os.path.join(pdf_output_folder, "failed_files.txt"))
    
    return extracted_texts, relevance_scores, mc_terms_found, failed_files

def save_extracted_texts(extracted_texts: Dict[str, str], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_texts, f, ensure_ascii=False, indent=4)
    logger.info(f"Extracted texts saved to: {output_file}")

def save_relevance_scores(relevance_scores: Dict[str, float], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(relevance_scores, f, ensure_ascii=False, indent=4)
    logger.info(f"Relevance scores saved to: {output_file}")

def save_mc_terms_found(mc_terms_found: Dict[str, List[str]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mc_terms_found, f, ensure_ascii=False, indent=4)
    logger.info(f"Management control terms found saved to: {output_file}")

def save_failed_files(failed_files: List[str], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in failed_files:
            f.write(f"{file}\n")
    logger.info(f"Failed files list saved to: {output_file}")


def main():
    output_file = os.path.join(OUTPUT_DIR, "extracted_texts.json")
    extracted_texts, failed_files = process_pdfs_in_batches()
    save_extracted_texts(extracted_texts, output_file)
    
    # Save failed files list
    with open(os.path.join(OUTPUT_DIR, "failed_files.txt"), 'w') as f:
        for file in failed_files:
            f.write(f"{file}\n")

if __name__ == "__main__":
    main()
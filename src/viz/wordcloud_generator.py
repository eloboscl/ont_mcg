# src/viz/wordcloud_generator.py

import logging
import os
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from wordcloud import WordCloud

logger = logging.getLogger(__name__) 

def prepare_text_for_wordcloud(documents: Dict[str, Dict], decade: tuple) -> str:
    """Prepare text for wordcloud by extracting cleaned text from the correct decade."""
    start_year, end_year = decade
    combined_text = []
    
    for doc in documents.values():
        try:
            year = int(doc.get('year', 0))
            if start_year <= year <= end_year:
                # Get the cleaned text, defaulting to regular content if cleaned isn't available
                text = doc.get('content', doc.get('content', ''))
                if text:
                    combined_text.append(text)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing document year: {e}")
    
    if not combined_text:
        logger.warning(f"No documents found for decade {start_year}-{end_year}")
        return ""
    
    return " ".join(combined_text)

def create_wordcloud(text: str, mc_terms: List[str]) -> WordCloud:
    """Create wordcloud with emphasis on management control terms."""
    if not text.strip():
        logger.warning("Empty text provided for wordcloud")
        return None
    
    # Split text into words and count frequencies
    words = text.lower().split()
    word_frequencies = Counter(words)
    
    # Increase weight for management control terms
    for term in mc_terms:
        term_lower = term.lower()
        if term_lower in word_frequencies:
            word_frequencies[term_lower] *= 2
    
    # Remove entries with zero frequency
    word_frequencies = {k: v for k, v in word_frequencies.items() if v > 0}
    
    if not word_frequencies:
        logger.warning("No valid words found for wordcloud after processing")
        return None

    # Create color scheme
    colors = ['#1d3d71', '#2958a4', '#3771c8', '#5d9cff', '#87bbff']
    cmap = LinearSegmentedColormap.from_list('economist', colors, N=len(colors))
    
    # Create wordcloud
    try:
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap=cmap,
            min_font_size=10,
            max_font_size=150,
            prefer_horizontal=0.7,
            relative_scaling=0.5
        ).generate_from_frequencies(word_frequencies)
        
        return wordcloud
    except Exception as e:
        logger.error(f"Error creating wordcloud: {e}")
        return None

def generate_wordclouds(documents: Dict[str, Dict], mc_terms: List[str], output_dir: str):
    """Generate wordclouds for each decade."""
    # Ensure output directory exists
    wordcloud_dir = os.path.join(output_dir, 'wordclouds')
    os.makedirs(wordcloud_dir, exist_ok=True)
    
    # Define decades
    decades = [
        (1970, 1979, '1970s'),
        (1980, 1989, '1980s'),
        (1990, 1999, '1990s'),
        (2000, 2009, '2000s'),
        (2010, 2019, '2010s'),
        (2020, 2029, '2020s')
    ]
    
    wordcloud_images = []
    
    for start_year, end_year, decade_name in decades:
        logger.info(f"Processing decade: {decade_name}")
        
        # Prepare text for this decade
        decade_text = prepare_text_for_wordcloud(documents, (start_year, end_year))
        
        if decade_text:
            wordcloud = create_wordcloud(decade_text, mc_terms)
            
            if wordcloud:
                # Save individual wordcloud
                filename = f'wc_{decade_name}_mc_terms.png'
                filepath = os.path.join(wordcloud_dir, filename)
                
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Management Control Terms: {decade_name}', 
                         fontsize=16, pad=20)
                plt.tight_layout(pad=0)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Store the image for the composite
                wordcloud_images.append(Image.open(filepath))
                logger.info(f"Created wordcloud for {decade_name}")
            else:
                logger.warning(f"Could not create wordcloud for {decade_name}")
        else:
            logger.warning(f"No text available for {decade_name}")
    
    if wordcloud_images:
        create_composite_image(wordcloud_images, wordcloud_dir)
    else:
        logger.warning("No wordcloud images generated to create composite")

def create_composite_image(images: List[Image.Image], output_dir: str):
    """Create a composite image from individual wordclouds."""
    if not images:
        logger.warning("No images provided for composite")
        return
    
    # Calculate dimensions for 2x3 grid
    img_width = images[0].width
    img_height = images[0].height
    composite_width = img_width * 2
    composite_height = img_height * 3
    
    # Create new image with white background
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    
    # Paste images in 2x3 grid
    for idx, img in enumerate(images):
        x = (idx % 2) * img_width
        y = (idx // 2) * img_height
        composite.paste(img, (x, y))
    
    # Save composite image
    composite_path = os.path.join(output_dir, 'wc_composite_mc_terms.png')
    composite.save(composite_path, 'PNG', quality=100)
    logger.info(f"Created composite wordcloud image: {composite_path}")

def run_wordcloud_analysis(documents: Dict[str, Dict], mc_terms: List[str], output_dir: str):
    """Main function to run wordcloud analysis."""
    logger.info("Starting wordcloud analysis...")
    try:
        generate_wordclouds(documents, mc_terms, output_dir)
        logger.info("Wordcloud analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in wordcloud analysis: {str(e)}")
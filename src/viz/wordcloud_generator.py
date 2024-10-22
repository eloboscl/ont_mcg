import json
import os
import random
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from wordcloud import WordCloud


def generate_wordclouds(analyzed_documents: Dict[str, Dict], mc_terms: List[str], output_dir: str):
    # Ensure the output directory exists
    wordcloud_dir = os.path.join(output_dir, 'wordclouds')
    os.makedirs(wordcloud_dir, exist_ok=True)

    # Prepare data for each decade
    decades = {
        '1970s': (1970, 1979),
        '1980s': (1980, 1989),
        '1990s': (1990, 1999),
        '2000s': (2000, 2009),
        '2010s': (2010, 2019),
        '2020s': (2020, 2024)
    }

    decade_texts = {decade: '' for decade in decades}

    for doc in analyzed_documents.values():
        year = int(doc.get('year', 0))
        for decade, (start, end) in decades.items():
            if start <= year <= end:
                decade_texts[decade] += doc.get('cleaned_text', '') + ' '

    # Generate and save wordclouds for each decade
    wordcloud_images = []
    for decade, text in decade_texts.items():
        wordcloud = create_wordcloud(text, mc_terms)
        
        # Save individual wordcloud
        filename = f'wc_{decade}_mc_terms.png'
        filepath = os.path.join(wordcloud_dir, filename)
        wordcloud.to_file(filepath)
        
        # Store the image for the composite
        wordcloud_images.append(Image.open(filepath))

    # Create and save composite image
    composite_image = create_composite_image(wordcloud_images)
    composite_image.save(os.path.join(wordcloud_dir, 'wc_composite_mc_terms.png'))

def create_wordcloud(text: str, mc_terms: List[str]) -> WordCloud:
    word_frequencies = Counter(text.split())
    
    for term in mc_terms:
        if term.lower() in word_frequencies:
            word_frequencies[term.lower()] *= 2

    # Create a color map similar to The Economist's style
    colors = ['#1d3d71', '#2958a4', '#3771c8', '#5d9cff', '#87bbff']
    n_bins = len(colors)
    cmap = LinearSegmentedColormap.from_list('economist', colors, N=n_bins)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return tuple(int(x*255) for x in cmap(random.randint(0, n_bins-1)))

    wordcloud = WordCloud(width=1600, height=1200, 
                          background_color='white', 
                          min_font_size=9,
                          color_func=color_func,
                          font_path='/input/EconoSansReduced-53BookExpanded.ttf',  # Replace with actual path
                          ).generate_from_frequencies(word_frequencies)
    
    return wordcloud

def create_composite_image(images: List[Image.Image]) -> Image.Image:
    # Determine the size of the composite image
    width = max(img.width for img in images)
    height = sum(img.height for img in images)

    # Create a new image with the calculated size
    composite = Image.new('RGB', (width, height), color='white')

    # Paste the individual wordcloud images into the composite
    y_offset = 0
    for img in images:
        composite.paste(img, (0, y_offset))
        y_offset += img.height

    return composite

# This function should be called from your main script
def run_wordcloud_analysis(analyzed_documents: Dict[str, Dict], mc_terms: List[str], output_dir: str):
    generate_wordclouds(analyzed_documents, mc_terms, output_dir)
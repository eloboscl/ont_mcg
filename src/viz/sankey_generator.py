import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def load_decade_data(wordcloud_dir: str) -> Dict[str, pd.DataFrame]:
    """Load frequency data for each decade."""
    decades = {}
    for decade in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
        file_path = os.path.join(wordcloud_dir, f'wordcloud_freq_{decade}.csv')
        if os.path.exists(file_path):
            decades[decade] = pd.read_csv(file_path)
    return decades

def get_top_terms(decade_data: Dict[str, pd.DataFrame], n_terms: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """Get top N terms for each decade with their frequencies."""
    top_terms = {}
    for decade, df in decade_data.items():
        top_terms[decade] = list(zip(df['concept'].head(n_terms), df['frequency'].head(n_terms)))
    return top_terms

def prepare_sankey_data(top_terms: Dict[str, List[Tuple[str, float]]]) -> Tuple[List[str], List[Dict]]:
    """Prepare data for Sankey diagram."""
    decades = list(top_terms.keys())
    all_terms = set()
    term_indices = {}
    current_index = 0
    
    # First pass: collect all unique terms and assign indices
    for decade_terms in top_terms.values():
        for term, _ in decade_terms:
            if term not in term_indices:
                term_indices[term] = current_index
                current_index += 1
    
    # Prepare nodes
    nodes = []
    for decade in decades:
        for term, freq in top_terms[decade]:
            nodes.append(f"{term} ({decade})")
    
    # Prepare links
    links = []
    for i in range(len(decades) - 1):
        current_decade = decades[i]
        next_decade = decades[i + 1]
        current_terms = dict(top_terms[current_decade])
        next_terms = dict(top_terms[next_decade])
        
        # Find common terms between consecutive decades
        for term in current_terms:
            if term in next_terms:
                source_idx = nodes.index(f"{term} ({current_decade})")
                target_idx = nodes.index(f"{term} ({next_decade})")
                value = min(current_terms[term], next_terms[term])
                links.append({
                    'source': source_idx,
                    'target': target_idx,
                    'value': value
                })
    
    return nodes, links

def create_sankey_diagrams(wordcloud_dir: str, output_dir: str, n_terms: int = 5):
    """Create both horizontal and vertical Sankey diagrams."""
    logger.info("Loading decade data...")
    decade_data = load_decade_data(wordcloud_dir)
    
    if not decade_data:
        logger.error("No decade data found!")
        return
    
    logger.info("Getting top terms...")
    top_terms = get_top_terms(decade_data, n_terms)
    
    logger.info("Preparing Sankey data...")
    nodes, links = prepare_sankey_data(top_terms)
    
    # Create horizontal Sankey
    fig_h = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="#1d3d71"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color='rgba(29, 61, 113, 0.4)'  # Semi-transparent blue
        )
    )])
    
    fig_h.update_layout(
        title_text="Evolution of Top Management Control Terms Across Decades (Horizontal)",
        font_size=12,
        width=1200,
        height=800
    )
    
    # Create vertical Sankey
    fig_v = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="#1d3d71"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color='rgba(29, 61, 113, 0.4)'  # Semi-transparent blue
        ),
        orientation='v'
    )])
    
    fig_v.update_layout(
        title_text="Evolution of Top Management Control Terms Across Decades (Vertical)",
        font_size=12,
        width=800,
        height=1200
    )
    
    # Save the figures
    logger.info("Saving Sankey diagrams...")
    horizontal_path = os.path.join(output_dir, 'sankey_horizontal.html')
    vertical_path = os.path.join(output_dir, 'sankey_vertical.html')
    horizontal_png_path = os.path.join(output_dir, 'sankey_horizontal.png')
    vertical_png_path = os.path.join(output_dir, 'sankey_vertical.png')
    
    # Save interactive HTML versions
    fig_h.write_html(horizontal_path)
    fig_v.write_html(vertical_path)
    
    # Save static PNG versions
    fig_h.write_image(horizontal_png_path, scale=2)  # Higher resolution for Word
    fig_v.write_image(vertical_png_path, scale=2)
    
    # Save data to Excel for reference
    excel_path = os.path.join(output_dir, 'term_evolution_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Save top terms for each decade
        for decade, terms in top_terms.items():
            df = pd.DataFrame(terms, columns=['Term', 'Frequency'])
            df.to_excel(writer, sheet_name=decade, index=False)
        
        # Save links data
        links_df = pd.DataFrame(links)
        links_df['source_term'] = links_df['source'].map(lambda x: nodes[x])
        links_df['target_term'] = links_df['target'].map(lambda x: nodes[x])
        links_df.to_excel(writer, sheet_name='Term_Flows', index=False)
    
    logger.info(f"Saved diagrams to {output_dir}")
    logger.info(f"Saved term evolution data to {excel_path}")

if __name__ == "__main__":
    # Example usage
    wordcloud_dir = "path/to/wordclouds"
    output_dir = "path/to/output"
    create_sankey_diagrams(wordcloud_dir, output_dir, n_terms=5)
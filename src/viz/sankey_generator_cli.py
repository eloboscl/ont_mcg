#!/usr/bin/env python3
import argparse
import logging
import os

import pandas as pd
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sankey_from_csv(input_file: str, output_dir: str, n_terms: int = 5):
    """
    Create Sankey diagrams from a composite CSV file.
    
    Args:
        input_file: Path to input CSV with columns [rank, concept, frequency, decade]
        output_dir: Directory to save output files
        n_terms: Number of top terms to include per decade
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the CSV file
        logger.info(f"Reading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Get top N terms for each decade
        decades = sorted(df['decade'].unique())
        top_terms = {}
        for decade in decades:
            decade_df = df[df['decade'] == decade].head(n_terms)
            top_terms[decade] = list(zip(decade_df['concept'], decade_df['frequency']))
        
        # Prepare nodes and links
        nodes = []
        node_indices = {}
        current_index = 0
        
        # Create nodes
        for decade in decades:
            for term, _ in top_terms[decade]:
                node_label = f"{term} ({decade})"
                nodes.append(node_label)
                node_indices[node_label] = current_index
                current_index += 1
        
        # Create links
        links = []
        for i in range(len(decades) - 1):
            current_decade = decades[i]
            next_decade = decades[i + 1]
            current_terms = dict(top_terms[current_decade])
            next_terms = dict(top_terms[next_decade])
            
            for term in current_terms:
                if term in next_terms:
                    source_idx = node_indices[f"{term} ({current_decade})"]
                    target_idx = node_indices[f"{term} ({next_decade})"]
                    value = min(current_terms[term], next_terms[term])
                    links.append({
                        'source': source_idx,
                        'target': target_idx,
                        'value': value
                    })
        
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
                color='rgba(29, 61, 113, 0.4)'
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
                color='rgba(29, 61, 113, 0.4)'
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
        fig_h.write_html(os.path.join(output_dir, 'sankey_horizontal.html'))
        fig_v.write_html(os.path.join(output_dir, 'sankey_vertical.html'))
        fig_h.write_image(os.path.join(output_dir, 'sankey_horizontal.png'), scale=2)
        fig_v.write_image(os.path.join(output_dir, 'sankey_vertical.png'), scale=2)
        
        # Save flow data to Excel
        excel_path = os.path.join(output_dir, 'term_evolution_data.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            # Save top terms for each decade
            for decade in decades:
                decade_df = df[df['decade'] == decade].head(n_terms)
                decade_df.to_excel(writer, sheet_name=decade, index=False)
            
            # Save links data
            links_df = pd.DataFrame(links)
            links_df['source_term'] = links_df['source'].map(lambda x: nodes[x])
            links_df['target_term'] = links_df['target'].map(lambda x: nodes[x])
            links_df.to_excel(writer, sheet_name='Term_Flows', index=False)
        
        logger.info(f"Saved diagrams to {output_dir}")
        logger.info(f"Saved term evolution data to {excel_path}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Sankey diagrams from term frequency CSV")
    parser.add_argument('input_file', help="Path to input CSV file")
    parser.add_argument('output_dir', help="Directory to save output files")
    parser.add_argument('--top_terms', type=int, default=5, 
                       help="Number of top terms to include per decade (default: 5)")
    
    args = parser.parse_args()
    
    create_sankey_from_csv(args.input_file, args.output_dir, args.top_terms)

if __name__ == "__main__":
    main()
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_bar_plot(df: pd.DataFrame, n_terms: int, output_dir: str):
    """Create a detailed bar plot showing top terms by decade."""
    decades = sorted(df['decade'].unique())
    fig = make_subplots(
        rows=len(decades), cols=1,
        subplot_titles=[f"Top {n_terms} Terms: {decade}" for decade in decades],
        vertical_spacing=0.05
    )

    for idx, decade in enumerate(decades, 1):
        decade_df = df[df['decade'] == decade].head(n_terms)
        
        fig.add_trace(
            go.Bar(
                x=decade_df['frequency'],
                y=decade_df['concept'],
                orientation='h',
                marker_color='#1d3d71',
                text=decade_df['frequency'],
                textposition='auto',
                name=decade
            ),
            row=idx, col=1
        )

        # Update layout for each subplot
        fig.update_xaxes(title_text="Frequency", row=idx, col=1)
        fig.update_yaxes(title_text="Terms", row=idx, col=1, autorange="reversed")

    # Update overall layout
    fig.update_layout(
        title={
            'text': f'Evolution of Top {n_terms} Management Control Terms by Decade',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        showlegend=False,
        width=1200,
        height=300 * len(decades),  # Moved height here
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=100, b=50),  # Added margins for better spacing
        annotations=[
            dict(
                text="Source: Management Control Literature Analysis (1970-2024)",
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    # Save the figure
    fig.write_html(os.path.join(output_dir, 'top_terms_by_decade.html'))
    fig.write_image(os.path.join(output_dir, 'top_terms_by_decade.png'), scale=2)
    return




def create_term_evolution_plot(df: pd.DataFrame, n_terms: int, output_dir: str):
    """
    Create a vertical term evolution plot with proportional decade splits and term connections.
    
    Args:
        df: DataFrame with columns [rank, concept, frequency, decade]
        n_terms: Number of top terms to show per decade
        output_dir: Directory to save output files
    """
    logger.info("Creating term evolution plot...")
    
    # Sort decades and get top terms for each
    decades = sorted(df['decade'].unique())
    decade_data = {}
    all_terms = set()
    
    # Process data for each decade
    for decade in decades:
        decade_df = df[df['decade'] == decade].head(n_terms)
        total_freq = decade_df['frequency'].sum()
        
        # Calculate proportions and store data
        decade_data[decade] = {
            'terms': decade_df['concept'].tolist(),
            'frequencies': decade_df['frequency'].tolist(),
            'proportions': (decade_df['frequency'] / total_freq).tolist()
        }
        all_terms.update(decade_df['concept'])
    
    # Create figure
    fig = go.Figure()
    
    # Calculate dimensions
    n_decades = len(decades)
    decade_height = 1.0 / n_decades
    y_positions = {decade: 1 - (i + 0.5) * decade_height 
                  for i, decade in enumerate(decades)}
    
    # Color palette for terms
    colors = px.colors.qualitative.Set3
    term_colors = {term: colors[i % len(colors)] 
                  for i, term in enumerate(all_terms)}
    
    # Draw decade bars and labels
    for decade in decades:
        data = decade_data[decade]
        y_pos = y_positions[decade]
        
        # Calculate x positions for terms within decade
        cumsum = np.cumsum([0] + data['proportions'])
        
        # Add stacked bar segments
        for i, (term, freq, prop) in enumerate(zip(data['terms'], 
                                                 data['frequencies'],
                                                 data['proportions'])):
            fig.add_trace(go.Bar(
                x=[prop],
                y=[y_pos],
                orientation='h',
                name=term,
                showlegend=True,
                marker_color=term_colors[term],
                text=f"{term}<br>({freq})",
                textposition="auto",
                hoverinfo="text",
                base=cumsum[i],
            ))
        
        # Add decade label
        fig.add_annotation(
            x=-0.05,
            y=y_pos,
            text=decade,
            showarrow=False,
            xanchor="right",
            font=dict(size=14)
        )
    
    # Draw connections between identical terms in consecutive decades
    for i in range(len(decades) - 1):
        current_decade = decades[i]
        next_decade = decades[i + 1]
        
        current_data = decade_data[current_decade]
        next_data = decade_data[next_decade]
        
        current_y = y_positions[current_decade]
        next_y = y_positions[next_decade]
        
        # Find matching terms
        for term in set(current_data['terms']) & set(next_data['terms']):
            # Get positions in each decade
            current_idx = current_data['terms'].index(term)
            next_idx = next_data['terms'].index(term)
            
            # Calculate x positions
            current_x = sum(current_data['proportions'][:current_idx]) + current_data['proportions'][current_idx]/2
            next_x = sum(next_data['proportions'][:next_idx]) + next_data['proportions'][next_idx]/2
            
            # Draw connection
            fig.add_trace(go.Scatter(
                x=[current_x, next_x],
                y=[current_y, next_y],
                mode='lines',
                line=dict(
                    color=term_colors[term],
                    width=2 * np.sqrt(min(current_data['frequencies'][current_idx],
                                        next_data['frequencies'][next_idx]))
                ),
                showlegend=False,
                hoverinfo="none"
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Evolution of Top {n_terms} Management Control Terms by Decade (1970-2024)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        barmode='stack',
        showlegend=True,
        legend_title="Terms",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        ),
        width=1200,
        height=800,
        xaxis=dict(
            title="Proportion within Decade",
            tickformat=".0%",
            range=[-0.1, 1.1]  # Add space for decade labels
        ),
        yaxis=dict(
            showticklabels=False,
            range=[0, 1]
        ),
        margin=dict(l=100, r=200, t=100, b=50),  # Adjust margins for labels
        annotations=[
            dict(
                text="Note: Bar width shows term proportion within decade.<br>"
                     "Connections show term persistence between decades.<br>"
                     "Numbers in parentheses show absolute frequency.",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                text="Source: Management Control Literature Analysis",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )
    
    # Save the figure
    fig.write_html(os.path.join(output_dir, 'term_evolution.html'))
    fig.write_image(os.path.join(output_dir, 'term_evolution.png'), scale=2)
    
    # Save data to Excel for reference
    excel_path = os.path.join(output_dir, 'term_evolution_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for decade in decades:
            decade_df = df[df['decade'] == decade].head(n_terms)
            decade_df.to_excel(writer, sheet_name=decade, index=False)
    
    logger.info(f"Term evolution visualization saved to {output_dir}")
    return

def create_dynamic_term_sankey(df: pd.DataFrame, n_terms: int, output_dir: str):
    """
    Create Sankey diagrams showing term emergence and disappearance across decades.
    Includes special nodes for new and departed terms between decades.
    """
    logger.info("Creating dynamic term evolution Sankey diagram...")
    
    decades = sorted(df['decade'].unique())
    
    # Get top terms for each decade
    decade_terms = {}
    for decade in decades:
        decade_df = df[df['decade'] == decade].head(n_terms)
        decade_terms[decade] = set(decade_df['concept'])
    
    # Initialize node and link tracking
    nodes = []
    links = []
    node_indices = {}
    current_index = 0
    
    # Add initial decade terms
    first_decade = decades[0]
    for term in decade_terms[first_decade]:
        term_freq = df[(df['decade'] == first_decade) & (df['concept'] == term)]['frequency'].iloc[0]
        node_label = f"{term}<br>({first_decade}: {term_freq})"
        nodes.append(node_label)
        node_indices[f"{term}_{first_decade}"] = current_index
        current_index += 1
    
    # Process transitions between decades
    for i in range(len(decades) - 1):
        current_decade = decades[i]
        next_decade = decades[i + 1]
        
        current_terms = decade_terms[current_decade]
        next_terms = decade_terms[next_decade]
        
        # Add "New Terms" and "Departed Terms" nodes for this transition
        new_terms_idx = current_index
        nodes.append(f"New Terms<br>{next_decade}")
        current_index += 1
        
        departed_terms_idx = current_index
        nodes.append(f"Departed Terms<br>{current_decade}-{next_decade}")
        current_index += 1
        
        # Add nodes for next decade's terms
        for term in next_terms:
            term_freq = df[(df['decade'] == next_decade) & (df['concept'] == term)]['frequency'].iloc[0]
            node_label = f"{term}<br>({next_decade}: {term_freq})"
            nodes.append(node_label)
            node_indices[f"{term}_{next_decade}"] = current_index
            current_index += 1
        
        # Create links
        # 1. Persistent terms
        persistent_terms = current_terms.intersection(next_terms)
        for term in persistent_terms:
            source_idx = node_indices[f"{term}_{current_decade}"]
            target_idx = node_indices[f"{term}_{next_decade}"]
            value = df[(df['decade'] == next_decade) & (df['concept'] == term)]['frequency'].iloc[0]
            links.append({
                'source': source_idx,
                'target': target_idx,
                'value': value,
                'color': 'rgba(29, 61, 113, 0.4)'  # Blue for persistent terms
            })
        
        # 2. Disappeared terms
        disappeared_terms = current_terms - next_terms
        for term in disappeared_terms:
            source_idx = node_indices[f"{term}_{current_decade}"]
            value = df[(df['decade'] == current_decade) & (df['concept'] == term)]['frequency'].iloc[0]
            links.append({
                'source': source_idx,
                'target': departed_terms_idx,
                'value': value,
                'color': 'rgba(255, 0, 0, 0.3)'  # Red for departed terms
            })
        
        # 3. New terms
        new_terms = next_terms - current_terms
        for term in new_terms:
            target_idx = node_indices[f"{term}_{next_decade}"]
            value = df[(df['decade'] == next_decade) & (df['concept'] == term)]['frequency'].iloc[0]
            links.append({
                'source': new_terms_idx,
                'target': target_idx,
                'value': value,
                'color': 'rgba(0, 255, 0, 0.3)'  # Green for new terms
            })
    
    # Create horizontal Sankey
    fig_h = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=['#1d3d71' if 'New Terms' not in node and 'Departed Terms' not in node 
                   else '#00ff00' if 'New Terms' in node
                   else '#ff0000' for node in nodes]  # Color-code special nodes
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color=[link['color'] for link in links]
        )
    )])

    fig_h.update_layout(
        title={
            'text': 'Dynamic Evolution of Management Control Terms (1970-2024)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        width=1600,
        height=1000,
        annotations=[
            dict(
                text="Node colors: Blue = Terms, Green = New Terms, Red = Departed Terms<br>"
                     "Link colors: Blue = Persistent, Green = Emerging, Red = Departing",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                text="Source: Management Control Literature Analysis (1970-2024)",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    # Create vertical version
    fig_v = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=['#1d3d71' if 'New Terms' not in node and 'Departed Terms' not in node 
                   else '#00ff00' if 'New Terms' in node
                   else '#ff0000' for node in nodes]
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color=[link['color'] for link in links]
        ),
        orientation='v'
    )])

    fig_v.update_layout(
        title={
            'text': 'Dynamic Evolution of Management Control Terms (1970-2024)',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        width=1000,
        height=1600,
        annotations=[
            dict(
                text="Node colors: Blue = Terms, Green = New Terms, Red = Departed Terms<br>"
                     "Link colors: Blue = Persistent, Green = Emerging, Red = Departing",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                text="Source: Management Control Literature Analysis (1970-2024)",
                xref="paper", yref="paper",
                x=0.5, y=-0.07,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    # Save figures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for fig, orientation in [(fig_h, 'horizontal'), (fig_v, 'vertical')]:
        base_name = f'dynamic_sankey_{orientation}_{timestamp}'
        fig.write_html(os.path.join(output_dir, f'{base_name}.html'))
        fig.write_image(os.path.join(output_dir, f'{base_name}.png'), scale=2)

    # Save analysis to Excel
    excel_path = os.path.join(output_dir, f'term_dynamics_analysis_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Overall statistics
        stats_data = []
        for i in range(len(decades) - 1):
            current_decade = decades[i]
            next_decade = decades[i + 1]
            current_terms = decade_terms[current_decade]
            next_terms = decade_terms[next_decade]
            
            stats_data.append({
                'Transition': f'{current_decade} → {next_decade}',
                'Persistent Terms': len(current_terms.intersection(next_terms)),
                'New Terms': len(next_terms - current_terms),
                'Departed Terms': len(current_terms - next_terms),
                'Total Terms': len(next_terms)
            })
        
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Term Dynamics', index=False)
        
        # Detailed term lists by decade
        for decade in decades:
            decade_df = df[df['decade'] == decade].head(n_terms)
            decade_df.to_excel(writer, sheet_name=f'{decade}_Terms', index=False)
    
    logger.info(f"Saved dynamic term evolution visualizations and analysis to {output_dir}")
    return

def create_enhanced_sankey(df: pd.DataFrame, n_terms: int, output_dir: str):
    """Create enhanced Sankey diagrams with academic styling."""
    decades = sorted(df['decade'].unique())
    
    # Get top terms for each decade
    top_terms = {}
    for decade in decades:
        decade_df = df[df['decade'] == decade].head(n_terms)
        top_terms[decade] = list(zip(decade_df['concept'], decade_df['frequency']))
    
    # Prepare nodes and links
    nodes = []
    node_indices = {}
    current_index = 0
    
    # Create nodes with decade information
    for decade in decades:
        for term, freq in top_terms[decade]:
            node_label = f"{term}<br>({decade}: {freq})"
            nodes.append(node_label)
            node_indices[f"{term} ({decade})"] = current_index
            current_index += 1
    
    # Create links with enhanced logic
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
            pad=20,
            thickness=30,
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
        title={
            'text': 'Evolution of Management Control Terms Across Decades (1970-2024)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        width=1600,
        height=1000,
        annotations=[
            dict(
                text="Note: Node size represents term frequency. Links show term persistence across decades.",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                text="Source: Management Control Literature Analysis (1970-2024)",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    # Create vertical Sankey
    fig_v = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
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
        title={
            'text': 'Evolution of Management Control Terms Across Decades (1970-2024)',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        width=1000,
        height=1600,
        annotations=[
            dict(
                text="Note: Node size represents term frequency. Links show term persistence across decades.",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                text="Source: Management Control Literature Analysis (1970-2024)",
                xref="paper", yref="paper",
                x=0.5, y=-0.07,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    # Save the figures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for fig, orientation in [(fig_h, 'horizontal'), (fig_v, 'vertical')]:
        base_name = f'sankey_{orientation}_{timestamp}'
        fig.write_html(os.path.join(output_dir, f'{base_name}.html'))
        fig.write_image(os.path.join(output_dir, f'{base_name}.png'), scale=2)

    # Save data to Excel with detailed analysis
    excel_path = os.path.join(output_dir, f'term_evolution_analysis_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Summary sheet
        pd.DataFrame({
            'Analysis Parameter': [
                'Time Period',
                'Number of Terms per Decade',
                'Total Unique Terms',
                'Most Frequent Term',
                'Most Persistent Term'
            ],
            'Value': [
                '1970-2024',
                str(n_terms),
                str(len(set(df['concept']))),
                df.nlargest(1, 'frequency')['concept'].iloc[0],
                'TBD'  # You might want to calculate this based on your criteria
            ]
        }).to_excel(writer, sheet_name='Analysis Summary', index=False)

        # Decade sheets with detailed statistics
        for decade in decades:
            decade_df = df[df['decade'] == decade].head(n_terms)
            decade_df.to_excel(writer, sheet_name=f'{decade}_Analysis', index=False)

    logger.info(f"Saved visualizations and analysis to {output_dir}")
    return

def validate_paths(input_file: str, output_dir: str) -> tuple[Path, Path]:
    """Validate input and output paths."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.suffix == '.csv':
        raise ValueError(f"Input file must be a CSV file, got: {input_path.suffix}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return input_path, output_path

def main():
    parser = argparse.ArgumentParser(description="Generate academic visualizations for term evolution")
    parser.add_argument('input_file', help="Path to input CSV file containing term frequency data")
    parser.add_argument('output_dir', help="Directory to save output visualizations")
    parser.add_argument('--top_terms', type=int, default=10, 
                       help="Number of top terms to include per decade (default: 10)")
    parser.add_argument('--dpi', type=int, default=300,
                       help="DPI for output images (default: 300)")
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        input_path, output_path = validate_paths(args.input_file, args.output_dir)
        logger.info(f"Input file: {input_path}")
        logger.info(f"Output directory: {output_path}")
        
        # Read data
        logger.info("Reading input data...")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Create visualizations
        #logger.info("Creating bar plot...")
        #create_bar_plot(df, args.top_terms, output_path)
        
        logger.info("Creating dynamic term Sankey diagram...")
        create_dynamic_term_sankey(df, args.top_terms, output_path)
        
        logger.info(f"All visualizations completed successfully. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
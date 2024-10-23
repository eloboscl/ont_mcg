#!/usr/bin/env python3
"""
Term Evolution Visualization Generator
Usage: python term_evolution.py input.csv output_dir --top_terms 5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_enhanced_term_evolution(df: pd.DataFrame, n_terms: int, output_dir: str):
    """
    Create two complementary visualizations:
    1. Streamlined Evolution Plot
    2. Term Trajectory Grid
    """
    decades = sorted(df['decade'].unique())
    
    # First Visualization: Streamlined Evolution
    fig1 = go.Figure()
    
    # Process data
    decade_data = {}
    all_terms = set()
    
    for decade in decades:
        decade_df = df[df['decade'] == decade].head(n_terms)
        total_freq = decade_df['frequency'].sum()
        
        decade_data[decade] = {
            'terms': decade_df['concept'].tolist(),
            'frequencies': decade_df['frequency'].tolist(),
            'proportions': (decade_df['frequency'] / total_freq).tolist()
        }
        all_terms.update(decade_df['concept'])
    
    # Use a distinct color palette
    colors = px.colors.qualitative.Dark24[:len(all_terms)]  # More distinct colors
    term_colors = {term: colors[i] for i, term in enumerate(all_terms)}
    
    # Calculate positions
    n_decades = len(decades)
    y_positions = {decade: 1 - (i + 0.5) / n_decades for i, decade in enumerate(decades)}
    
    # Draw bars with improved spacing
    for decade in decades:
        data = decade_data[decade]
        y_pos = y_positions[decade]
        cumsum = np.cumsum([0] + data['proportions'])
        
        for i, (term, freq, prop) in enumerate(zip(data['terms'], 
                                                 data['frequencies'],
                                                 data['proportions'])):
            # Add bar with enhanced labels
            fig1.add_trace(go.Bar(
                x=[prop],
                y=[y_pos],
                orientation='h',
                name=term,
                showlegend=True,
                marker_color=term_colors[term],
                text=f"{term}<br>{freq} ({prop:.1%})",
                textposition="auto",
                hoverinfo="text",
                base=cumsum[i],
            ))
    
    # Add smoother connections with opacity based on frequency
    for i in range(len(decades) - 1):
        current_decade = decades[i]
        next_decade = decades[i + 1]
        
        current_data = decade_data[current_decade]
        next_data = decade_data[next_decade]
        
        current_y = y_positions[current_decade]
        next_y = y_positions[next_decade]
        
        for term in set(current_data['terms']) & set(next_data['terms']):
            current_idx = current_data['terms'].index(term)
            next_idx = next_data['terms'].index(term)
            
            current_x = sum(current_data['proportions'][:current_idx]) + current_data['proportions'][current_idx]/2
            next_x = sum(next_data['proportions'][:next_idx]) + next_data['proportions'][next_idx]/2
            
            # Calculate opacity based on frequency
            opacity = min(current_data['frequencies'][current_idx],
                        next_data['frequencies'][next_idx]) / max(df['frequency'])
            
            fig1.add_trace(go.Scatter(
                x=[current_x, next_x],
                y=[current_y, next_y],
                mode='lines',
                line=dict(
                    color=term_colors[term],
                    width=3,
                    shape='spline',  # Smoother curves
                    opacity=0.3 + 0.7 * opacity  # Minimum 0.3 opacity
                ),
                showlegend=False,
                hoverinfo="none"
            ))
    
    # Update layout
    fig1.update_layout(
        title={
            'text': f'Evolution of Top {n_terms} Management Control Terms (1970-2024)',
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
            font=dict(size=12),
            itemsizing='constant'  # Consistent legend item sizes
        ),
        width=1200,
        height=800,
        xaxis=dict(
            title="Proportion within Decade",
            tickformat=".0%",
            range=[-0.1, 1.1],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=True,
            ticktext=decades,
            tickvals=[y_positions[d] for d in decades],
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        margin=dict(l=100, r=200, t=100, b=50)
    )
    
    # Second Visualization: Term Trajectory Grid
    fig2 = go.Figure()
    
    # Create a grid of term trajectories
    all_terms_sorted = sorted(all_terms)
    for i, term in enumerate(all_terms_sorted):
        frequencies = []
        for decade in decades:
            decade_df = df[df['decade'] == decade]
            freq = decade_df[decade_df['concept'] == term]['frequency'].iloc[0] if term in decade_df['concept'].values else 0
            frequencies.append(freq)
        
        fig2.add_trace(go.Scatter(
            x=decades,
            y=frequencies,
            name=term,
            mode='lines+markers',
            line=dict(color=term_colors[term], width=2),
            marker=dict(size=10),
            hovertemplate="%{x}<br>%{y} occurrences<extra></extra>"
        ))
    
    fig2.update_layout(
        title="Term Frequency Trajectories Across Decades",
        xaxis_title="Decade",
        yaxis_title="Frequency",
        showlegend=True,
        legend_title="Terms",
        width=1200,
        height=800,
        template="simple_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Save visualizations
    fig1.write_html(os.path.join(output_dir, 'term_evolution_enhanced.html'))
    fig1.write_image(os.path.join(output_dir, 'term_evolution_enhanced.png'), scale=2)
    
    fig2.write_html(os.path.join(output_dir, 'term_trajectories.html'))
    fig2.write_image(os.path.join(output_dir, 'term_trajectories.png'), scale=2)
    return

def create_term_evolution_plot(df: pd.DataFrame, n_terms: int, output_dir: str):
    """Create enhanced term evolution visualization with proper error handling."""
    decades = sorted(df['decade'].unique())
    
    # Process data
    decade_data = {}
    all_terms = set()
    max_freq = df['frequency'].max()
    
    # First pass: collect data safely
    for decade in decades:
        decade_df = df[df['decade'] == decade]
        if len(decade_df) > 0:
            # Get top N terms for this decade
            decade_df = decade_df.head(n_terms)
            total_freq = decade_df['frequency'].sum()
            
            # Store data
            decade_data[decade] = {
                'terms': decade_df['concept'].tolist(),
                'frequencies': decade_df['frequency'].tolist(),
                'proportions': (decade_df['frequency'] / total_freq).tolist() if total_freq > 0 else []
            }
            all_terms.update(decade_df['concept'])
        else:
            logger.warning(f"No data found for decade: {decade}")
            decade_data[decade] = {
                'terms': [],
                'frequencies': [],
                'proportions': []
            }
    
    # Use a better color palette
    colors = px.colors.qualitative.D3[:len(all_terms)]
    term_colors = {term: colors[i % len(colors)] for i, term in enumerate(all_terms)}
    
    # Create figure
    fig = go.Figure()
    
    # Calculate positions
    n_decades = len(decades)
    y_positions = {decade: 1 - (i + 0.5) / n_decades for i, decade in enumerate(decades)}
    
    # Draw decade bars
    for decade in decades:
        data = decade_data[decade]
        if not data['terms']:  # Skip if no data
            continue
            
        y_pos = y_positions[decade]
        cumsum = np.cumsum([0] + data['proportions'])
        
        for i, (term, freq, prop) in enumerate(zip(data['terms'], 
                                                 data['frequencies'],
                                                 data['proportions'])):
            # Add bar segment
            fig.add_trace(go.Bar(
                x=[prop],
                y=[y_pos],
                orientation='h',
                name=term,
                showlegend=True,
                marker_color=term_colors[term],
                text=f"{term}<br>{freq} ({prop:.1%})",
                textposition="auto",
                hoverinfo="text",
                base=cumsum[i],
            ))
    
    # Draw connections with varying widths
    for i in range(len(decades) - 1):
        current_decade = decades[i]
        next_decade = decades[i + 1]
        
        current_data = decade_data[current_decade]
        next_data = decade_data[next_decade]
        
        # Skip if either decade has no data
        if not current_data['terms'] or not next_data['terms']:
            continue
        
        current_y = y_positions[current_decade]
        next_y = y_positions[next_decade]
        
        # Draw connections for matching terms
        matching_terms = set(current_data['terms']) & set(next_data['terms'])
        for term in matching_terms:
            try:
                current_idx = current_data['terms'].index(term)
                next_idx = next_data['terms'].index(term)
                
                # Calculate positions
                current_x = sum(current_data['proportions'][:current_idx]) + current_data['proportions'][current_idx]/2
                next_x = sum(next_data['proportions'][:next_idx]) + next_data['proportions'][next_idx]/2
                
                # Calculate line width based on frequency
                freq_factor = min(current_data['frequencies'][current_idx],
                                next_data['frequencies'][next_idx]) / max_freq
                line_width = 1 + 4 * freq_factor  # Width between 1 and 5
                
                # Add connection line
                fig.add_trace(go.Scatter(
                    x=[current_x, next_x],
                    y=[current_y, next_y],
                    mode='lines',
                    line=dict(
                        color=term_colors[term],
                        width=line_width,
                        shape='spline'
                    ),
                    showlegend=False,
                    hoverinfo="none"
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing connection for term {term}: {str(e)}")
                continue
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Evolution of Top {n_terms} Management Control Terms (1970-2024)',
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
            font=dict(size=12),
            itemsizing='constant'
        ),
        width=1200,
        height=800,
        xaxis=dict(
            title="Proportion within Decade",
            tickformat=".0%",
            range=[-0.1, 1.1],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=True,
            ticktext=decades,
            tickvals=[y_positions[d] for d in decades],
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        margin=dict(l=100, r=200, t=100, b=50),
        annotations=[
            dict(
                text="Line width indicates term persistence strength between decades",
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
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, 'term_evolution.html'))
    fig.write_image(os.path.join(output_dir, 'term_evolution.png'), scale=2)
    
    # Save data to Excel for reference
    excel_path = os.path.join(output_dir, 'term_evolution_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Summary sheet
        summary_data = []
        for decade in decades:
            data = decade_data[decade]
            summary_data.append({
                'Decade': decade,
                'Number of Terms': len(data['terms']),
                'Top Terms': ', '.join(data['terms']),
                'Total Frequency': sum(data['frequencies'])
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed data per decade
        for decade in decades:
            data = decade_data[decade]
            if data['terms']:
                decade_df = pd.DataFrame({
                    'Term': data['terms'],
                    'Frequency': data['frequencies'],
                    'Proportion': data['proportions']
                })
                decade_df.to_excel(writer, sheet_name=f'Decade_{decade}', index=False)

    logger.info(f"Term evolution visualization saved to {output_dir}")
    return

def validate_paths(input_file: str, output_dir: str) -> tuple[Path, Path]:
    """Validate input and output paths."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.suffix == '.csv':
        raise ValueError(f"Input file must be a CSV file, got: {input_path.suffix}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    return input_path, output_path

def main():
    parser = argparse.ArgumentParser(
        description="Generate term evolution visualization from frequency data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', help="Path to input CSV file")
    parser.add_argument('output_dir', help="Directory to save output files")
    parser.add_argument('--top_terms', type=int, default=5,
                       help="Number of top terms to show per decade")
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        input_path, output_path = validate_paths(args.input_file, args.output_dir)
        
        # Read data
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Create visualization
        create_term_evolution_plot(df, args.top_terms, output_path)
        #create_enhanced_term_evolution(df, args.top_terms, output_path)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
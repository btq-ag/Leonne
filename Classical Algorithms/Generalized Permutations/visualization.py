#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization.py

This script creates advanced network-based visualizations of permutation 
algorithms from the generalizedPermutations module. It demonstrates how different
Fisher-Yates shuffle variations operate in the context of various network
topologies (random, small-world, and scale-free networks).

The visualizations include:
1. Static network renderings for different topologies
2. Animated permutation operations showing element movements across network nodes
3. Color-coded nodes indicating fixed points vs. permuted elements
4. Curved arrows illustrating the mapping relationships during permutation

Each algorithm is applied to different network structures to help visualize how
permutation patterns emerge and how they interact with underlying network topology.

Author: Jeffrey Morais, BTQ
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Import functions from generalizedPermutations.py
from generalizedPermutations import (
    FisherYates, 
    permutationOrbitFisherYates, 
    symmetricNDFisherYates, 
    consensusFisherYates
)

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Define colors
COLORS = {
    'permutation': '#4CAF50',  # Green
    'orbit': '#F44336',        # Red
    'neutral': '#2196F3',      # Blue
    'symmetric': '#FFC107',    # Yellow/Gold
    'background': '#111111',
    'edge': '#555555',
    'highlight': '#9C27B0'     # Purple
}

def create_network_visualization(n_nodes=7, network_type='small_world'):
    """Create a network visualization for permutation demonstration"""
    # Create the network
    if network_type == 'random':
        G = nx.erdos_renyi_graph(n_nodes, 0.4)
    elif network_type == 'small_world':
        G = nx.watts_strogatz_graph(n_nodes, 3, 0.3)
    elif network_type == 'scale_free':
        G = nx.barabasi_albert_graph(n_nodes, 2)
    else:
        G = nx.complete_graph(n_nodes)
    
    # Plot the network
    plt.figure(figsize=(10, 8), facecolor=COLORS['background'])
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=COLORS['neutral'], 
            edge_color=COLORS['edge'], width=1.5, font_color='white',
            node_size=700, font_size=12)
    
    plt.title(f"{network_type.title()} Network", color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{network_type}_network.png"), 
                dpi=300, facecolor=COLORS['background'], bbox_inches='tight')
    plt.close()
    
    return G, pos

def create_permutation_animation(algorithm, input_set, G, pos, network_type, frames=20):
    """Create animation showing permutation algorithm on a network"""
    # Run algorithm to get final permutation
    result = algorithm(input_set.copy(), False)
    if isinstance(result, tuple):
        final_perm = result[0]
    else:
        final_perm = result
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    def update(frame):
        ax.clear()
        
        # For first frame, show initial state
        if frame == 0:
            current_perm = input_set.copy()
        # For last frame, show final state
        elif frame == frames - 1:
            current_perm = final_perm
        # For intermediate frames, interpolate
        else:
            current_perm = input_set.copy()
            num_elements_to_change = int((frame / (frames - 1)) * len(input_set))
            for i in range(num_elements_to_change):
                current_perm[i] = final_perm[i]
        
        # Draw network edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color=COLORS['edge'])
        
        # Draw nodes with colors based on permutation state
        node_colors = []
        for node in range(len(input_set)):
            if current_perm[node] == node:  # Fixed point
                node_colors.append(COLORS['symmetric'])
            else:  # Moved element
                node_colors.append(COLORS['permutation'])
                
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=range(len(input_set)), 
                              node_color=node_colors, node_size=700, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_family='monospace', font_color='white')
        
        # Add arrows showing permutation mapping
        for i in range(len(input_set)):
            if current_perm[i] != i:
                # Draw curved arrow from i to current_perm[i]
                ax.annotate("", 
                          xy=pos[current_perm[i]], 
                          xytext=pos[i],
                          arrowprops=dict(arrowstyle="->", color=COLORS['highlight'], 
                                         connectionstyle="arc3,rad=0.3", lw=2))
        
        ax.set_title(f"{algorithm.__name__} on {network_type} network - Frame {frame+1}/{frames}", 
                    color='white')
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=False)
    
    # Save animation
    filename = f"{algorithm.__name__}_{network_type}_animation.gif"
    ani.save(os.path.join(output_dir, filename), writer='pillow', fps=4, 
              dpi=120, savefig_kwargs={'facecolor': COLORS['background']})
    
    plt.close()

def generate_visualizations():
    """Generate visualizations for all permutation algorithms"""
    # Define input set
    input_set = list(range(7))
    
    # Define algorithms
    algorithms = [
        FisherYates,
        permutationOrbitFisherYates,
        symmetricNDFisherYates,
        consensusFisherYates
    ]
    
    # Define network types
    network_types = ['random', 'small_world', 'scale_free']
    
    # Create network visualizations
    networks = {}
    for network_type in network_types:
        print(f"Creating {network_type} network visualization...")
        G, pos = create_network_visualization(len(input_set), network_type)
        networks[network_type] = (G, pos)
    
    # Generate animations for each algorithm on each network
    for algorithm in algorithms:
        for network_type, (G, pos) in networks.items():
            print(f"Creating animation for {algorithm.__name__} on {network_type} network...")
            create_permutation_animation(algorithm, input_set, G, pos, network_type)

if __name__ == "__main__":
    print("Generating permutation visualizations...")
    generate_visualizations()
    print("All visualizations completed!")

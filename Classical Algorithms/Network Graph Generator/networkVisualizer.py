#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkVisualizer.py

This script creates visualizations and animations for the Graph Generator Toolkit.
It's built to complement graphGenerator.py by visualizing the bipartite graphs,
network evolution, and consensus simulations.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import os
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import random
import time

# Import functions from graphGenerator
from graphGenerator import consensusShuffle, edgeAssignment, sequenceVerifier

# Set output directory to current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define colors for different node types and consensus states
COLORS = {
    'u_nodes': '#2196F3',    # Blue for U nodes
    'v_nodes': '#FFC107',    # Yellow for V nodes
    'consensus': '#4CAF50',  # Green for consensus
    'dissent': '#F44336',    # Red for dissent
    'edge': '#555555',       # Gray for edges
    'highlight': '#9C27B0'   # Purple for highlighting
}

def create_network_evolution_animation(degree_sequence, u_size, v_size, n_frames=30, filename="network_evolution_animation"):
    """
    Create an animation showing network evolution under repeated consensus shuffles.
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_frames : int
        Number of animation frames
    filename : str
        Base filename for the output animation
    """
    # Verify the sequence and get initial edge assignment
    if not sequenceVerifier(degree_sequence, u_size, v_size, False):
        print("Invalid degree sequence. Cannot create animation.")
        return
    
    # Get initial edge assignment
    initial_edges = edgeAssignment(degree_sequence, u_size, v_size, False)
    
    # Determine figure size based on network size
    base_size = 10
    size_factor = max(1, np.log2(max(u_size, v_size)) / 2)
    fig_size = (base_size * size_factor, base_size * size_factor)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Generate a series of network states through consensus shuffling
    edge_sets = [initial_edges]
    current_edges = initial_edges.copy()
    
    for _ in range(n_frames - 1):
        current_edges = consensusShuffle(current_edges, False).tolist()
        edge_sets.append(current_edges)
    
    # Create a function to convert edge sets to networkx graphs
    def create_graph_from_edges(edges):
        G = nx.DiGraph()  # Using directed graph to show edge directionality
        
        # Add all nodes
        for u in range(u_size):
            G.add_node(f'u{u}', type='u')
        for v in range(v_size):
            G.add_node(f'v{v}', type='v')
        
        # Add edges
        for u, v in edges:
            G.add_edge(f'u{u}', f'v{v}')
        
        return G
    
    # Create initial graph and set positions
    G_initial = create_graph_from_edges(edge_sets[0])
    
    # Use a circular layout with U nodes on left, V nodes on right
    pos = {}
    
    # Calculate node size based on network size
    node_size = max(100, min(500, 1000 / np.sqrt(u_size + v_size)))
    font_size = max(6, min(12, 18 / np.sqrt(u_size + v_size)))
    
    # Position U nodes in a semicircle on the left
    radius_factor = 1.0 + (max(u_size, v_size) / 10)
    for i in range(u_size):
        angle = np.pi/2 + (i * np.pi) / (u_size - 1) if u_size > 1 else np.pi
        pos[f'u{i}'] = np.array([-np.cos(angle) * radius_factor, np.sin(angle) * radius_factor])
    
    # Position V nodes in a semicircle on the right
    for i in range(v_size):
        angle = np.pi/2 + (i * np.pi) / (v_size - 1) if v_size > 1 else np.pi
        pos[f'v{i}'] = np.array([np.cos(angle) * radius_factor, np.sin(angle) * radius_factor])
    
    # Animation update function
    def update(frame):
        ax.clear()
        
        # Create graph for current frame
        G = create_graph_from_edges(edge_sets[frame])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes if n.startswith('u')],
                              node_color=COLORS['u_nodes'],
                              node_size=node_size,
                              alpha=0.8,
                              ax=ax)
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes if n.startswith('v')],
                              node_color=COLORS['v_nodes'],
                              node_size=node_size,
                              alpha=0.8,
                              ax=ax)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos,
                              arrowstyle='-|>',
                              arrowsize=max(5, min(15, 20 / np.sqrt(len(edge_sets[frame])))),
                              width=max(0.5, min(1.5, 3 / np.sqrt(len(edge_sets[frame])))),
                              edge_color=COLORS['edge'],
                              alpha=0.7,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold', ax=ax)
        
        # Add title and info
        ax.set_title(f'Network Evolution - Frame {frame+1}/{n_frames}', fontsize=15)
        ax.text(0.02, 0.98, f"Shuffle: {frame}", transform=ax.transAxes, 
               fontsize=12, verticalalignment='top')
        
        # Add network statistics
        ax.text(0.02, 0.02, 
               f"Nodes: {u_size}+{v_size}, Edges: {len(edge_sets[frame])}", 
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        
        ax.axis('off')
        ax.set_xlim(-radius_factor*1.5, radius_factor*1.5)
        ax.set_ylim(-radius_factor*1.5, radius_factor*1.5)
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    
    # Save the animation
    dpi_value = max(80, min(120, 200 / size_factor))
    ani.save(os.path.join(output_dir, f"{filename}.gif"), writer='pillow', fps=3, dpi=dpi_value)
    plt.close()
    
    print(f"Saved network evolution animation to {filename}.gif")

def visualize_permutation_stability(degree_sequence, u_size, v_size, n_shuffles=50, filename="permutation_stability"):
    """
    Visualize the stability of permutations by tracking edge persistence.
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_shuffles : int
        Number of consensus shuffles to analyze
    filename : str
        Base filename for the output visualization
    """
    # Verify the sequence and get initial edge assignment
    if not sequenceVerifier(degree_sequence, u_size, v_size, False):
        print("Invalid degree sequence. Cannot visualize stability.")
        return
    
    # Get initial edge assignment
    initial_edges = edgeAssignment(degree_sequence, u_size, v_size, False)
    
    # Create matrix to track edge persistence
    edge_tracker = np.zeros((u_size, v_size))
    current_edges = initial_edges.copy()
    
    # Determine appropriate figure size based on network size
    fig_size = (10, 8)
    if max(u_size, v_size) > 10:
        fig_size = (12, 10)
    
    # Perform shuffles and track edge persistence
    for _ in tqdm(range(n_shuffles), desc=f"Analyzing {n_shuffles} shuffles"):
        # Update edge tracker based on current edges
        for u, v in current_edges:
            edge_tracker[u, v] += 1
        
        # Perform consensus shuffle
        current_edges = consensusShuffle(current_edges, False).tolist()
    
    # Normalize edge persistence
    edge_tracker = edge_tracker / n_shuffles
    
    # Visualize the edge persistence matrix
    plt.figure(figsize=fig_size)
    
    # Determine the appropriate colormap based on network size
    if u_size <= 5 and v_size <= 5:
        cmap = plt.cm.viridis
    else:
        # For larger networks, use a colormap that highlights variations better
        cmap = plt.cm.plasma
    
    # Create the heatmap with adjustments for large networks
    im = plt.imshow(edge_tracker, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, label='Edge Persistence Probability')
    
    # Add grid lines for clarity
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.3)
    
    # Add labels and title with stats
    edge_count = len(initial_edges)
    total_possible = u_size * v_size
    edge_density = edge_count / total_possible
    avg_persistence = np.mean(edge_tracker[edge_tracker > 0])
    
    title = f'Edge Persistence Probability Matrix\n'
    title += f'Network: {u_size}×{v_size}, Edges: {edge_count}, Density: {edge_density:.2f}, Shuffles: {n_shuffles}'
    
    plt.title(title, fontsize=14)
    plt.xlabel('V Nodes', fontsize=12)
    plt.ylabel('U Nodes', fontsize=12)
    
    # Set tick labels with appropriate font size
    font_size = max(6, min(10, 14 - 0.4 * max(u_size, v_size)))
    
    plt.xticks(range(v_size), [f'v{i}' for i in range(v_size)], fontsize=font_size)
    plt.yticks(range(u_size), [f'u{i}' for i in range(u_size)], fontsize=font_size)
    
    # Add stats annotation
    plt.annotate(
        f"Avg persistence: {avg_persistence:.3f}\nStd dev: {np.std(edge_tracker[edge_tracker > 0]):.3f}",
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation stability visualization to {filename}.png")

def generate_complex_network_degree_sequence(u_size=8, v_size=8, density=0.6, power_law_exp=2.1):
    """
    Generate a complex network degree sequence with power-law-like properties.
    
    Parameters:
    -----------
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    density : float
        Desired edge density (0-1)
    power_law_exp : float
        Power law exponent for degree distribution
    
    Returns:
    --------
    degree_sequence : list
        A valid degree sequence for the complex network
    """
    # Calculate total expected edges based on density
    max_edges = u_size * v_size
    target_edges = int(max_edges * density)
    
    # Generate power-law-like degree sequence for U
    u_sequence = []
    remaining_edges = target_edges
    
    # Generate degrees for U nodes following power-law-like distribution
    for i in range(u_size):
        if i == 0:  # Hub node
            degree = min(v_size, int(target_edges * 0.3))  # Hub gets ~30% of edges
        else:
            # Power law: probability ~ k^(-power_law_exp)
            scale = remaining_edges / (u_size - i) * 0.9  # Scale factor, gradually reducing
            degree = min(v_size, max(1, int(scale * (1 / ((i+1) ** (power_law_exp/3))))))
        
        u_sequence.append(degree)
        remaining_edges -= degree
        
        if remaining_edges <= 0:
            # Fill remaining nodes with minimum degree
            u_sequence.extend([1] * (u_size - len(u_sequence)))
            break
    
    # Ensure we don't exceed target edges
    while sum(u_sequence) > target_edges:
        idx = np.random.randint(1, len(u_sequence))  # Don't reduce hub
        if u_sequence[idx] > 1:
            u_sequence[idx] -= 1
    
    # Now generate V sequence to match U
    v_counts = [0] * v_size
    u_sum = sum(u_sequence)
    
    # Distribute connections according to preferential attachment
    node_weights = np.ones(v_size)
    
    for u_node, degree in enumerate(u_sequence):
        for _ in range(degree):
            # Choose v node with probability proportional to its weight (preferential attachment)
            probabilities = node_weights / np.sum(node_weights)
            v_node = np.random.choice(range(v_size), p=probabilities)
            
            v_counts[v_node] += 1
            node_weights[v_node] += 0.5  # Increase weight (preferential attachment)
    
    # Make sure all nodes have at least one connection
    for i in range(len(v_counts)):
        if v_counts[i] == 0 and sum(u_sequence) < target_edges:
            v_counts[i] = 1
            # Find a u node with available capacity
            for j in range(len(u_sequence)):
                if u_sequence[j] < v_size:
                    u_sequence[j] += 1
                    break
    
    # Rebalance if needed
    while sum(v_counts) != sum(u_sequence):
        if sum(v_counts) < sum(u_sequence):
            # Add to V
            idx = np.random.randint(0, len(v_counts))
            v_counts[idx] += 1
        else:
            # Remove from V
            candidates = [i for i, count in enumerate(v_counts) if count > 1]
            if candidates:
                idx = np.random.choice(candidates)
                v_counts[idx] -= 1
    
    # Create the full degree sequence
    degree_sequence = u_sequence + v_counts
    
    # Verify and return
    if sequenceVerifier(degree_sequence, u_size, v_size, False):
        return degree_sequence
    else:
        # If verification fails, try recursively with slightly adjusted parameters
        print("Generated sequence invalid, retrying with adjusted parameters...")
        return generate_complex_network_degree_sequence(u_size, v_size, density*0.95, power_law_exp*1.05)

# Main execution when run as a script
if __name__ == "__main__":
    print("Network Graph Generator Visualizer")
    print("==================================")
    
    # Example 1: Small network
    print("\nExample 1: Small network")
    degree_sequence_small = [2, 1, 1, 2, 1, 1]
    u_size_small, v_size_small = 3, 3
    
    if sequenceVerifier(degree_sequence_small, u_size_small, v_size_small):
        edges_small = edgeAssignment(degree_sequence_small, u_size_small, v_size_small)
        
        # Create network evolution animation
        create_network_evolution_animation(degree_sequence_small, u_size_small, v_size_small, 
                                         20, "small_network_evolution_animation")
        
        # Visualize permutation stability
        visualize_permutation_stability(degree_sequence_small, u_size_small, v_size_small, 
                                      30, "small_permutation_stability")
    
    # Example 2: Medium network
    print("\nExample 2: Medium network")
    degree_sequence_medium = [3, 2, 2, 1, 2, 3, 1, 2]
    u_size_medium, v_size_medium = 4, 4
    
    if sequenceVerifier(degree_sequence_medium, u_size_medium, v_size_medium):
        edges_medium = edgeAssignment(degree_sequence_medium, u_size_medium, v_size_medium)
        
        # Create network evolution animation
        create_network_evolution_animation(degree_sequence_medium, u_size_medium, v_size_medium, 
                                         20, "medium_network_evolution_animation")
        
        # Visualize permutation stability
        visualize_permutation_stability(degree_sequence_medium, u_size_medium, v_size_medium, 
                                      30, "medium_permutation_stability")
    
    # Example 3: Large complex network
    print("\nExample 3: Large complex network")
    u_size_large, v_size_large = 20, 20
    
    # Generate a complex network degree sequence
    print("Generating complex network degree sequence...")
    degree_sequence_large = generate_complex_network_degree_sequence(u_size_large, v_size_large, 0.2, 2.5)
    
    print(f"Generated degree sequence: {degree_sequence_large}")
    print(f"U sequence: {degree_sequence_large[:u_size_large]}")
    print(f"V sequence: {degree_sequence_large[u_size_large:]}")
    
    if sequenceVerifier(degree_sequence_large, u_size_large, v_size_large, False):
        edges_large = edgeAssignment(degree_sequence_large, u_size_large, v_size_large, False)
        
        print("Generating large network evolution animation...")
        # Create network evolution animation with more frames for the complex network
        create_network_evolution_animation(degree_sequence_large, u_size_large, v_size_large, 
                                         30, "very_large_complex_network_evolution_animation")
        
        print("Generating large network stability visualization...")
        # Visualize permutation stability with more shuffles for better statistics
        visualize_permutation_stability(degree_sequence_large, u_size_large, v_size_large, 
                                      100, "very_large_complex_network_stability")
    
    print("\nAll visualizations and animations have been generated.")

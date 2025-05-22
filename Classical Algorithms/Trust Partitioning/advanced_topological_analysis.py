#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_topological_analysis.py

This script provides advanced topological analysis functions for comparing
different types of networks and visualizing their topological characteristics.
It builds on the topologicalPartitioner module and provides more detailed
analysis and comparison capabilities.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import traceback
from tqdm import tqdm
import warnings
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from topologicalPartitioner import (
    compute_betti_numbers, 
    compute_persistent_homology_sequence,
    construct_distance_matrix,
    extract_topological_features,
    plot_persistence_diagram,
    plot_betti_curves
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topo_network_animations')
os.makedirs(output_dir, exist_ok=True)

#########################
# Network Generation Functions
#########################

def generate_random_network(n_nodes=20, p=0.15):
    """Generate an Erdős–Rényi random network"""
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G

def generate_scale_free_network(n_nodes=20, m=2):
    """Generate a Barabási–Albert scale-free network"""
    G = nx.barabasi_albert_graph(n_nodes, m)
    return G

def generate_small_world_network(n_nodes=20, k=4, p=0.1):
    """Generate a Watts–Strogatz small-world network"""
    G = nx.watts_strogatz_graph(n_nodes, k, p)
    return G

def generate_ring_network(n_nodes=20):
    """Generate a ring network with high β₁"""
    G = nx.cycle_graph(n_nodes)
    return G

def generate_complete_network(n_nodes=20):
    """Generate a complete network (high connectivity)"""
    G = nx.complete_graph(n_nodes)
    return G

def generate_clustered_network(n_clusters=3, n_nodes_per_cluster=7, p_intra=0.7, p_inter=0.05):
    """Generate a network with clear cluster structure"""
    G = nx.Graph()
    
    # Add nodes
    node_id = 0
    clusters = []
    
    for i in range(n_clusters):
        cluster = list(range(node_id, node_id + n_nodes_per_cluster))
        clusters.append(cluster)
        G.add_nodes_from(cluster)
        node_id += n_nodes_per_cluster
    
    # Add intra-cluster edges (high probability)
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                if np.random.random() < p_intra:
                    G.add_edge(cluster[i], cluster[j])
    
    # Add inter-cluster edges (low probability)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for node_i in clusters[i]:
                for node_j in clusters[j]:
                    if np.random.random() < p_inter:
                        G.add_edge(node_i, node_j)
    
    return G

#########################
# Advanced Topological Analysis
#########################

def compute_topological_signature(G, max_dimension=2, filtration_steps=20):
    """
    Compute a topological signature for a network
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph
    max_dimension : int
        Maximum homology dimension
    filtration_steps : int
        Number of filtration steps
        
    Returns:
    --------
    dict : Dictionary with topological signature
    """
    # Convert graph to distance matrix
    # We use shortest path distance as our metric
    try:
        dist_matrix = np.zeros((len(G), len(G)))
        for i in range(len(G)):
            for j in range(len(G)):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    try:
                        dist_matrix[i, j] = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        dist_matrix[i, j] = np.inf
        
        # Normalize distances to [0,1]
        finite_dists = dist_matrix[~np.isinf(dist_matrix)]
        if len(finite_dists) > 0:
            max_dist = np.max(finite_dists)
            if max_dist > 0:
                dist_matrix[~np.isinf(dist_matrix)] /= max_dist
            dist_matrix[np.isinf(dist_matrix)] = 2.0  # Set infinite distances to a value beyond the normalized range
    except:
        # Fallback to adjacency-based distance if shortest path fails
        adj_matrix = nx.to_numpy_array(G)
        dist_matrix = 1 - adj_matrix  # Convert adjacency to distance (1 = no edge, 0 = edge)
        np.fill_diagonal(dist_matrix, 0)  # Set diagonal to 0 (distance to self)
    
    # Compute persistent homology
    ph_result = compute_persistent_homology_sequence(dist_matrix, max_dimension, filtration_steps)
    
    # Extract topological features at multiple filtration values
    features = {}
    filtration_values = ph_result['filtration_values']
    betti_seq = ph_result['betti_sequences']
    
    # Store whole sequences
    features['filtration_values'] = filtration_values
    for dim in range(max_dimension + 1):
        features[f'betti{dim}_sequence'] = betti_seq[f'betti{dim}']
    
    # Compute statistics of Betti numbers
    for dim in range(max_dimension + 1):
        seq = betti_seq[f'betti{dim}']
        features[f'betti{dim}_max'] = np.max(seq)
        features[f'betti{dim}_mean'] = np.mean(seq)
        features[f'betti{dim}_std'] = np.std(seq)
    
    # Compute persistence at fixed filtration values
    checkpoints = [0.1, 0.3, 0.5, 0.7, 0.9]
    for eps in checkpoints:
        # Find closest filtration value
        idx = np.argmin(np.abs(filtration_values - eps))
        for dim in range(max_dimension + 1):
            features[f'betti{dim}_at_{eps}'] = betti_seq[f'betti{dim}'][idx]
    
    return features

def compute_topological_distance(sig1, sig2, max_dimension=2):
    """
    Compute distance between two topological signatures
    
    Parameters:
    -----------
    sig1, sig2 : dict
        Topological signatures
    max_dimension : int
        Maximum dimension to consider
        
    Returns:
    --------
    float : Distance between signatures
    """
    distance = 0
    weights = {0: 1.0, 1: 1.5, 2: 2.0}  # Higher weight for higher dimensions
    
    # Compare Betti sequences with dynamic time warping
    for dim in range(max_dimension + 1):
        seq1 = sig1[f'betti{dim}_sequence']
        seq2 = sig2[f'betti{dim}_sequence']
        
        # Simple L2 distance between sequences
        l2_dist = np.sqrt(np.sum((np.array(seq1) - np.array(seq2))**2))
        distance += weights[dim] * l2_dist
    
    # Compare statistics
    stat_keys = ['max', 'mean', 'std']
    for dim in range(max_dimension + 1):
        for stat in stat_keys:
            key = f'betti{dim}_{stat}'
            diff = np.abs(sig1[key] - sig2[key])
            distance += 0.1 * weights[dim] * diff
    
    return distance

def analyze_network_collection(network_generators, names, max_dimension=2):
    """
    Analyze a collection of networks and compare their topological properties
    
    Parameters:
    -----------
    network_generators : list
        List of functions that generate networks
    names : list
        Names corresponding to the network generators
    max_dimension : int
        Maximum homology dimension
        
    Returns:
    --------
    dict : Analysis results
    """
    results = {
        'networks': [],
        'signatures': [],
        'names': names
    }
    
    print("Generating and analyzing networks...")
    for i, generator in enumerate(tqdm(network_generators)):
        G = generator()
        results['networks'].append(G)
        
        # Compute topological signature
        signature = compute_topological_signature(G, max_dimension)
        results['signatures'].append(signature)
    
    # Compute distance matrix between all networks
    n_networks = len(results['networks'])
    distance_matrix = np.zeros((n_networks, n_networks))
    
    print("Computing topological distances...")
    for i in range(n_networks):
        for j in range(i, n_networks):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                dist = compute_topological_distance(
                    results['signatures'][i], 
                    results['signatures'][j],
                    max_dimension
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    results['distance_matrix'] = distance_matrix
    
    return results

#########################
# Advanced Visualization Functions
#########################

def visualize_betti_curves_comparison(results, highlight_dims=[0, 1]):
    """
    Create a comparison of Betti curves for different networks
    
    Parameters:
    -----------
    results : dict
        Results from analyze_network_collection
    highlight_dims : list
        Dimensions to highlight
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    n_networks = len(results['networks'])
    n_dims = len(highlight_dims)
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims))
    if n_dims == 1:
        axes = [axes]
    
    # Color map for different networks
    colors = plt.cm.tab10.colors
    
    # Plot each dimension
    for dim_idx, dim in enumerate(highlight_dims):
        ax = axes[dim_idx]
        
        for i, signature in enumerate(results['signatures']):
            name = results['names'][i]
            filtration = signature['filtration_values']
            betti_seq = signature[f'betti{dim}_sequence']
            
            ax.plot(filtration, betti_seq, 
                    label=name, color=colors[i % len(colors)], linewidth=2)
        
        ax.set_xlabel('Filtration Value (ε)')
        ax.set_ylabel(f'Betti-{dim} (β{dim})')
        ax.set_title(f'Betti-{dim} Curves Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig

def visualize_distance_matrix(results):
    """
    Visualize the topological distance matrix as a heatmap
    
    Parameters:
    -----------
    results : dict
        Results from analyze_network_collection
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    distance_matrix = results['distance_matrix']
    names = results['names']
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(distance_matrix, annot=True, cmap=cmap, 
                xticklabels=names, yticklabels=names, ax=ax)
    
    ax.set_title('Topological Distance Matrix')
    plt.tight_layout()
    
    return fig

def visualize_network_clustering(results):
    """
    Visualize hierarchical clustering of networks based on topology
    
    Parameters:
    -----------
    results : dict
        Results from analyze_network_collection
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    distance_matrix = results['distance_matrix']
    names = results['names']
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method='ward')
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=names, leaf_rotation=90, ax=ax)
    
    ax.set_title('Hierarchical Clustering of Networks by Topological Structure')
    ax.set_xlabel('Network Type')
    ax.set_ylabel('Distance')
    
    plt.tight_layout()
    return fig

def visualize_networks_with_topology(results, max_dimension=2):
    """
    Create a grid of network visualizations colored by topological features
    
    Parameters:
    -----------
    results : dict
        Results from analyze_network_collection
    max_dimension : int
        Maximum homology dimension
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    networks = results['networks']
    signatures = results['signatures']
    names = results['names']
    
    # Create a grid of network visualizations
    n_networks = len(networks)
    n_cols = min(3, n_networks)
    n_rows = (n_networks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_networks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color map for topological features
    for i, G in enumerate(networks):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get node centrality as a proxy for topological importance
        try:
            # Beta centrality captures both direct and indirect connections
            centrality = nx.eigenvector_centrality_numpy(G)
        except:
            # Fallback to degree centrality if eigen fails
            centrality = nx.degree_centrality(G)
        
        # Get node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on centrality
        node_colors = [centrality[node] for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              cmap=plt.cm.plasma, node_size=80, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        
        # Add network name and Betti numbers
        sig = signatures[i]
        betti_str = ", ".join([f"β{d}={sig[f'betti{d}_at_0.5']}" for d in range(max_dimension+1)])
        ax.set_title(f"{names[i]}\n{betti_str}")
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n_networks, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_3d_persistence_landscape(signature, dim=1, n_landscapes=5):
    """
    Create a 3D visualization of persistence landscapes
    
    Parameters:
    -----------
    signature : dict
        Topological signature
    dim : int
        Homology dimension to visualize
    n_landscapes : int
        Number of landscape functions to compute
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Create persistence landscape from a betti sequence (simplified approach)
    filtration = signature['filtration_values']
    betti_seq = signature[f'betti{dim}_sequence']
    
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mesh grid for the landscape
    x = np.array(filtration)
    y = np.arange(1, n_landscapes + 1)
    X, Y = np.meshgrid(x, y)
    
    # Create landscape functions (simplified)
    Z = np.zeros_like(X)
    for i in range(n_landscapes):
        # Higher landscapes get progressively smaller values
        Z[i, :] = np.array(betti_seq) / (i + 1)
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # Add labels
    ax.set_xlabel('Filtration Value (ε)')
    ax.set_ylabel('Landscape Level (k)')
    ax.set_zlabel(f'Persistence Landscape Value')
    ax.set_title(f'Persistence Landscape for H{dim}')
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig

#########################
# Main Analysis Function
#########################

def run_network_topology_comparison():
    """
    Run a comprehensive analysis comparing topological properties of different network types
    """
    print("Starting advanced topological analysis of different network types...")
    
    # Define network generators
    network_generators = [
        generate_random_network,
        generate_scale_free_network,
        generate_small_world_network,
        generate_ring_network,
        generate_complete_network,
        generate_clustered_network
    ]
    
    names = [
        "Random (Erdős–Rényi)",
        "Scale-Free (Barabási–Albert)",
        "Small-World (Watts–Strogatz)",
        "Ring Network",
        "Complete Network",
        "Clustered Network"
    ]
    
    # Run analysis
    results = analyze_network_collection(network_generators, names)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Compare Betti curves
    fig_betti = visualize_betti_curves_comparison(results, highlight_dims=[0, 1])
    output_path = os.path.join(output_dir, "betti_curves_comparison.png")
    fig_betti.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_betti)
    
    # 2. Visualize distance matrix
    fig_dist = visualize_distance_matrix(results)
    output_path = os.path.join(output_dir, "topological_distance_matrix.png")
    fig_dist.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_dist)
    
    # 3. Visualize network clustering
    fig_cluster = visualize_network_clustering(results)
    output_path = os.path.join(output_dir, "network_topology_clustering.png")
    fig_cluster.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cluster)
    
    # 4. Visualize networks with topology
    fig_networks = visualize_networks_with_topology(results)
    output_path = os.path.join(output_dir, "networks_with_topology.png")
    fig_networks.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_networks)
    
    # 5. Create 3D persistence landscape for the first network
    try:
        fig_landscape = create_3d_persistence_landscape(results['signatures'][0], dim=1)
        output_path = os.path.join(output_dir, "persistence_landscape_3d.png")
        fig_landscape.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_landscape)
    except Exception as e:
        print(f"Warning: Could not create 3D persistence landscape: {e}")
    
    print("Analysis complete. Results saved to:", output_dir)
    return results

if __name__ == "__main__":
    try:
        print("Advanced Topological Network Analysis")
        print("------------------------------------")
        
        # Run the main analysis
        results = run_network_topology_comparison()
        
        print("\nProcess complete! Check the output directory for visualizations.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()

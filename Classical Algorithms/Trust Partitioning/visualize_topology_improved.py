#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_topology_improved.py

This script demonstrates improved topological analysis and visualization using the topologicalPartitioner
module. It provides enhanced error handling and visualization options for analyzing network topology.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import traceback
from topologicalPartitioner import (
    compute_betti_numbers, 
    compute_persistent_homology_sequence,
    construct_distance_matrix,
    extract_topological_features,
    plot_persistence_diagram,
    plot_betti_curves,
    topological_partitioner
)

# Set output directory to the current folder (Trust Partitioning)
output_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure the directory exists (it should already)
os.makedirs(output_dir, exist_ok=True)

def visualize_network_with_topology_robust(G, pos, topo_features, title="Network with Topological Features"):
    """
    Visualize a network with topological features highlighted, with improved error handling
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph to visualize
    pos : dict
        Dictionary of node positions
    topo_features : dict
        Dictionary of topological features
    title : str
        Title for the visualization
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with network visualization
    """
    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "Empty Graph - No Nodes to Display", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    # Get node centrality as a proxy for topological importance
    try:
        node_importance = dict(nx.degree(G))
    except:
        # Fallback to simple degree if centrality calculation fails
        node_importance = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    try:
        # Get edge betweenness as a proxy for topological significance of edges
        edge_betweenness = nx.edge_betweenness_centrality(G)
    except:
        # Fallback to simple weight if betweenness calculation fails
        edge_betweenness = {edge: G.edges[edge].get('weight', 1.0) 
                           for edge in G.edges()}
    
    # Create node color map based on topological importance
    node_colors = [node_importance[node] for node in G.nodes()]
    
    # Safety check for empty edge list
    if G.edges():
        # Create edge color map based on edge betweenness
        edge_colors = [edge_betweenness[edge] for edge in G.edges()]
        # Calculate edge widths
        edge_widths = [1 + 5 * edge_betweenness[edge] for edge in G.edges()]
    else:
        edge_colors = []
        edge_widths = []
    
    # Draw the network
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, 
                                  cmap=plt.cm.plasma, alpha=0.8, ax=ax)
    
    # Draw edges with varying width based on edge betweenness if there are edges
    if G.edges():
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                              edge_cmap=plt.cm.viridis, alpha=0.6, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
    
    # Add Betti numbers as title information
    ax.set_title(f"{title}\nBetti Numbers: β₀={topo_features['betti0']}, " +
                f"β₁={topo_features['betti1']}, β₂={topo_features['betti2']}")
    
    # Add a colorbar for node importance if there are nodes
    if G.nodes() and len(set(node_colors)) > 1:
        try:
            # Explicitly set the colormap normalization
            vmin = min(node_colors)
            vmax = max(node_colors)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            
            # Create a ScalarMappable with the colormap and normalization
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
            sm.set_array([])
            
            # Create colorbar with explicit axes reference
            cbar = fig.colorbar(sm, ax=ax, label="Node Importance")
        except Exception as e:
            print(f"Warning: Could not create colorbar: {e}")
    
    ax.axis('off')
    return fig

def generate_example_networks_with_topology(n_types=3, n_nodes_per_network=6, noise_level=0.1):
    """
    Generate different types of networks with interesting topological properties
    
    Parameters:
    -----------
    n_types : int
        Number of different network types to generate
    n_nodes_per_network : int
        Approximate number of nodes per network
    noise_level : float
        Level of noise to add to trust values
    
    Returns:
    --------
    dict : Dictionary with networks, trust matrix, node security, and network security
    """
    # Initialize networks container
    networks = []
    node_counter = 0
    
    # Network type 1: Dense cluster (high β₀, low β₁)
    n_nodes1 = n_nodes_per_network
    network1 = list(range(node_counter, node_counter + n_nodes1))
    networks.append(network1)
    node_counter += n_nodes1
    
    # Network type 2: Ring structure (high β₁)
    n_nodes2 = n_nodes_per_network
    network2 = list(range(node_counter, node_counter + n_nodes2))
    networks.append(network2)
    node_counter += n_nodes2
    
    # Network type 3: Multiple components (high β₀)
    n_nodes3 = n_nodes_per_network
    network3 = list(range(node_counter, node_counter + n_nodes3))
    networks.append(network3)
    node_counter += n_nodes3
    
    # Total number of nodes
    n_total = node_counter
    
    # Initialize trust matrix with random noise
    trust_matrix = np.random.rand(n_total, n_total) * noise_level
    np.fill_diagonal(trust_matrix, 0)  # No trust to self
    
    # Modify trust values to reflect network topologies
    
    # Network 1: Dense cluster - high trust within network
    for i in network1:
        for j in network1:
            if i != j:
                trust_matrix[i, j] = 0.7 + 0.3 * np.random.rand()
    
    # Network 2: Ring structure - trust between adjacent nodes
    for idx, i in enumerate(network2):
        next_idx = (idx + 1) % len(network2)
        next_node = network2[next_idx]
        trust_matrix[i, next_node] = 0.8 + 0.2 * np.random.rand()
        trust_matrix[next_node, i] = 0.8 + 0.2 * np.random.rand()
    
    # Network 3: Multiple components - create several disconnected groups
    group_size = max(2, n_nodes3 // 3)
    for g in range(0, n_nodes3, group_size):
        group = network3[g:min(g+group_size, n_nodes3)]
        for i in group:
            for j in group:
                if i != j:
                    trust_matrix[i, j] = 0.6 + 0.4 * np.random.rand()
    
    # Define security parameters
    node_security = {i: np.random.uniform(0.1, 0.4) for i in range(n_total)}
    
    # Define network security parameters
    network_security = {}
    for network_idx, network in enumerate(networks):
        # Get security values for nodes in this network
        security_values = [node_security[node] for node in network]
        # Use most lenient security among top n/2 nodes
        sorted_security = sorted(security_values)
        most_lenient = np.max(sorted_security[len(sorted_security) // 2:])
        network_security[network_idx] = most_lenient
    
    return {
        'networks': networks,
        'trust_matrix': trust_matrix,
        'node_security': node_security,
        'network_security': network_security
    }

def analyze_topology(example_data):
    """
    Perform topological analysis and return detailed information
    
    Parameters:
    -----------
    example_data : dict
        Dictionary with networks, trust matrix, and security parameters
        
    Returns:
    --------
    dict : Dictionary with topological analysis results
    """
    try:
        networks = example_data['networks']
        trust_matrix = example_data['trust_matrix']
        
        # Convert to distance matrix
        distance_matrix = construct_distance_matrix(trust_matrix)
        
        # Compute persistent homology with multiple filtration values
        print("Computing persistent homology...")
        ph_sequence = compute_persistent_homology_sequence(distance_matrix, max_dimension=2, filtration_steps=20)
        
        # Compute Betti numbers at different filtration values
        filtration_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        betti_results = {}
        for fv in filtration_values:
            betti_results[fv] = compute_betti_numbers(distance_matrix, max_dimension=2, max_edge_length=fv)
        
        # Extract topological features
        features = extract_topological_features(distance_matrix, max_dimension=2)
        
        # Create a network graph for visualization
        G = nx.Graph()
        
        # Add all nodes
        all_nodes = [node for network in networks for node in network]
        G.add_nodes_from(all_nodes)
        
        # Add weighted edges based on trust matrix
        n = len(all_nodes)
        for i in range(n):
            for j in range(i+1, n):
                # Average the trust in both directions
                trust_val = (trust_matrix[i, j] + trust_matrix[j, i]) / 2
                if trust_val > 0.3:  # Only add edges with significant trust
                    G.add_edge(all_nodes[i], all_nodes[j], weight=trust_val)
        
        # Calculate position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Compute topological features for each individual network
        network_topo_features = {}
        
        for i, network in enumerate(networks):
            if len(network) < 2:
                # Can't compute topology for networks with < 2 nodes
                network_topo_features[i] = {
                    'betti0': 1 if network else 0,
                    'betti1': 0,
                    'betti2': 0
                }
                continue
                
            # Extract submatrix for this network
            indices = [all_nodes.index(node) for node in network]
            submatrix = distance_matrix[np.ix_(indices, indices)]
            
            # Compute features
            try:
                network_features = extract_topological_features(submatrix, max_dimension=2)
                network_topo_features[i] = network_features
            except Exception as e:
                print(f"Warning: Could not compute topology for network {i}: {e}")
                network_topo_features[i] = {
                    'betti0': 1 if network else 0,
                    'betti1': 0,
                    'betti2': 0
                }
        
        return {
            'G': G,
            'pos': pos,
            'ph_sequence': ph_sequence,
            'betti_results': betti_results,
            'features': features,
            'network_features': network_topo_features
        }
        
    except Exception as e:
        print(f"Error analyzing topology: {str(e)}")
        traceback.print_exc()
        return None

def visualize_topology_results(results, example_data):
    """
    Generate visualizations for the topological analysis
    
    Parameters:
    -----------
    results : dict
        Dictionary with topological analysis results
    example_data : dict
        Dictionary with original networks and trust data
        
    Returns:
    --------
    list : List of output file paths
    """
    output_files = []
    
    try:
        G = results['G']
        pos = results['pos']
        features = results['features']
        ph_sequence = results['ph_sequence']
        betti_results = results['betti_results']
        networks = example_data['networks']
        
        # 1. Plot persistence diagram (for filtration = 0.5)
        medium_filtration = 0.5
        persistence_diagram = plot_persistence_diagram(
            betti_results[medium_filtration]['persistence'], 
            max_dimension=2, 
            title=f"Persistence Diagram (ε={medium_filtration})"
        )
        output_path = os.path.join(output_dir, "persistence_diagram_improved.png")
        persistence_diagram.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(persistence_diagram)
        output_files.append(output_path)
        
        # 2. Plot Betti curves
        betti_curves = plot_betti_curves(
            ph_sequence['betti_sequences'], 
            ph_sequence['filtration_values'], 
            title="Betti Curves of Network Trust"
        )
        output_path = os.path.join(output_dir, "betti_curves_improved.png")
        betti_curves.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(betti_curves)
        output_files.append(output_path)
        
        # 3. Visualize network with topological features
        if len(G.nodes()) > 0:
            network_viz = visualize_network_with_topology_robust(
                G, pos, features, 
                title="Network Trust with Topological Features"
            )
            output_path = os.path.join(output_dir, "network_topology_improved.png")
            network_viz.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(network_viz)
            output_files.append(output_path)
        
        # 4. Generate a visualization of each network with its own topology
        fig, axes = plt.subplots(1, len(networks), figsize=(16, 5))
        if len(networks) == 1:
            axes = [axes]  # Handle the special case of only one network
            
        for i, network in enumerate(networks):
            ax = axes[i]
            if network:
                subG = G.subgraph(network)
                subpos = {node: pos[node] for node in subG.nodes()}
                
                # Draw this network
                nx.draw_networkx_nodes(subG, subpos, ax=ax, node_size=100, 
                                      alpha=0.8, node_color='skyblue')
                nx.draw_networkx_edges(subG, subpos, ax=ax, alpha=0.6)
                nx.draw_networkx_labels(subG, subpos, ax=ax, font_size=8)
                
                # Add Betti numbers
                if i in results['network_features']:
                    features = results['network_features'][i]
                    ax.set_title(f"Network {i+1}\nβ₀={features['betti0']}, " +
                                f"β₁={features['betti1']}, β₂={features['betti2']}")
                else:
                    ax.set_title(f"Network {i+1}")
            else:
                ax.text(0.5, 0.5, "Empty Network", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f"Network {i+1}")
            
            ax.set_axis_off()
        
        # Adjust layout
        plt.tight_layout()
        output_path = os.path.join(output_dir, "network_subgraphs.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_files.append(output_path)
        
        # 5. Generate summary visualization with all networks
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw each subnetwork with different colors
        for i, network in enumerate(networks):
            if network:  # Only process non-empty networks
                subG = G.subgraph(network)
                color = plt.cm.tab10(i % 10)
                nx.draw_networkx_nodes(subG, pos, node_color=[color]*len(subG), 
                                     node_size=100, label=f"Network {i+1}", alpha=0.8, ax=ax)
                
                # Draw edges within this subnetwork
                if subG.edges():
                    nx.draw_networkx_edges(subG, pos, edge_color=color, alpha=0.5, width=1.5, ax=ax)
        
        # Draw node labels
        if G.nodes():
            nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
        
        ax.set_title(f"Network Communities with Topological Features\nGlobal: β₀={features['betti0']}, " +
                  f"β₁={features['betti1']}, β₂={features['betti2']}")
        if networks:
            ax.legend()
        ax.set_axis_off()
        output_path = os.path.join(output_dir, "network_communities_improved.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_files.append(output_path)
        
        return output_files
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        traceback.print_exc()
        return output_files

def run_partitioning_experiment(example_data, topo_weights=[0.0, 0.5, 1.0]):
    """
    Run a topological partitioning experiment with different topology weights
    
    Parameters:
    -----------
    example_data : dict
        Dictionary with networks, trust matrix, and security parameters
    topo_weights : list
        List of topology weights to test
        
    Returns:
    --------
    dict : Dictionary with results for each weight
    """
    results = {}
    
    try:
        for weight in topo_weights:
            print(f"\n=== Running partitioning with topology weight = {weight} ===")
            
            # Run the partitioner
            new_networks, _, _, _, topo_results = topological_partitioner(
                example_data['networks'],
                example_data['trust_matrix'],
                example_data['node_security'],
                example_data['network_security'],
                max_dimension=2,
                topo_weight=weight
            )
            
            # Store results
            results[weight] = {
                'networks': new_networks,
                'topo_results': topo_results
            }
            
            # Print summary
            print(f"Original networks: {[len(net) for net in example_data['networks']]}")
            print(f"New networks: {[len(net) for net in new_networks]}")
            
            # Save a simple visualization of the new networks
            # Create a new graph for the partitioning result
            G = nx.Graph()
            all_nodes = [node for network in example_data['networks'] for node in network]
            G.add_nodes_from(all_nodes)
            
            # Add weighted edges based on trust matrix
            trust_matrix = example_data['trust_matrix']
            n = len(all_nodes)
            for i in range(n):
                for j in range(i+1, n):
                    # Average the trust in both directions
                    trust_val = (trust_matrix[i, j] + trust_matrix[j, i]) / 2
                    if trust_val > 0.3:  # Only add edges with significant trust
                        G.add_edge(all_nodes[i], all_nodes[j], weight=trust_val)
            
            # Calculate position layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create a visualization of the new networks
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw each subnetwork with different colors
            for i, network in enumerate(new_networks):
                if network:  # Only process non-empty networks
                    subG = G.subgraph(network)
                    color = plt.cm.tab10(i % 10)
                    nx.draw_networkx_nodes(subG, pos, node_color=[color]*len(subG), 
                                         node_size=100, label=f"Network {i+1}", alpha=0.8, ax=ax)
                    
                    # Draw edges within this subnetwork
                    if subG.edges():
                        nx.draw_networkx_edges(subG, pos, edge_color=color, alpha=0.5, width=1.5, ax=ax)
            
            # Draw node labels
            if G.nodes():
                nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
            
            ax.set_title(f"Network Communities after Partitioning (topo_weight={weight})")
            if new_networks:
                ax.legend()
            ax.set_axis_off()
            
            # Save figure
            output_path = os.path.join(output_dir, f"partitioning_weight_{weight}.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return results
    
    except Exception as e:
        print(f"Error in partitioning experiment: {str(e)}")
        traceback.print_exc()
        return results

def create_comparison_visualization(experiment_results, example_data):
    """
    Create a visualization comparing partitioning results with different weights
    
    Parameters:
    -----------
    experiment_results : dict
        Dictionary with results for each weight
    example_data : dict
        Dictionary with original data
        
    Returns:
    --------
    str : Path to the output file
    """
    try:
        # Create a single figure with multiple subplots
        n_weights = len(experiment_results)
        fig, axes = plt.subplots(1, n_weights, figsize=(16, 5))
        
        if n_weights == 1:
            axes = [axes]  # Handle the special case of only one weight
        
        # Create a graph with the original trust matrix
        G = nx.Graph()
        all_nodes = [node for network in example_data['networks'] for node in network]
        G.add_nodes_from(all_nodes)
        
        # Add weighted edges based on trust matrix
        trust_matrix = example_data['trust_matrix']
        n = len(all_nodes)
        for i in range(n):
            for j in range(i+1, n):
                # Average the trust in both directions
                trust_val = (trust_matrix[i, j] + trust_matrix[j, i]) / 2
                if trust_val > 0.3:  # Only add edges with significant trust
                    G.add_edge(all_nodes[i], all_nodes[j], weight=trust_val)
        
        # Calculate position layout (same for all subplots)
        pos = nx.spring_layout(G, seed=42)
        
        # Plot each result
        for i, (weight, result) in enumerate(sorted(experiment_results.items())):
            ax = axes[i]
            networks = result['networks']
            
            # Draw each subnetwork with different colors
            for j, network in enumerate(networks):
                if network:  # Only process non-empty networks
                    subG = G.subgraph(network)
                    color = plt.cm.tab10(j % 10)
                    nx.draw_networkx_nodes(subG, pos, node_color=[color]*len(subG), 
                                         node_size=100, label=f"Net {j+1}", alpha=0.8, ax=ax)
                    
                    # Draw edges within this subnetwork
                    if subG.edges():
                        nx.draw_networkx_edges(subG, pos, edge_color=color, alpha=0.5, width=1.5, ax=ax)
            
            # Draw node labels
            if G.nodes():
                nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
            
            ax.set_title(f"topo_weight = {weight}\n{len(networks)} networks")
            
            # Only show legend for the last subplot to save space
            if i == n_weights - 1 and networks:
                ax.legend()
                
            ax.set_axis_off()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "weight_comparison.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Improved Topological Network Visualization")
    print("------------------------------------------")
    
    try:
        # Generate example networks
        print("Generating example networks...")
        example_data = generate_example_networks_with_topology(n_types=3, n_nodes_per_network=8)
        
        # Analyze topology
        print("Analyzing topology...")
        topo_results = analyze_topology(example_data)
        
        if topo_results:
            # Generate visualizations
            print("Generating visualizations...")
            output_files = visualize_topology_results(topo_results, example_data)
            print(f"Created {len(output_files)} visualization files")
            
            # Run partitioning experiment
            print("Running partitioning experiment...")
            experiment_results = run_partitioning_experiment(
                example_data, 
                topo_weights=[0.0, 0.3, 0.7, 1.0]
            )
            
            # Create comparison visualization
            print("Creating comparison visualization...")
            comparison_path = create_comparison_visualization(experiment_results, example_data)
            if comparison_path:
                print(f"Comparison visualization saved to: {comparison_path}")
        
        print("\nProcess complete. Check the Trust Partitioning folder for results.")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        traceback.print_exc()
        print("\nProcess failed with errors.")

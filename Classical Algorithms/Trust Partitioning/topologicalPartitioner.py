#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topologicalPartitioner.py

This script implements persistent homology analysis for topological network partitioning.
It builds on the trustPartitioner.py framework by incorporating methods to calculate
Betti numbers and other topological features of networks, then partitions networks
based on this additional topological information.

The script also provides visualization capabilities for the topological features
and animations of the partitioning process.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from itertools import combinations
import gudhi as gd
from gudhi.simplex_tree import SimplexTree
from scipy.spatial.distance import pdist, squareform
import string
import time

# Set output directory to the current folder (Trust Partitioning)
output_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure the directory exists (it should already)
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Color schemes for different topological features
COLOR_SCHEMES = {
    'betti0': {'color': '#3a86ff', 'edge': '#0a58ca', 'name': 'Connected Components (Betti-0)'},
    'betti1': {'color': '#ff006e', 'edge': '#c30052', 'name': 'Cycles (Betti-1)'},
    'betti2': {'color': '#fb5607', 'edge': '#cc4600', 'name': 'Voids (Betti-2)'},
    'persistence': {'color': '#8338ec', 'edge': '#6423b3', 'name': 'Persistence'}
}

#########################
# Topological Functions
#########################

def construct_distance_matrix(trust_matrix):
    """
    Convert trust matrix to distance matrix for topological analysis
    Higher trust = lower distance (closer in topological space)
    """
    # Make sure trust values are in [0,1]
    trust_normalized = np.clip(trust_matrix, 0, 1)
    
    # Convert to distance: higher trust = lower distance
    # Add small epsilon to avoid zero distances
    distance_matrix = 1 - trust_normalized + 1e-10
    
    # Make sure the diagonal is zero (distance to self)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def compute_betti_numbers(distance_matrix, max_dimension=2, max_edge_length=1.0):
    """
    Compute Betti numbers using the Vietoris-Rips complex
    
    Parameters:
    -----------
    distance_matrix : np.array
        Distance matrix between nodes
    max_dimension : int
        Maximum homology dimension to compute
    max_edge_length : float
        Maximum filtration value
        
    Returns:
    --------
    dict : Dictionary with Betti numbers and persistence diagrams
    """
    # Initialize the Rips complex
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    
    # Create a simplex tree with simplices up to dimension max_dimension+1
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension+1)
    
    # Compute persistent homology
    persistence = simplex_tree.persistence()
    
    # Get persistence diagrams
    diagrams = simplex_tree.persistence_intervals_in_dimension
    
    # Initialize Betti numbers dictionary
    betti_numbers = {}
    persistence_dict = {}
    
    # Calculate Betti numbers for each dimension
    for dim in range(max_dimension+1):
        persistence_intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        
        # Count intervals that are still alive at max_edge_length
        # (birth <= max_edge_length and (death > max_edge_length or death == float('inf')))
        betti_count = sum(1 for (birth, death) in persistence_intervals 
                          if birth <= max_edge_length and (death > max_edge_length or death == float('inf')))
        
        betti_numbers[f'betti{dim}'] = betti_count
        persistence_dict[f'dim{dim}'] = persistence_intervals
    
    return {
        'betti_numbers': betti_numbers,
        'persistence': persistence_dict,
        'simplex_tree': simplex_tree
    }

def compute_persistent_homology_sequence(distance_matrix, max_dimension=2, filtration_steps=10):
    """
    Compute persistent homology at different filtration values
    
    Parameters:
    -----------
    distance_matrix : np.array
        Distance matrix between nodes
    max_dimension : int
        Maximum homology dimension to compute
    filtration_steps : int
        Number of filtration values to compute
        
    Returns:
    --------
    dict : Dictionary with Betti numbers sequences and filtration values
    """
    # Determine range of filtration values
    max_distance = np.max(distance_matrix)
    filtration_values = np.linspace(0, max_distance, filtration_steps)
    
    # Initialize results
    betti_sequences = {f'betti{dim}': [] for dim in range(max_dimension+1)}
    
    # Compute Betti numbers for each filtration value
    for epsilon in filtration_values:
        result = compute_betti_numbers(distance_matrix, max_dimension, epsilon)
        betti_nums = result['betti_numbers']
        
        for dim in range(max_dimension+1):
            betti_sequences[f'betti{dim}'].append(betti_nums[f'betti{dim}'])
    
    return {
        'betti_sequences': betti_sequences,
        'filtration_values': filtration_values
    }

def extract_topological_features(distance_matrix, max_dimension=2):
    """
    Extract key topological features from persistent homology
    
    Parameters:
    -----------
    distance_matrix : np.array
        Distance matrix between nodes
    max_dimension : int
        Maximum homology dimension to compute
        
    Returns:
    --------
    dict : Dictionary with topological features
    """
    # Compute persistent homology
    result = compute_betti_numbers(distance_matrix, max_dimension)
    
    # Extract persistence diagrams
    persistence_dict = result['persistence']
    
    # Calculate persistence metrics
    features = {}
    
    # Add Betti numbers
    features.update(result['betti_numbers'])
      # Calculate total persistence for each dimension
    for dim in range(max_dimension+1):
        intervals = persistence_dict[f'dim{dim}']
        
        # Total persistence (sum of all persistence lengths)
        if len(intervals) > 0:
            finite_intervals = [(birth, death) for birth, death in intervals if death != float('inf')]
            if finite_intervals:
                total_persistence = sum(death - birth for birth, death in finite_intervals)
                features[f'total_persistence_dim{dim}'] = total_persistence
            else:
                features[f'total_persistence_dim{dim}'] = 0
        else:
            features[f'total_persistence_dim{dim}'] = 0
            
        # Maximum persistence
        if len(intervals) > 0:
            finite_intervals = [(birth, death) for birth, death in intervals if death != float('inf')]
            if finite_intervals:
                max_persistence = max((death - birth) for birth, death in finite_intervals)
                features[f'max_persistence_dim{dim}'] = max_persistence
            else:
                features[f'max_persistence_dim{dim}'] = 0
        else:
            features[f'max_persistence_dim{dim}'] = 0
    
    return features

#########################
# Network Partitioning based on Topology
#########################

def topological_partitioner(input_networks, input_trust, nodes_security, networks_security, 
                           max_dimension=2, topo_weight=0.5):
    """
    Partition networks based on trust relationships and topological features
    
    Parameters:
    -----------
    input_networks : list
        List of networks, where each network is a list of node indices
    input_trust : np.array
        Trust matrix between all nodes
    nodes_security : dict
        Security parameters for each node
    networks_security : dict
        Security parameters for each network
    max_dimension : int
        Maximum homology dimension to compute
    topo_weight : float
        Weight given to topological features vs trust (0.0-1.0)
        
    Returns:
    --------
    tuple : Updated networks, trust matrix, node security, network security, and topological features
    """
    # Initializing information of the networks provided
    networks_copy = [j.copy() for j in input_networks]
    network_sizes = [len(i) for i in input_networks]
    number_of_networks = len(network_sizes)
    network_names = list(string.ascii_uppercase)[:number_of_networks]
    network_empty_sets = {k:[] for k in range(np.sum(network_sizes))}

    # Printing initial configuration of the networks
    print('--- Original state of networks ---')
    for i in range(len(input_networks)):
        print('Initial network', network_names[i],':', input_networks[i])
    
    # Calculate distance matrix from trust matrix
    distance_matrix = construct_distance_matrix(input_trust)
    
    # Compute topological features for the entire network
    print('Computing topological features...')
    global_topo_features = extract_topological_features(distance_matrix, max_dimension)
    print(f"Global Betti numbers: β₀={global_topo_features['betti0']}, β₁={global_topo_features['betti1']}, β₂={global_topo_features['betti2']}")
    
    # Compute topological features for each individual network
    network_topo_features = {}
    network_distance_matrices = {}
    
    # Extract the submatrices for each network
    size_sum = lambda alpha: sum(network_sizes[i] for i in range(alpha))
    
    for i in range(number_of_networks):
        start_idx = size_sum(i)
        end_idx = size_sum(i+1)
        
        # Extract the submatrix for this network
        network_trust = input_trust[start_idx:end_idx, start_idx:end_idx]
        network_distance = construct_distance_matrix(network_trust)
        network_distance_matrices[i] = network_distance
        
        # Compute topological features
        network_topo_features[i] = extract_topological_features(network_distance, max_dimension)
        print(f"Network {network_names[i]} Betti numbers: β₀={network_topo_features[i]['betti0']}, " +
              f"β₁={network_topo_features[i]['betti1']}, β₂={network_topo_features[i]['betti2']}")
    
    # Extract block matrices for trust
    block_matrices = []
    for i in range(number_of_networks):
        for j in range(number_of_networks):
            block_matrices.append(input_trust[size_sum(i):size_sum(i+1), size_sum(j):size_sum(j+1)])
    
    # Compute trust values for nodes and networks
    # This reuses the original r function from trustPartitioner.py
    node_trust_values = []
    for i in block_matrices:
        node_trust_values.append(r(i))
    
    network_trust_values = []
    for i in block_matrices:
        network_trust_values.append(r(i, False))
    
    # Compute topological compatibility between networks
    topo_compatibility = np.zeros((number_of_networks, number_of_networks))
    
    for i in range(number_of_networks):
        for j in range(number_of_networks):
            if i == j:
                topo_compatibility[i, j] = 1.0  # Maximum compatibility with self
                continue
                
            # Calculate compatibility score based on difference in Betti numbers
            betti_diff = sum(abs(network_topo_features[i][f'betti{dim}'] - network_topo_features[j][f'betti{dim}']) 
                             for dim in range(max_dimension+1))
            
            # Normalize by maximum possible difference
            max_betti_diff = sum(max(network_topo_features[i][f'betti{dim}'], network_topo_features[j][f'betti{dim}']) 
                                for dim in range(max_dimension+1))
            
            if max_betti_diff > 0:
                normalized_diff = betti_diff / max_betti_diff
                topo_compatibility[i, j] = 1 - normalized_diff
            else:
                topo_compatibility[i, j] = 1.0
    
    # Combine trust and topological information for partitioning decisions
    optimal_locations = []
    
    for n in range(len(input_networks)):
        # Get trust information for current network
        network_length = len(input_networks[n])
        target_trusts = node_trust_values[n*len(input_networks):(n+1)*len(input_networks)]
        
        # Initialize jump locations
        jump_locations = []
        
        # Combine trust with topological compatibility
        for i in range(network_length):
            node_jump_scores = []
            
            for j in range(number_of_networks):
                # Get trust score from node i in network n to network j
                trust_score = target_trusts[j][i]
                
                # Get topological compatibility from network n to network j
                topo_score = topo_compatibility[n, j]
                
                # Combine scores with weight parameter
                combined_score = (1 - topo_weight) * trust_score + topo_weight * topo_score
                node_jump_scores.append(combined_score)
            
            jump_locations.append(node_jump_scores)
        
        # Find optimal location for each node
        optimal_locations_temp = [loc.index(max(loc)) for loc in jump_locations]
        optimal_locations.append(optimal_locations_temp)
    
    # Print optimal locations
    print('Topologically-informed optimal locations:', optimal_locations)
    
    # Perform potential jumps for every node in every network
    for n in range(len(input_networks)):
        # Get information for current network
        current_network = input_networks[n]
        current_copy = networks_copy[n]
        current_optimal = optimal_locations[n]
        
        # Define external networks trusts in this one
        external_trusts = []
        trust_extractor = lambda lst, q: [[lst[i] for i in range(start, len(lst), q)] for start in range(q)]
        external_trusts = trust_extractor(network_trust_values, number_of_networks)[n]
        
        # Iterate through nodes of the network
        for i in current_network:
            # Only look at nodes originally in network
            if i in current_copy:
                # Extract the index of node i
                node_index = current_network.index(i)
                
                # Select optimal location for node i in network n
                location = current_optimal[node_index]
                location_network = networks_copy[location]

                # Non-trivial location condition
                if location != n:
                    print(f'Node {current_network[node_index]} in network {network_names[n]} wants to jump to network {network_names[location]}!')
                    
                    # Verify trust condition
                    if external_trusts[location][node_index] <= networks_security[location]:
                        # Move node from one network to another
                        jump_node = current_copy.pop(current_copy.index(i))
                        location_network.insert(len(location_network), jump_node)
                        print(f'Node {current_network[node_index]} has jumped to network {network_names[location]}!')
    
    # Print the outcome of all potential jump events
    print('Networks after jumping events:', networks_copy)
    
    # Perform abandons for every network
    for n in range(len(input_networks)):
        # Get information for current network
        current_network = input_networks[n]
        current_copy = networks_copy[n]

        # Define the trusts of nodes in their own network
        internal_trusts = node_trust_values[n*(number_of_networks+1)]
        
        # Iterate through nodes of the network
        for i in current_network:
            # Only look at nodes originally in the network
            if i in current_copy:
                # Extract the index of node i
                node_index = current_network.index(i)
                
                # Check if trust condition is violated
                if internal_trusts[node_index] > nodes_security[i]:
                    # Move node i to an empty set
                    jump_node = current_copy.pop(current_copy.index(i))
                    network_empty_sets[i].insert(0, jump_node)
                    print(f'Node {i} in network {network_names[n]} has abandoned its network!')

    # Print the outcome of abandon events
    print('Networks after abandon events:', networks_copy) 
    print('Empty sets after evolution:', network_empty_sets)
    
    # Create new networks from old ones using the networkCombiner from trustPartitioner
    new_networks = networkCombiner(networks_copy, network_empty_sets)
    print('Updated networks:', new_networks)
    
    # Calculate topological features for the new network configuration
    new_topo_features = {}
    
    # Collect all topological features
    topo_results = {
        'global_features': global_topo_features,
        'original_network_features': network_topo_features,
        'new_networks': new_networks,
        'topo_compatibility': topo_compatibility
    }
    
    return new_networks, input_trust, nodes_security, networks_security, topo_results

#########################
# Visualization Functions
#########################

def plot_persistence_diagram(persistence_dict, max_dimension=2, title="Persistence Diagram"):
    """Plot persistence diagram for multiple dimensions"""
    plt.figure(figsize=(10, 8))
    
    # Different colors for different dimensions
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each dimension
    for dim in range(max_dimension+1):
        if f'dim{dim}' in persistence_dict:
            intervals = persistence_dict[f'dim{dim}']
              # Extract birth and death times
            births = [birth for birth, death in intervals if death != float('inf')]
            deaths = [death for birth, death in intervals if death != float('inf')]
            
            # Plot finite intervals
            if len(births) > 0:
                plt.scatter(births, deaths, c=colors[dim % len(colors)], 
                            alpha=0.8, label=f"Dimension {dim}")
            
            # Plot points with infinite death time at the top of the plot
            inf_births = [birth for birth, death in intervals if death == float('inf')]
            if len(inf_births) > 0:
                # Find maximum death time to set y-coordinate for inf points
                if len(deaths) > 0:
                    y_inf = max(deaths) * 1.1
                else:
                    y_inf = 1.0
                
                plt.scatter(inf_births, [y_inf] * len(inf_births), 
                           marker='^', c=colors[dim % len(colors)], alpha=0.8)
    
    # Add diagonal line
    lims = plt.gca().get_xlim()
    plt.plot(lims, lims, 'k--', alpha=0.3)
    
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_betti_curves(betti_sequences, filtration_values, title="Betti Curves"):
    """Plot Betti numbers as function of filtration parameter"""
    plt.figure(figsize=(10, 6))
    
    # Different colors for different dimensions
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each dimension
    for dim, key in enumerate(sorted(betti_sequences.keys())):
        plt.plot(filtration_values, betti_sequences[key], 
                 label=f"β{dim}", color=colors[dim % len(colors)], linewidth=2)
    
    plt.xlabel('Filtration Value (ε)')
    plt.ylabel('Betti Number')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def visualize_network_with_topology(G, pos, topo_features, title="Network with Topological Features"):
    """Visualize network with node colors representing topological features"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get node-level topological importance
    # For example, we can use node degree as a simple proxy
    node_importance = dict(nx.degree(G))
    
    # Get edge betweenness as a proxy for topological significance of edges
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Create node color map based on topological importance
    node_colors = [node_importance[node] for node in G.nodes()]
    
    # Create edge color map based on edge betweenness
    edge_colors = [edge_betweenness[edge] for edge in G.edges()]
    
    # Draw the network
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, 
                                  cmap=plt.cm.plasma, alpha=0.8, ax=ax)
    
    # Draw edges with varying width based on edge betweenness
    edge_widths = [1 + 5 * edge_betweenness[edge] for edge in G.edges()] if G.edges() else []
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                          edge_cmap=plt.cm.viridis, alpha=0.6, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
    
    # Add Betti numbers as title information
    ax.set_title(f"{title}\nBetti Numbers: β₀={topo_features['betti0']}, " +
                f"β₁={topo_features['betti1']}, β₂={topo_features['betti2']}")
    
    # Add a colorbar for node importance
    if G.nodes():
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Node Importance")
    
    ax.axis('off')
    return fig

def create_topological_partitioning_animation(input_networks, input_trust, network_evolution, topo_results, 
                                             filename="topological_partitioning.gif", colormap='tab10'):
    """
    Create an animation showing the topological partitioning process
    
    Parameters:
    -----------
    input_networks : list
        List of initial networks
    input_trust : np.array
        Trust matrix between nodes
    network_evolution : list
        List of network states during evolution
    topo_results : dict
        Dictionary with topological features
    filename : str
        Filename for the output animation
    colormap : str
        Name of the colormap to use (e.g., 'tab10', 'viridis', 'plasma')
    """# Setup figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Setup legend subplot on the left
    ax_legend = plt.subplot2grid((2, 5), (0, 0), rowspan=2)
    ax_legend.axis('off')  # Remove axes for legend area
    
    # Setup network visualization subplot in the middle
    ax1 = plt.subplot2grid((2, 5), (0, 1), colspan=3, rowspan=2)
    
    # Setup persistence diagram subplot on the right
    ax2 = plt.subplot2grid((2, 5), (0, 4))
    
    # Setup Betti curves subplot on the right
    ax3 = plt.subplot2grid((2, 5), (1, 4))
    
    # Create full graph from trust matrix
    full_G = nx.Graph()
    
    # Add all nodes
    all_nodes = [node for network in input_networks for node in network]
    full_G.add_nodes_from(all_nodes)
    
    # Add weighted edges based on trust matrix
    n = len(all_nodes)
    for i in range(n):
        for j in range(i+1, n):
            # Average the trust in both directions
            trust_val = (input_trust[i, j] + input_trust[j, i]) / 2
            if trust_val > 0:  # Only add edges with non-zero trust
                full_G.add_edge(all_nodes[i], all_nodes[j], weight=trust_val)
      # Calculate position layout once (for consistent visualization)
    # Use a smaller k value (0.4) for tighter, less spread out layout
    pos = nx.spring_layout(full_G, seed=42, k=0.4)
    
    # Convert distance matrix from trust
    distance_matrix = construct_distance_matrix(input_trust)
    
    # Compute persistent homology sequences
    ph_results = compute_persistent_homology_sequence(distance_matrix)
    betti_sequences = ph_results['betti_sequences']
    filtration_values = ph_results['filtration_values']
      # Function to update the animation at each frame
    def update(frame):
        # For the first 5 frames, stay on the initial configuration (pause at beginning)
        effective_frame = 0 if frame < 5 else frame - 5
        
        # Clear all axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax_legend.clear()
        
        # Get the current network configuration
        # Calculate frame position in the sequence
        frame_position = min(effective_frame, len(network_evolution)-1)
        
        # Add an interpolation for smoother transition
        if frame_position < len(network_evolution) - 1 and effective_frame > 0:
            # Calculate fractional part for interpolation
            frac = effective_frame - int(effective_frame)
            if frac > 0 and frac < 1:
                # If we're between two frames, calculate progress for interpolation
                current_networks = network_evolution[frame_position]
            else:
                current_networks = network_evolution[frame_position]
        else:
            current_networks = network_evolution[frame_position]
        
        # Create colored subnetworks
        subgraphs = []
        for i, network in enumerate(current_networks):
            if network:  # Only process non-empty networks
                subG = full_G.subgraph(network)
                subgraphs.append((subG, f"Network {i+1}"))
        
        # Get the appropriate colormap object
        cmap = plt.cm.get_cmap(colormap)
        
        # Draw each subnetwork with different colors
        for i, (subG, label) in enumerate(subgraphs):
            color = cmap(i % 10 if colormap == 'tab10' else i/max(1, len(subgraphs)-1))
            nx.draw_networkx_nodes(subG, pos, ax=ax1, node_color=[color]*len(subG), 
                                 node_size=100, label=label, alpha=0.8)
            
            # Draw edges within this subnetwork, check if edges exist first
            if len(subG.edges()) > 0:
                nx.draw_networkx_edges(subG, pos, ax=ax1, edge_color=color, alpha=0.5, width=1.5)
        
        # Draw node labels if graph is not empty
        if full_G.nodes():
            nx.draw_networkx_labels(full_G, pos, ax=ax1, font_size=8, font_family="sans-serif")
        
        # Add title with frame information
        if effective_frame == 0:
            ax1.set_title("Initial Network Configuration")
        else:
            # Calculate the filtration index based on the frame progression
            progress = frame / float(total_frames - 1)  # Normalize to 0-1 range
            filtration_idx = min(int(progress * len(filtration_values)), len(filtration_values)-1)
            # Show current progress and filtration value
            ax1.set_title(f"Network Evolution - Step {effective_frame} (Filtration: {filtration_values[filtration_idx]:.2f})")
        
        if subgraphs:  # Only add legend if there are networks to show
            # Create a separate legend in the dedicated legend subplot
            legend_handles = []
            legend_labels = []
            
            for i, (subG, label) in enumerate(subgraphs):
                color = cmap(i % 10 if colormap == 'tab10' else i/max(1, len(subgraphs)-1))
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=color, markersize=10, label=label))
                legend_labels.append(label)
            
            ax_legend.legend(handles=legend_handles, labels=legend_labels, 
                           loc='center', fontsize=12, frameon=False)
            ax_legend.set_title("Network Legend", fontsize=14)
        
        ax1.set_axis_off()
        
        # Update persistence diagram - show a subset of filtration values based on frame
        # Calculate the filtration index based on the frame progression (0 to 1)
        # Ensure we progress through all filtration values by the end
        progress = frame / float(total_frames - 1)  # Normalize to 0-1 range
        filtration_idx = min(int(progress * len(filtration_values)), len(filtration_values)-1)
        
        # Compute Betti numbers at this filtration value
        result = compute_betti_numbers(distance_matrix, max_dimension=2, 
                                      max_edge_length=filtration_values[filtration_idx])
        
        # Extract persistence information
        persistence_dict = result['persistence']
        
        # Plot persistence diagram
        colors = ['blue', 'red', 'green']
        for dim in range(3):  # Plot dimensions 0, 1, 2
            if f'dim{dim}' in persistence_dict:
                intervals = persistence_dict[f'dim{dim}']
                
                # Extract birth and death times
                births = [birth for birth, death in intervals if death != float('inf')]
                deaths = [death for birth, death in intervals if death != float('inf')]
                
                # Plot finite intervals
                if len(births) > 0:
                    ax2.scatter(births, deaths, c=colors[dim], alpha=0.7, label=f"Dimension {dim}")
                
                # Plot infinite intervals
                inf_births = [birth for birth, death in intervals if death == float('inf')]
                if len(inf_births) > 0:
                    max_val = 1.0 if len(deaths) == 0 else max(deaths) * 1.1
                    ax2.scatter(inf_births, [max_val] * len(inf_births), 
                              marker='^', c=colors[dim], alpha=0.7)
        
        # Add diagonal
        lims = ax2.get_xlim()
        ax2.plot(lims, lims, 'k--', alpha=0.3)
        
        ax2.set_xlabel('Birth')
        ax2.set_ylabel('Death')
        ax2.set_title(f"Persistence Diagram (ε={filtration_values[filtration_idx]:.2f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Betti curves with current filtration highlighted
        for dim in range(3):
            ax3.plot(filtration_values, betti_sequences[f'betti{dim}'], 
                   label=f"β{dim}", color=colors[dim], linewidth=2)
              # Highlight current filtration value
            current_betti = betti_sequences[f'betti{dim}'][filtration_idx]
            ax3.scatter([filtration_values[filtration_idx]], [current_betti], 
                      color=colors[dim], s=100, edgecolor='black', zorder=10)
        
        ax3.axvline(x=filtration_values[filtration_idx], color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Filtration Value (ε)')
        ax3.set_ylabel('Betti Number')
        ax3.set_title("Betti Curves")
        ax3.legend()
        ax3.grid(True, alpha=0.3)        
        return (ax_legend, ax1, ax2, ax3)
    
    # Create the animation with frames for the initial pause and intermediate steps
    total_frames = 20  # Increase number of frames to show more of the progression
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=500, blit=False)
    
    # Save the animation
    output_path = os.path.join(output_dir, filename)
    try:
        ani.save(output_path, writer='pillow', fps=2)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save animation: {e}")
        try:
            # Try saving a simpler version - just a sequence of images instead
            print("Trying to save static images instead...")
            for frame in range(len(network_evolution)):
                fig.clear()
                ax = fig.add_subplot(111)
                
                # Get current configuration
                current_networks = network_evolution[frame]
                
                # Draw subnetworks
                for i, network in enumerate(current_networks):
                    if network:
                        subG = full_G.subgraph(network)
                        color = plt.cm.tab10(i % 10)
                        nx.draw_networkx_nodes(subG, pos, ax=ax, node_color=[color]*len(subG), 
                                             node_size=100, label=f"Network {i+1}", alpha=0.8)
                        
                        if len(subG.edges()) > 0:
                            nx.draw_networkx_edges(subG, pos, ax=ax, edge_color=color, alpha=0.5, width=1.5)
                
                if full_G.nodes():
                    nx.draw_networkx_labels(full_G, pos, ax=ax, font_size=8, font_family="sans-serif")
                
                ax.set_title(f"Network Configuration - Step {frame}")
                if current_networks:
                    ax.legend(loc='upper right')
                ax.set_axis_off()
                
                # Save the frame
                frame_path = os.path.join(output_dir, f"network_frame_{frame}.png")
                fig.savefig(frame_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            print(f"Static frames saved to {output_dir}")
        except Exception as inner_e:
            print(f"Could not save static frames either: {inner_e}")
    
    plt.close(fig)
    return ani

#########################
# Helper Functions (from trustPartitioner.py)
#########################

# Import from trustPartitioner.py
def r(trustAinB, forward=True):
    """
    Compute trust values between nodes and networks.
    Imported from trustPartitioner.py
    """
    # Computing the r_i(B) values for all nodes i in A
    if forward == True:
        # Initializing the network and empty trust values
        networkSize = len(trustAinB) #|A|
        trustValues = np.zeros(networkSize)
        for i in range(networkSize):
            trustValues[i] =  1/(networkSize) * np.sum(trustAinB[i])
            # {r_i(B),...,r_k(B)}
             
    # Computing the r_A(j) values for all nodes j in B
    if forward == False:
        # Initializing the network and empty trust values
        networkSize = len(trustAinB.T) #|B|
        trustValues = np.zeros(networkSize)
        for j in range(networkSize):
            trustValues[j] =  1/(networkSize) * np.sum(trustAinB.T[j])
            # {r_A(l),...,r_A(m)}
    
    # Returning the desired trust array
    return trustValues

def networkCombiner(inputNetworks, inputEmptySets):
    """
    Combine networks and empty sets into a total ordered network.
    Imported from trustPartitioner.py
    """
    # Removing any empty sets
    emptyRemover = lambda set: [i for i in set if i]

    # Extracting values from empty sets
    emptyValues = sorted(list(inputEmptySets.values()))
    
    # Extracting non-empty networks
    extractedNetworks = emptyRemover(inputNetworks)
    extractedEmptySets = emptyRemover(emptyValues)
    
    # Combining them into a total collection of networks
    collectedNetworks = extractedNetworks + extractedEmptySets
    
    # Ordering individual networks
    networkOrdering = [sorted(i) for i in collectedNetworks]
    
    # Returning the combined ordered network
    return networkOrdering

#########################
# Main Test Function
#########################

if __name__ == "__main__":
    print("Topological Network Partitioner")
    print("---------------------------------")
    
    # Define example networks
    N = [0, 1, 2]  # Network N with nodes: i,j,k = 0,1,2
    M = [3, 4]     # Network M with nodes: l,m = 3,4
    P = [5, 6, 7]  # Network P with nodes: p,q,r = 5,6,7
    networks = [N, M, P]
    
    # Create a random trust matrix
    n_nodes = sum(len(network) for network in networks)
    trust_matrix = np.random.rand(n_nodes, n_nodes)
    np.fill_diagonal(trust_matrix, 0)  # No trust to self
    
    # Define security parameters
    node_security = {i: np.random.uniform(0.1, 0.5) for i in range(n_nodes)}
    
    # Define network security parameters
    network_security = {}
    for network_idx, network in enumerate(networks):
        # Get security values for nodes in this network
        security_values = [node_security[node] for node in network]
        # Use most lenient security among top n/2 nodes
        sorted_security = sorted(security_values)
        most_lenient = np.max(sorted_security[len(sorted_security) // 2:])
        network_security[network_idx] = most_lenient
    
    # Run the topological partitioner
    new_networks, _, _, _, topo_results = topological_partitioner(
        networks, trust_matrix, node_security, network_security, max_dimension=2, topo_weight=0.5
    )
    
    # Create a visualization of the partitioning
    network_evolution = [networks, new_networks]
    
    # Create animation
    _ = create_topological_partitioning_animation(
        networks, trust_matrix, network_evolution, topo_results, 
        filename="topological_partitioning.gif"
    )
    
    # Print a Betti number analysis summary
    print("\nTopological Analysis Summary:")
    print("----------------------------")
    global_features = topo_results["global_features"]
    print(f"Global Network: β₀={global_features['betti0']}, β₁={global_features['betti1']}, β₂={global_features['betti2']}")
    
    for network_idx, features in topo_results["original_network_features"].items():
        network_name = list(string.ascii_uppercase)[network_idx]
        print(f"Network {network_name}: β₀={features['betti0']}, β₁={features['betti1']}, β₂={features['betti2']}")
    
    print("\nPartitioning complete. The resulting networks should have more homogeneous topological features.")

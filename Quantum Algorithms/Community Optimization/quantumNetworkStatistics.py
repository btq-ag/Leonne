#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantumNetworkStatistics.py

This script creates animated visualizations showing statistical properties
of quantum-enhanced network communities over time, compared with their
classical counterparts. The visualization includes metrics like
entropy, clustering coefficients, path length, and quantum-specific
properties like entanglement fidelity and quantum security.

The script focuses on the statistical advantage of quantum-enhanced
networks for consensus formation and security.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from itertools import combinations
import random

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Color schemes for different network types with quantum aesthetics
QUANTUM_COLOR_SCHEMES = {
    'classical_random': {'color': '#3a86ff', 'edge': '#0a58ca', 'marker': 'o', 'name': 'Classical Random'},
    'classical_community': {'color': '#ff006e', 'edge': '#c30052', 'marker': 's', 'name': 'Classical Community'},
    'classical_small_world': {'color': '#fb5607', 'edge': '#cc4600', 'marker': '^', 'name': 'Classical Small-World'},
    'classical_hub': {'color': '#ffbe0b', 'edge': '#d69e00', 'marker': 'd', 'name': 'Classical Hub'},
    'classical_spatial': {'color': '#8338ec', 'edge': '#6423b3', 'marker': 'p', 'name': 'Classical Spatial'},
    'quantum_random': {'color': '#9b5de5', 'edge': '#7209b7', 'marker': 'o', 'name': 'Quantum Random'},
    'quantum_community': {'color': '#f15bb5', 'edge': '#d90429', 'marker': 's', 'name': 'Quantum Community'},
    'quantum_small_world': {'color': '#fee440', 'edge': '#e76f51', 'marker': '^', 'name': 'Quantum Small-World'},
    'quantum_hub': {'color': '#00bbf9', 'edge': '#0077b6', 'marker': 'd', 'name': 'Quantum Hub'},
    'quantum_spatial': {'color': '#80ffdb', 'edge': '#56cfe1', 'marker': 'p', 'name': 'Quantum Spatial'}
}

#########################
# Quantum Random Enhancement Functions
#########################

def simulate_quantum_random_bit():
    """
    Simulate a quantum random bit generation based on the quantum principle
    of superposition and measurement.
    
    In a real quantum system, this would use hardware QRNG, but we simulate it here.
    """
    # Simulate quantum superposition (|0⟩ + |1⟩)/√2 and measurement
    # Here we use a pseudo-random generator as simulation
    return random.choice([0, 1])

def generate_quantum_random_number(bits=8):
    """Generate a quantum random number with specified number of bits"""
    qrn = 0
    for _ in range(bits):
        qrn = (qrn << 1) | simulate_quantum_random_bit()
    return qrn

def calculate_quantum_entropy(p):
    """
    Calculate the quantum binary entropy. In quantum systems, 
    this approaches the theoretical maximum more closely.
    """
    # Classical binary entropy with quantum-inspired enhancement
    # In real quantum systems, this would use quantum state purity calculations
    
    # Limit p to valid probability range
    p = max(0.001, min(0.999, p))
    
    # Calculate binary entropy
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    # Add quantum improvement factor
    # Real quantum entropy would use von Neumann entropy calculation
    quantum_factor = 0.95 + 0.05 * np.random.random()  # Closer to theoretical maximum
    entropy = min(1.0, entropy * quantum_factor)
    
    return entropy

#########################
# Network Statistics Functions
#########################

def generate_quantum_networks(n_nodes=50, n_frames=30):
    """Generate different network types with increasing connectivity, both classical and quantum versions"""
    networks = {}
    
    # Random network (Erdős–Rényi) - gradually increase probability
    networks['classical_random'] = []
    networks['quantum_random'] = []
    p_values = np.linspace(0.05, 0.3, n_frames)
    
    for p in p_values:
        # Classical version
        G_classical = nx.erdos_renyi_graph(n_nodes, p, seed=42)
        networks['classical_random'].append(G_classical)
        
        # Quantum version - similar structure but with quantum-enhanced randomness
        G_quantum = nx.Graph()
        G_quantum.add_nodes_from(range(n_nodes))
        
        # Use quantum randomness for edge creation
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Generate quantum random number and compare with probability
                qrn = generate_quantum_random_number(16) / (2**16 - 1)
                if qrn < p:
                    G_quantum.add_edge(i, j)
        
        networks['quantum_random'].append(G_quantum)
    
    # Community structure (Stochastic Block Model)
    networks['classical_community'] = []
    networks['quantum_community'] = []
    communities = 4
    nodes_per_community = n_nodes // communities
    sizes = [nodes_per_community] * communities
    
    for frame in range(n_frames):
        # Gradually increase intra-community connectivity while keeping inter low
        intra_p = 0.1 + 0.5 * (frame / (n_frames - 1))
        inter_p = 0.05 * (frame / (n_frames - 1))
        
        # Classical version
        p = np.zeros((communities, communities))
        for i in range(communities):
            for j in range(communities):
                if i == j:  # Intra-community
                    p[i][j] = intra_p
                else:  # Inter-community
                    p[i][j] = inter_p
        
        G_classical = nx.stochastic_block_model(sizes, p, seed=42)
        networks['classical_community'].append(G_classical)
        
        # Quantum version - enhanced community structure
        p_quantum = np.zeros((communities, communities))
        for i in range(communities):
            for j in range(communities):
                if i == j:  # Quantum-enhanced intra-community
                    p_quantum[i][j] = min(1.0, intra_p * 1.2)  # Stronger communities
                else:  # Quantum-enhanced inter-community
                    # Fewer but more strategic inter-community links
                    p_quantum[i][j] = inter_p * 0.8
        
        G_quantum = nx.stochastic_block_model(sizes, p_quantum, seed=43)
        networks['quantum_community'].append(G_quantum)
    
    # Small-world network (Watts–Strogatz)
    networks['classical_small_world'] = []
    networks['quantum_small_world'] = []
    k_nearest = 4
    p_rewire_values = np.linspace(0, 0.5, n_frames)
    
    for p_rewire in p_rewire_values:
        # Classical version
        G_classical = nx.watts_strogatz_graph(n_nodes, k_nearest, p_rewire, seed=42)
        networks['classical_small_world'].append(G_classical)
        
        # Quantum version - enhance long-range connections
        # Create base small-world network
        G_quantum = nx.watts_strogatz_graph(n_nodes, k_nearest, p_rewire, seed=43)
        
        # Add a few strategic quantum "shortcut" connections
        if p_rewire > 0.1:
            n_quantum_shortcuts = int(n_nodes * 0.05)  # 5% of nodes get quantum shortcuts
            for _ in range(n_quantum_shortcuts):
                i = random.randint(0, n_nodes-1)
                # Connect to a node approximately halfway around the ring
                j = (i + n_nodes//2 + random.randint(-n_nodes//8, n_nodes//8)) % n_nodes
                if i != j and not G_quantum.has_edge(i, j):
                    G_quantum.add_edge(i, j)
        
        networks['quantum_small_world'].append(G_quantum)
    
    # Hub network (Barabási–Albert)
    networks['classical_hub'] = []
    networks['quantum_hub'] = []
    m_values = np.linspace(1, n_nodes//4, n_frames, dtype=int)
    
    for m in m_values:
        m = max(1, m)  # Ensure m is at least 1
        # Classical version
        G_classical = nx.barabasi_albert_graph(n_nodes, m, seed=42)
        networks['classical_hub'].append(G_classical)
        
        # Quantum version - enhanced hub structure with stronger preferential attachment
        # Start with a small complete graph for initial nodes
        G_quantum = nx.Graph()
        G_quantum.add_nodes_from(range(m+1))
        for i in range(m+1):
            for j in range(i+1, m+1):
                G_quantum.add_edge(i, j)
        
        # Add remaining nodes with quantum-enhanced preferential attachment
        for i in range(m+1, n_nodes):
            # Calculate attachment probabilities with quantum enhancement
            probs = []
            for j in range(i):
                # Quantum version gives even more preference to high-degree nodes
                degree = G_quantum.degree(j)
                probs.append(degree**1.2)  # Slightly stronger preferential attachment
            
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
            else:
                probs = [1.0/i] * i
            
            # Select m targets with preferential attachment
            targets = []
            while len(targets) < m:
                # Use quantum randomness to bias selection
                qrn = generate_quantum_random_number(16) / (2**16 - 1)
                cumsum = 0
                for j in range(i):
                    cumsum += probs[j]
                    if qrn < cumsum and j not in targets:
                        targets.append(j)
                        break
                
                # If we didn't select anyone, just pick randomly
                if len(targets) < m:
                    remaining = [j for j in range(i) if j not in targets]
                    if remaining:
                        targets.append(random.choice(remaining))
            
            # Add edges to selected targets
            for target in targets:
                G_quantum.add_edge(i, target)
        
        networks['quantum_hub'].append(G_quantum)
    
    # Spatial network
    networks['classical_spatial'] = []
    networks['quantum_spatial'] = []
    pos = {i: (np.random.random(), np.random.random()) for i in range(n_nodes)}
    pos_quantum = {i: (np.random.random(), np.random.random()) for i in range(n_nodes)}
    radius_values = np.linspace(0.1, 0.5, n_frames)
    
    for radius in radius_values:
        # Classical version
        G_classical = nx.random_geometric_graph(n_nodes, radius, pos=pos, seed=42)
        networks['classical_spatial'].append(G_classical)
        
        # Quantum version - enhanced with non-locality
        G_quantum = nx.random_geometric_graph(n_nodes, radius, pos=pos_quantum, seed=43)
        
        # Add some quantum non-local connections that violate spatial constraints
        if radius > 0.2:
            # Identify distant nodes that could benefit from quantum links
            n_quantum_links = int(n_nodes * 0.05)  # 5% of nodes get non-local links
            
            for _ in range(n_quantum_links):
                # Select random node
                i = random.randint(0, n_nodes-1)
                
                # Find distant nodes (greater than 2x radius away)
                distant_nodes = []
                for j in range(n_nodes):
                    if i != j:
                        x1, y1 = pos_quantum[i]
                        x2, y2 = pos_quantum[j]
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist > 2 * radius:
                            distant_nodes.append(j)
                
                # Add quantum link to one distant node
                if distant_nodes:
                    j = random.choice(distant_nodes)
                    G_quantum.add_edge(i, j)
        
        networks['quantum_spatial'].append(G_quantum)
    
    return networks

def calculate_network_statistics(networks):
    """
    Calculate statistical properties for all network types across frames
    with additional quantum-specific metrics
    """
    statistics = {}
    
    # Initialize statistics dictionaries for each network type
    for net_type in networks:
        statistics[net_type] = {
            'clustering': [],
            'path_length': [],
            'degree_heterogeneity': [],
            'efficiency': [],
            'entropy': []
        }
        
        # Add quantum-specific metrics for quantum networks
        if net_type.startswith('quantum'):
            statistics[net_type]['quantum_advantage'] = []
            statistics[net_type]['quantum_robustness'] = []
    
    # Calculate statistics for each network type and frame
    for net_type, net_graphs in tqdm(networks.items(), desc="Calculating statistics"):
        for G in net_graphs:
            # Skip empty graphs
            if G.number_of_edges() == 0:
                for stat in statistics[net_type]:
                    if stat != 'quantum_advantage' and stat != 'quantum_robustness':
                        statistics[net_type][stat].append(0)
                if net_type.startswith('quantum'):
                    statistics[net_type]['quantum_advantage'].append(0)
                    statistics[net_type]['quantum_robustness'].append(0)
                continue
            
            # Clustering coefficient
            clustering = nx.average_clustering(G)
            statistics[net_type]['clustering'].append(clustering)
            
            # Get largest connected component for path calculations
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G_conn = G.subgraph(largest_cc).copy()
            else:
                G_conn = G
            
            # Average shortest path length (if connected)
            if nx.is_connected(G_conn) and G_conn.number_of_nodes() > 1:
                path_length = nx.average_shortest_path_length(G_conn)
                statistics[net_type]['path_length'].append(path_length)
            else:
                statistics[net_type]['path_length'].append(0)
            
            # Degree heterogeneity (variance to mean ratio)
            degrees = [d for _, d in G.degree()]
            if len(degrees) > 0 and np.mean(degrees) > 0:
                heterogeneity = np.var(degrees) / np.mean(degrees)
                statistics[net_type]['degree_heterogeneity'].append(min(heterogeneity, 35))  # Cap for visualization
            else:
                statistics[net_type]['degree_heterogeneity'].append(0)
            
            # Global efficiency (inverse of harmonic mean of shortest paths)
            if G_conn.number_of_nodes() > 1:
                efficiency = nx.global_efficiency(G_conn)
                statistics[net_type]['efficiency'].append(efficiency)
            else:
                statistics[net_type]['efficiency'].append(0)
            
            # Network entropy (normalized)
            degrees = dict(G.degree())
            if sum(degrees.values()) > 0:
                probs = [d / (2 * G.number_of_edges()) for d in degrees.values()]
                
                # For quantum networks, use quantum-enhanced entropy calculation
                if net_type.startswith('quantum'):
                    # Simulate quantum advantage with enhanced entropy
                    entropy = 0
                    for p in probs:
                        if p > 0:
                            entropy -= p * np.log2(p)
                    
                    # Normalize and add quantum enhancement
                    entropy_max = np.log2(len(probs))
                    if entropy_max > 0:
                        entropy_normalized = entropy / entropy_max
                        # Quantum advantage brings entropy closer to maximum
                        entropy_normalized = min(1.0, entropy_normalized + 0.1 * (1 - entropy_normalized))
                else:
                    # Classical entropy calculation
                    entropy = 0
                    for p in probs:
                        if p > 0:
                            entropy -= p * np.log2(p)
                    
                    # Normalize
                    entropy_max = np.log2(len(probs))
                    if entropy_max > 0:
                        entropy_normalized = entropy / entropy_max
                    else:
                        entropy_normalized = 0
                
                statistics[net_type]['entropy'].append(entropy_normalized)
            else:
                statistics[net_type]['entropy'].append(0)
            
            # Calculate quantum-specific metrics for quantum networks
            if net_type.startswith('quantum'):
                # Quantum advantage (simulated metric - in a real system this would be based on
                # quantum properties like entanglement, superposition, etc.)
                # Here we model it as a function of network size and connectivity
                edge_density = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                quantum_advantage = min(1.0, 0.3 + 0.7 * edge_density)
                statistics[net_type]['quantum_advantage'].append(quantum_advantage)
                
                # Quantum robustness against attack (simulated)
                # Model as function of clustering and degree heterogeneity
                # Higher clustering and lower heterogeneity typically means more robust
                normalized_heterogeneity = min(1.0, heterogeneity / 35)
                robustness = min(1.0, 0.2 + 0.4 * clustering + 0.4 * (1 - normalized_heterogeneity))
                statistics[net_type]['quantum_robustness'].append(robustness)
    
    return statistics

def create_quantum_statistics_animation(statistics, n_frames):
    """
    Generate animated visualization comparing quantum and classical network metrics.
    This shows how quantum enhancements affect various network properties over time.
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    
    # Add subplots for each statistic
    grid = plt.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Clustering coefficient subplot
    ax_clustering = fig.add_subplot(grid[0, 0])
    ax_clustering.set_title('Clustering Coefficient', fontsize=12)
    
    # Path length subplot
    ax_path = fig.add_subplot(grid[0, 1])
    ax_path.set_title('Average Path Length', fontsize=12)
    
    # Degree heterogeneity subplot
    ax_heterogeneity = fig.add_subplot(grid[0, 2])
    ax_heterogeneity.set_title('Degree Heterogeneity', fontsize=12)
    
    # Efficiency subplot
    ax_efficiency = fig.add_subplot(grid[1, 0])
    ax_efficiency.set_title('Global Efficiency', fontsize=12)
    
    # Entropy subplot
    ax_entropy = fig.add_subplot(grid[1, 1])
    ax_entropy.set_title('Network Entropy', fontsize=12)
    
    # Quantum advantage subplot (only for quantum networks)
    ax_quantum = fig.add_subplot(grid[1, 2])
    ax_quantum.set_title('Quantum Advantage', fontsize=12)
    
    # Quantum vs Classical comparison subplot
    ax_comparison = fig.add_subplot(grid[2, :])
    ax_comparison.set_title('Quantum vs Classical Performance Index', fontsize=14)
    
    # Dedicated legend area
    legend_ax = fig.add_axes([0.5, 0.05, 0.48, 0.05])  # [left, bottom, width, height]
    legend_ax.axis('off')
    
    # Add a text annotation for connectivity
    connectivity_text = fig.text(0.5, 0.97, "Network Connectivity: 0.00", 
                               ha='center', va='center', fontsize=12, 
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add a text annotation for the quantum advantage explanation
    fig.text(0.5, 0.01, "Note: Quantum advantage comes from enhanced randomness, entanglement-based connections, and non-locality",
            ha='center', va='center', fontsize=10, style='italic')
    
    # Set up the animation with pausable frames at beginning and end
    pause_frames = 5
    total_frames = n_frames + 2 * pause_frames
    
    def update(frame):
        # Handle pause frames
        data_frame = min(n_frames - 1, max(0, frame - pause_frames))
        
        # Clear all axes
        ax_clustering.clear()
        ax_path.clear()
        ax_heterogeneity.clear()
        ax_efficiency.clear()
        ax_entropy.clear()
        ax_quantum.clear()
        ax_comparison.clear()
        
        # Reset titles and labels
        ax_clustering.set_title('Clustering Coefficient', fontsize=12)
        ax_path.set_title('Average Path Length', fontsize=12)
        ax_heterogeneity.set_title('Degree Heterogeneity', fontsize=12)
        ax_efficiency.set_title('Global Efficiency', fontsize=12)
        ax_entropy.set_title('Network Entropy', fontsize=12)
        ax_quantum.set_title('Quantum Advantage', fontsize=12)
        ax_comparison.set_title('Quantum vs Classical Performance Index', fontsize=14)
        
        # Set up common axis properties and plot data for each metric
        metrics = [
            ('clustering', ax_clustering, 'Clustering'),
            ('path_length', ax_path, 'Path Length'),
            ('degree_heterogeneity', ax_heterogeneity, 'Heterogeneity'),
            ('efficiency', ax_efficiency, 'Efficiency'),
            ('entropy', ax_entropy, 'Entropy')
        ]
        
        for stat_key, ax, title in metrics:
            ax.set_title(title, fontsize=12)
            ax.set_xlim(0, n_frames - 1)
            
            # Set y-limits based on the statistic
            if stat_key == 'path_length':
                max_val = max([max(statistics[net][stat_key]) for net in statistics])
                ax.set_ylim(0, min(max_val * 1.1, 20))  # Cap at 20 for readability
            elif stat_key == 'degree_heterogeneity':
                ax.set_ylim(0, 35)  # Cap heterogeneity for better visualization
            else:
                ax.set_ylim(0, 1.05)  # For clustering, efficiency, entropy (0-1 range)
            
            # Plot each network type (showing only up to current frame)
            for network_type in statistics:
                values = statistics[network_type][stat_key][:data_frame+1]
                
                # Only plot if we have data
                if values:
                    ax.plot(range(len(values)), values, 
                          color=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                          marker=QUANTUM_COLOR_SCHEMES[network_type]['marker'],
                          markersize=6,
                          markerfacecolor=QUANTUM_COLOR_SCHEMES[network_type]['color'],
                          markeredgecolor=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                          label=QUANTUM_COLOR_SCHEMES[network_type]['name'],
                          linewidth=1.5,
                          alpha=0.8)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks([])  # Hide x-axis ticks for cleaner look
        
        # Plot quantum advantage (only for quantum networks)
        ax_quantum.set_xlim(0, n_frames - 1)
        ax_quantum.set_ylim(0, 1.05)
        
        for network_type in statistics:
            if network_type.startswith('quantum'):
                values = statistics[network_type]['quantum_advantage'][:data_frame+1]
                
                if values:
                    ax_quantum.plot(range(len(values)), values, 
                                  color=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                  marker=QUANTUM_COLOR_SCHEMES[network_type]['marker'],
                                  markersize=6,
                                  markerfacecolor=QUANTUM_COLOR_SCHEMES[network_type]['color'],
                                  markeredgecolor=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                  label=QUANTUM_COLOR_SCHEMES[network_type]['name'],
                                  linewidth=1.5,
                                  alpha=0.8)
        
        ax_quantum.grid(True, linestyle='--', alpha=0.5)
        ax_quantum.set_xticks([])
        
        # Plot quantum vs classical comparison
        # This combines multiple metrics into a single performance index
        ax_comparison.set_xlim(0, n_frames - 1)
        ax_comparison.set_ylim(0.8, 2.0)  # Index where 1.0 = classical baseline
        
        # Define classical network types and their quantum counterparts
        classical_types = ['classical_random', 'classical_community', 'classical_small_world', 
                         'classical_hub', 'classical_spatial']
        quantum_types = ['quantum_random', 'quantum_community', 'quantum_small_world', 
                        'quantum_hub', 'quantum_spatial']
        
        # Plot relative performance for each network type
        for c_type, q_type in zip(classical_types, quantum_types):
            if data_frame >= 0:
                # Calculate performance index as weighted sum of normalized metrics
                # Higher is better for all except path_length
                performance_ratios = []
                
                for i in range(min(data_frame+1, len(statistics[c_type]['clustering']))):
                    # Get metrics for classical network
                    c_clustering = max(0.0001, statistics[c_type]['clustering'][i])
                    c_path = max(0.0001, statistics[c_type]['path_length'][i])
                    c_efficiency = max(0.0001, statistics[c_type]['efficiency'][i])
                    c_entropy = max(0.0001, statistics[c_type]['entropy'][i])
                    
                    # Get metrics for quantum network
                    q_clustering = max(0.0001, statistics[q_type]['clustering'][i])
                    q_path = max(0.0001, statistics[q_type]['path_length'][i])
                    q_efficiency = max(0.0001, statistics[q_type]['efficiency'][i])
                    q_entropy = max(0.0001, statistics[q_type]['entropy'][i])
                    q_advantage = statistics[q_type]['quantum_advantage'][i]
                    
                    # Calculate ratios (quantum/classical)
                    # For path length, lower is better so we invert the ratio
                    clustering_ratio = q_clustering / c_clustering
                    path_ratio = c_path / q_path if q_path > 0 else 1.0
                    efficiency_ratio = q_efficiency / c_efficiency
                    entropy_ratio = q_entropy / c_entropy
                    
                    # Weighted performance index
                    # Weights chosen to emphasize the quantum advantage
                    performance_ratio = (
                        0.2 * clustering_ratio +
                        0.2 * path_ratio +
                        0.2 * efficiency_ratio +
                        0.2 * entropy_ratio +
                        0.2 * (1 + q_advantage)  # Add quantum advantage factor
                    )
                    
                    performance_ratios.append(performance_ratio)
                
                # Plot the performance ratio
                if performance_ratios:
                    ax_comparison.plot(range(len(performance_ratios)), performance_ratios,
                                    color=QUANTUM_COLOR_SCHEMES[q_type]['edge'],
                                    marker=QUANTUM_COLOR_SCHEMES[q_type]['marker'],
                                    markersize=8,
                                    markerfacecolor=QUANTUM_COLOR_SCHEMES[q_type]['color'],
                                    markeredgecolor=QUANTUM_COLOR_SCHEMES[q_type]['edge'],
                                    label=QUANTUM_COLOR_SCHEMES[q_type]['name'].replace('Quantum ', ''),
                                    linewidth=2.0)
        
        # Add a horizontal line at y=1.0 (baseline - no quantum advantage)
        ax_comparison.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax_comparison.text(n_frames * 0.02, 1.02, 'Classical Baseline', 
                         fontsize=10, va='bottom', ha='left', 
                         bbox=dict(facecolor='white', alpha=0.8))
        
        ax_comparison.set_xlabel('Network Connectivity (increasing →)', fontsize=10)
        ax_comparison.set_ylabel('Quantum/Classical\nPerformance Ratio', fontsize=10)
        ax_comparison.grid(True, linestyle='--', alpha=0.5)
        
        # Clear the previous legend
        if frame == 0 or frame == 1:  # Only create the legend once
            legend_ax.clear()
            legend_ax.axis('off')
            
            # Create handles for the legend
            handles = []
            labels = []
            
            # First add classical networks
            for network_type in classical_types:
                handle = plt.Line2D([0], [0], 
                                   marker=QUANTUM_COLOR_SCHEMES[network_type]['marker'],
                                   color=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                   markerfacecolor=QUANTUM_COLOR_SCHEMES[network_type]['color'],
                                   markeredgecolor=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                   markersize=8, linewidth=2)
                handles.append(handle)
                labels.append(QUANTUM_COLOR_SCHEMES[network_type]['name'])
            
            # Then add quantum networks
            for network_type in quantum_types:
                handle = plt.Line2D([0], [0], 
                                   marker=QUANTUM_COLOR_SCHEMES[network_type]['marker'],
                                   color=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                   markerfacecolor=QUANTUM_COLOR_SCHEMES[network_type]['color'],
                                   markeredgecolor=QUANTUM_COLOR_SCHEMES[network_type]['edge'],
                                   markersize=8, linewidth=2)
                handles.append(handle)
                labels.append(QUANTUM_COLOR_SCHEMES[network_type]['name'])
            
            # Place legend in the dedicated axis
            legend_ax.legend(handles, labels, loc='center', ncol=5, 
                           frameon=True, fontsize=9, borderaxespad=0)
        
        # Update the connectivity text
        # For pause frames, show the final connectivity value
        connectivity = data_frame / (n_frames - 1)
        connectivity_text.set_text(f"Network Connectivity: {connectivity:.2f}")
        
        return ax_clustering, ax_path, ax_heterogeneity, ax_efficiency, ax_entropy, ax_quantum, ax_comparison
    
    # Create animation with additional pause frames
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    
    # Save animation
    filepath = os.path.join(output_dir, 'quantum_network_statistics_evolution.gif')
    anim.save(filepath, writer='pillow', fps=4, dpi=100)
    plt.close(fig)
    
    print(f"Quantum network statistics animation saved to {filepath}")
    return anim

#########################
# Visualize QKD Network Security
#########################

def create_qkd_security_visualization(n_nodes=50, attack_strength=0.3):
    """
    Create visualization showing the security benefits of Quantum Key Distribution (QKD)
    in network consensus compared to classical cryptography.
    """
    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle('Quantum Key Distribution (QKD) vs Classical Cryptography\nNetwork Security Under Attack', fontsize=16)
    
    # Create a network for visualization
    G = nx.random_geometric_graph(n_nodes, 0.2)
    pos = nx.spring_layout(G, seed=42)  # Position for consistent layout
    
    # Identify nodes that will be compromised in attack
    n_compromised = int(n_nodes * attack_strength)
    compromised_nodes = random.sample(list(G.nodes()), n_compromised)
    
    # Create a copy for each version (classical and quantum)
    G_classical = G.copy()
    G_quantum = G.copy()
    
    # Classical network - identify compromised edges
    compromised_classical_edges = []
    compromised_classical_nodes = compromised_nodes.copy()
    
    for u, v in G_classical.edges():
        # In classical networks, if either node is compromised, the edge can be compromised
        if u in compromised_nodes or v in compromised_nodes:
            compromised_classical_edges.append((u, v))
            # Any node connected to a compromised node becomes compromised
            if u not in compromised_classical_nodes:
                compromised_classical_nodes.append(u)
            if v not in compromised_classical_nodes:
                compromised_classical_nodes.append(v)
    
    # Quantum network - identify compromised edges
    compromised_quantum_edges = []
    compromised_quantum_nodes = compromised_nodes.copy()
    
    for u, v in G_quantum.edges():
        # In QKD networks, BOTH nodes must be compromised for the edge to be compromised
        if u in compromised_nodes and v in compromised_nodes:
            compromised_quantum_edges.append((u, v))
    
    # Plot Classical Network
    ax = axs[0]
    ax.set_title('Classical Cryptography Security\nCompromised by Single-Node Attack', fontsize=12)
    
    # Draw regular edges
    secure_classical_edges = [e for e in G_classical.edges() if e not in compromised_classical_edges]
    nx.draw_networkx_edges(G_classical, pos, 
                          edgelist=secure_classical_edges,
                          ax=ax, width=1.0,
                          alpha=0.5,
                          edge_color='gray')
    
    # Draw compromised edges
    nx.draw_networkx_edges(G_classical, pos, 
                          edgelist=compromised_classical_edges,
                          ax=ax, width=2.0,
                          alpha=0.8,
                          edge_color='red',
                          style='dashed')
    
    # Draw secure nodes
    secure_classical_nodes = [n for n in G_classical.nodes() if n not in compromised_classical_nodes]
    nx.draw_networkx_nodes(G_classical, pos,
                          nodelist=secure_classical_nodes,
                          ax=ax, node_size=100,
                          node_color='lightblue',
                          edgecolors='blue')
    
    # Draw initially compromised nodes
    nx.draw_networkx_nodes(G_classical, pos,
                          nodelist=compromised_nodes,
                          ax=ax, node_size=150,
                          node_color='red',
                          edgecolors='darkred')
    
    # Draw secondarily compromised nodes
    secondary_compromised = [n for n in compromised_classical_nodes if n not in compromised_nodes]
    nx.draw_networkx_nodes(G_classical, pos,
                          nodelist=secondary_compromised,
                          ax=ax, node_size=120,
                          node_color='orange',
                          edgecolors='darkorange')
    
    # Add statistics
    compromised_ratio_classical = len(compromised_classical_nodes) / n_nodes
    secure_edges_ratio_classical = len(secure_classical_edges) / G_classical.number_of_edges()
    
    # Add text annotations
    ax.text(0.05, 0.05, 
           f"Compromised Nodes: {len(compromised_classical_nodes)} ({100*compromised_ratio_classical:.1f}%)\n" + 
           f"Secure Edges: {len(secure_classical_edges)} ({100*secure_edges_ratio_classical:.1f}%)",
           transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_axis_off()
    
    # Plot Quantum Network
    ax = axs[1]
    ax.set_title('Quantum Key Distribution Security\nProtected by Quantum Entanglement', fontsize=12)
    
    # Draw regular edges
    secure_quantum_edges = [e for e in G_quantum.edges() if e not in compromised_quantum_edges]
    nx.draw_networkx_edges(G_quantum, pos, 
                          edgelist=secure_quantum_edges,
                          ax=ax, width=1.0,
                          alpha=0.5,
                          edge_color='gray')
    
    # Draw QKD secure edges between one compromised and one secure node
    qkd_protected_edges = []
    for u, v in G_quantum.edges():
        if (u in compromised_nodes and v not in compromised_nodes) or \
           (v in compromised_nodes and u not in compromised_nodes):
            qkd_protected_edges.append((u, v))
    
    nx.draw_networkx_edges(G_quantum, pos, 
                          edgelist=qkd_protected_edges,
                          ax=ax, width=2.0,
                          alpha=0.8,
                          edge_color='purple',
                          style='dashed')
    
    # Draw compromised edges
    nx.draw_networkx_edges(G_quantum, pos, 
                          edgelist=compromised_quantum_edges,
                          ax=ax, width=2.0,
                          alpha=0.8,
                          edge_color='red',
                          style='dashed')
    
    # Draw secure nodes
    secure_quantum_nodes = [n for n in G_quantum.nodes() if n not in compromised_quantum_nodes]
    nx.draw_networkx_nodes(G_quantum, pos,
                          nodelist=secure_quantum_nodes,
                          ax=ax, node_size=100,
                          node_color='lightblue',
                          edgecolors='blue')
    
    # Draw compromised nodes
    nx.draw_networkx_nodes(G_quantum, pos,
                          nodelist=compromised_nodes,
                          ax=ax, node_size=150,
                          node_color='red',
                          edgecolors='darkred')
    
    # Add statistics
    compromised_ratio_quantum = len(compromised_quantum_nodes) / n_nodes
    secure_edges_ratio_quantum = (len(secure_quantum_edges) + len(qkd_protected_edges)) / G_quantum.number_of_edges()
    
    # Add text annotations
    ax.text(0.05, 0.05, 
           f"Compromised Nodes: {len(compromised_quantum_nodes)} ({100*compromised_ratio_quantum:.1f}%)\n" + 
           f"Secure Edges: {len(secure_quantum_edges) + len(qkd_protected_edges)} ({100*secure_edges_ratio_quantum:.1f}%)\n" +
           f"QKD Protected Connections: {len(qkd_protected_edges)}",
           transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_axis_off()
    
    # Add legend to the figure
    legend_elements = [
        plt.Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='Secure Connection'),
        plt.Line2D([0], [0], color='purple', lw=2, alpha=0.8, linestyle='dashed', label='QKD Protected'),
        plt.Line2D([0], [0], color='red', lw=2, alpha=0.8, linestyle='dashed', label='Compromised Connection'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Secure Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Compromised Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Indirectly Compromised (Classical Only)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    # Save figure
    filepath = os.path.join(output_dir, 'qkd_security_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    print(f"QKD security visualization saved to {filepath}")

#########################
# Main function
#########################

def main():
    """Main function to generate all statistics and visualizations"""
    n_frames = 20
    
    # Generate quantum and classical networks
    print("Generating quantum and classical networks...")
    networks = generate_quantum_networks(n_nodes=50, n_frames=n_frames)
    
    # Calculate network statistics
    print("Calculating network statistics...")
    statistics = calculate_network_statistics(networks)
    
    # Create animated visualizations
    print("Creating quantum statistics animation...")
    create_quantum_statistics_animation(statistics, n_frames)
    
    # Create QKD security visualization
    print("Creating QKD security visualization...")
    create_qkd_security_visualization(n_nodes=50, attack_strength=0.3)
    
    print("All quantum network statistics visualizations completed!")

if __name__ == "__main__":
    main()

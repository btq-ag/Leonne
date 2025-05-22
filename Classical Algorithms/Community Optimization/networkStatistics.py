#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkStatistics.py

This script creates animated visualizations showing statistical properties
of different network community types over time, including metrics like 
clustering coefficients, path length, and other network characteristics.

The script focuses on statistical animation of network metrics (clustering, 
path length, degree heterogeneity, efficiency) across different network types.

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

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
# Remove the line that creates a subdirectory

# Set random seed for reproducibility
np.random.seed(42)

# Color schemes for different network types with more distinct colors
COLOR_SCHEMES = {
    'random': {'color': '#3a86ff', 'edge': '#0a58ca', 'marker': 'o', 'name': 'Random (Erdős–Rényi)'},
    'community': {'color': '#ff006e', 'edge': '#c30052', 'marker': 's', 'name': 'Community (Stochastic Block)'},
    'small_world': {'color': '#fb5607', 'edge': '#cc4600', 'marker': '^', 'name': 'Small-world (Watts–Strogatz)'},
    'hub': {'color': '#ffbe0b', 'edge': '#d69e00', 'marker': 'd', 'name': 'Hub (Barabási–Albert)'},
    'spatial': {'color': '#8338ec', 'edge': '#6423b3', 'marker': 'p', 'name': 'Spatial (Geographic)'}
}

#########################
# Network Statistics Animation
#########################

def generate_networks(n_nodes=50, n_frames=30):
    """Generate different network types with increasing connectivity"""
    networks = {}
    
    # Random network (Erdős–Rényi) - gradually increase probability
    networks['random'] = []
    p_values = np.linspace(0.05, 0.3, n_frames)
    for p in p_values:
        G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
        networks['random'].append(G)
    
    # Community structure (Stochastic Block Model)
    networks['community'] = []
    communities = 4
    nodes_per_community = n_nodes // communities
    sizes = [nodes_per_community] * communities
    
    for frame in range(n_frames):
        # Gradually increase intra-community connectivity while keeping inter low
        intra_p = 0.1 + 0.5 * (frame / (n_frames - 1))
        inter_p = 0.05 * (frame / (n_frames - 1))
        
        p = np.zeros((communities, communities))
        for i in range(communities):
            for j in range(communities):
                if i == j:  # Intra-community
                    p[i][j] = intra_p
                else:  # Inter-community
                    p[i][j] = inter_p
        
        G = nx.stochastic_block_model(sizes, p, seed=42)
        networks['community'].append(G)
    
    # Small-world network (Watts–Strogatz)
    networks['small_world'] = []
    k_nearest = 4
    p_rewire_values = np.linspace(0, 0.5, n_frames)
    
    for p_rewire in p_rewire_values:
        G = nx.watts_strogatz_graph(n_nodes, k_nearest, p_rewire, seed=42)
        networks['small_world'].append(G)
    
    # Hub network (Barabási–Albert) - gradually add edges
    networks['hub'] = []
    m_values = np.linspace(1, n_nodes//4, n_frames, dtype=int)
    
    for m in m_values:
        m = max(1, m)  # Ensure m is at least 1
        G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
        networks['hub'].append(G)
    
    # Spatial network - gradually increase radius
    networks['spatial'] = []
    pos = {i: (np.random.random(), np.random.random()) for i in range(n_nodes)}
    radius_values = np.linspace(0.1, 0.5, n_frames)
    
    for radius in radius_values:
        G = nx.random_geometric_graph(n_nodes, radius, pos=pos, seed=42)
        networks['spatial'].append(G)
    
    return networks

def calculate_network_statistics(networks):
    """Calculate statistics for each network type and frame"""
    statistics = {}
    
    for network_type, graph_list in networks.items():
        statistics[network_type] = {
            'clustering': [],
            'path_length': [],
            'degree_heterogeneity': [],
            'efficiency': [],
            'density': []
        }
        
        for G in tqdm(graph_list, desc=f"Calculating stats for {network_type}"):
            # 1. Average clustering coefficient
            clustering = nx.average_clustering(G)
            statistics[network_type]['clustering'].append(clustering)
            
            # 2. Average shortest path length (if the graph is connected)
            if nx.is_connected(G):
                path_length = nx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, calculate average within components
                components = list(nx.connected_components(G))
                path_length = 0
                total_pairs = 0
                
                for component in components:
                    if len(component) > 1:
                        subgraph = G.subgraph(component)
                        n = len(component)
                        path_length += nx.average_shortest_path_length(subgraph) * (n * (n-1) / 2)
                        total_pairs += n * (n-1) / 2
                
                if total_pairs > 0:
                    path_length /= total_pairs
                else:
                    path_length = float('inf')
            
            statistics[network_type]['path_length'].append(path_length)
            
            # 3. Degree heterogeneity (variance/mean of degree distribution)
            degrees = [d for _, d in G.degree()]
            mean_degree = np.mean(degrees) if degrees else 0
            if mean_degree > 0:
                heterogeneity = np.var(degrees) / mean_degree
            else:
                heterogeneity = 0
            statistics[network_type]['degree_heterogeneity'].append(heterogeneity)
            
            # 4. Global efficiency
            if len(G) > 1:
                efficiency = nx.global_efficiency(G)
            else:
                efficiency = 0
            statistics[network_type]['efficiency'].append(efficiency)
            
            # 5. Graph density
            density = nx.density(G)
            statistics[network_type]['density'].append(density)
    
    return statistics

def create_statistics_animation(statistics, n_frames):
    """Create animation of network statistics evolution"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    axs = axs.flatten()
    
    # Define the statistics to plot
    plot_stats = [
        ('clustering', 'Average Clustering Coefficient'),
        ('path_length', 'Average Shortest Path Length'),
        ('degree_heterogeneity', 'Degree Heterogeneity'),
        ('efficiency', 'Global Efficiency')
    ]
    
    # Set a consistent title for all frames
    fig.suptitle("Network Statistics Evolution", fontsize=16, y=0.98)
    
    # Create a text object for connectivity value that we'll update
    connectivity_text = fig.text(0.5, 0.01, "", ha='center', fontsize=12)
    
    # Add 5 pause frames at the end
    pause_frames = 5
    total_frames = n_frames + pause_frames
    
    # Create a legend axis at the bottom center
    legend_ax = fig.add_axes([0.3, 0.04, 0.4, 0.02])  # [left, bottom, width, height]
    legend_ax.axis('off')
    
    # Function to update the plot for each frame
    def update(frame):
        # If we're in the pause frames, use the last data frame
        data_frame = min(frame, n_frames - 1)
        
        for i, (stat_key, title) in enumerate(plot_stats):
            ax = axs[i]
            ax.clear()
            ax.set_title(title, fontsize=14)
            ax.set_xlim(0, n_frames - 1)
            
            # Set y-limits based on the statistic
            if stat_key == 'path_length':
                max_val = max([max(statistics[net][stat_key]) for net in statistics])
                ax.set_ylim(0, min(max_val * 1.1, 20))  # Cap at 20 for readability
            elif stat_key == 'degree_heterogeneity':
                ax.set_ylim(0, 35)  # Cap heterogeneity for better visualization
            else:
                ax.set_ylim(0, 1.05)  # For clustering and efficiency (0-1 range)
            
            # Plot each network type
            for network_type in statistics:
                values = statistics[network_type][stat_key][:data_frame+1]
                ax.plot(range(data_frame+1), values, 
                        color=COLOR_SCHEMES[network_type]['edge'],
                        marker=COLOR_SCHEMES[network_type]['marker'],
                        markersize=8,
                        markerfacecolor=COLOR_SCHEMES[network_type]['color'],
                        markeredgecolor=COLOR_SCHEMES[network_type]['edge'],
                        label=COLOR_SCHEMES[network_type]['name'])
            
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Clear the previous legend
        if frame == 0 or frame == 1:  # Only create the legend once
            legend_ax.clear()
            legend_ax.axis('off')
            
            # Create handles for the legend
            handles = []
            labels = []
            for network_type in statistics:
                handle = plt.Line2D([0], [0], 
                                  marker=COLOR_SCHEMES[network_type]['marker'],
                                  color=COLOR_SCHEMES[network_type]['edge'],
                                  markerfacecolor=COLOR_SCHEMES[network_type]['color'],
                                  markeredgecolor=COLOR_SCHEMES[network_type]['edge'],
                                  markersize=8, linewidth=2)
                handles.append(handle)
                labels.append(COLOR_SCHEMES[network_type]['name'])
            
            # Place legend in the dedicated axis
            legend_ax.legend(handles, labels, loc='center', ncol=5, 
                           frameon=True, fontsize=10, borderaxespad=0)
        
        # Update the connectivity text
        # For pause frames, show the final connectivity value
        connectivity = data_frame / (n_frames - 1)
        connectivity_text.set_text(f"Network Connectivity: {connectivity:.2f}")
        
        return axs
    
    # Create animation with additional pause frames
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    
    # Save animation
    anim.save(os.path.join(output_dir, 'network_statistics_evolution.gif'), 
              writer='pillow', fps=4, dpi=100)
    plt.close(fig)
    
    print(f"Network statistics animation saved to {os.path.join(output_dir, 'network_statistics_evolution.gif')}")
    return anim

#########################
# Main function
#########################

def main():
    n_frames = 20
    output_dir = os.path.dirname(os.path.abspath(__file__))
      # Generate network statistics animation
    print("Generating networks...")
    networks = generate_networks(n_nodes=50, n_frames=n_frames)
    
    print("Calculating network statistics...")
    statistics = calculate_network_statistics(networks)
    
    print("Creating statistics animation...")
    create_statistics_animation(statistics, n_frames)
    
    # Create a variation with different network parameters
    print("Creating variation of network statistics...")
    
    # Variation 1: Larger networks with more distinct community structure
    def create_statistics_variation():
        networks_var = {}
        
        # Random network with higher density
        networks_var['random'] = []
        p_values = np.linspace(0.08, 0.35, n_frames)
        for p in p_values:
            G = nx.erdos_renyi_graph(60, p, seed=43)
            networks_var['random'].append(G)
        
        # Community structure with stronger communities
        networks_var['community'] = []
        communities = 5  # More communities
        nodes_per_community = 12  # More nodes per community
        sizes = [nodes_per_community] * communities
        
        for frame in range(n_frames):
            # Higher intra-community, lower inter-community
            intra_p = 0.15 + 0.6 * (frame / (n_frames - 1))
            inter_p = 0.02 * (frame / (n_frames - 1))
            
            p = np.zeros((communities, communities))
            for i in range(communities):
                for j in range(communities):
                    if i == j:  # Intra-community
                        p[i][j] = intra_p
                    else:  # Inter-community
                        p[i][j] = inter_p
            
            G = nx.stochastic_block_model(sizes, p, seed=43)
            networks_var['community'].append(G)
        
        # Small-world with different parameters
        networks_var['small_world'] = []
        k_nearest = 6  # More initial connections
        p_rewire_values = np.linspace(0, 0.6, n_frames)  # More rewiring
        
        for p_rewire in p_rewire_values:
            G = nx.watts_strogatz_graph(60, k_nearest, p_rewire, seed=43)
            networks_var['small_world'].append(G)
        
        # Hub network with more hub emphasis
        networks_var['hub'] = []
        m_values = np.linspace(2, 15, n_frames, dtype=int)  # More preferential attachment
        
        for m in m_values:
            m = max(2, m)  # Ensure m is at least 2
            G = nx.barabasi_albert_graph(60, m, seed=43)
            networks_var['hub'].append(G)
        
        # Spatial network with different radius
        networks_var['spatial'] = []
        pos = {i: (np.random.random(), np.random.random()) for i in range(60)}
        radius_values = np.linspace(0.15, 0.4, n_frames)  # Different radius progression
        
        for radius in radius_values:
            G = nx.random_geometric_graph(60, radius, pos=pos, seed=43)
            networks_var['spatial'].append(G)
        
        print("Calculating variation statistics...")
        statistics_var = calculate_network_statistics(networks_var)
        
        # Create new figure for the variation
        fig_var, axs_var = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
        axs_var = axs_var.flatten()
        
        # Set title
        fig_var.suptitle("Network Statistics Evolution - Variation", fontsize=16, y=0.98)
          # Create a text object for connectivity value
        connectivity_text_var = fig_var.text(0.5, 0.01, "", ha='center', fontsize=12)
        
        # Add legend axis
        legend_ax_var = fig_var.add_axes([0.3, 0.04, 0.4, 0.02])
        legend_ax_var.axis('off')
        
        # Define the statistics to plot
        plot_stats = [
            ('clustering', 'Average Clustering Coefficient'),
            ('path_length', 'Average Shortest Path Length'),
            ('degree_heterogeneity', 'Degree Heterogeneity'),
            ('efficiency', 'Global Efficiency')
        ]
        
        def update_var(frame):
            data_frame = min(frame, n_frames - 1)
            
            for i, (stat_key, title) in enumerate(plot_stats):
                ax = axs_var[i]
                ax.clear()
                ax.set_title(title, fontsize=14)
                ax.set_xlim(0, n_frames - 1)
                
                # Set y-limits
                if stat_key == 'path_length':
                    max_val = max([max(statistics_var[net][stat_key]) for net in statistics_var])
                    ax.set_ylim(0, min(max_val * 1.1, 20))
                elif stat_key == 'degree_heterogeneity':
                    ax.set_ylim(0, 35)
                else:
                    ax.set_ylim(0, 1.05)
                
                # Plot each network type
                for network_type in statistics_var:
                    values = statistics_var[network_type][stat_key][:data_frame+1]
                    ax.plot(range(data_frame+1), values, 
                            color=COLOR_SCHEMES[network_type]['edge'],
                            marker=COLOR_SCHEMES[network_type]['marker'],
                            markersize=8,
                            markerfacecolor=COLOR_SCHEMES[network_type]['color'],
                            markeredgecolor=COLOR_SCHEMES[network_type]['edge'],
                            label=COLOR_SCHEMES[network_type]['name'])
                
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create legend once
            if frame == 0 or frame == 1:
                legend_ax_var.clear()
                legend_ax_var.axis('off')
                
                handles = []
                labels = []
                for network_type in statistics_var:
                    handle = plt.Line2D([0], [0], 
                                       marker=COLOR_SCHEMES[network_type]['marker'],
                                       color=COLOR_SCHEMES[network_type]['edge'],
                                       markerfacecolor=COLOR_SCHEMES[network_type]['color'],
                                       markeredgecolor=COLOR_SCHEMES[network_type]['edge'],
                                       markersize=8, linewidth=2)
                    handles.append(handle)
                    labels.append(COLOR_SCHEMES[network_type]['name'])
                
                legend_ax_var.legend(handles, labels, loc='center', ncol=5, 
                                    frameon=True, fontsize=10, borderaxespad=0)
            
            # Update connectivity text
            connectivity = data_frame / (n_frames - 1)
            connectivity_text_var.set_text(f"Network Connectivity: {connectivity:.2f}")
            
            return axs_var
        
        # Create and save variation animation
        total_frames = n_frames + 5  # Add pause frames
        anim_var = animation.FuncAnimation(fig_var, update_var, frames=total_frames, interval=200, blit=False)
        
        variation_file = os.path.join(output_dir, 'network_statistics_evolution_variation.gif')
        anim_var.save(variation_file, writer='pillow', fps=4, dpi=100)
        plt.close(fig_var)
        
        print(f"Network statistics variation saved to {variation_file}")
    
    create_statistics_variation()
    
    print("Network statistics visualizations completed successfully!")

if __name__ == "__main__":
    main()
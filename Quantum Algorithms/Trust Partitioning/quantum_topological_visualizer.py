#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_topological_visualizer.py

Enhanced visualization module for quantum trust partitioning with topological features.
This module provides high-quality visualizations and animations inspired by the classical
implementations but tailored for quantum network properties.

Key features:
1. Network visualization with quantum-specific features
2. Animated topological filtration process
3. Persistence diagrams and Betti curves with quantum aesthetics
4. Stability analysis plots
5. 3D simplicial complex visualizations

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import networkx as nx
import os
import random
from itertools import combinations, product
import gudhi as gd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d import proj3d
import warnings
from scipy.spatial import Delaunay
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Define enhanced color schemes for quantum topology visualizations
QUANTUM_COLORS = {
    'background': '#f0f0f8',
    'dark_background': '#0c0c2c',
    'nodes': '#1e1e3f',
    'edges': '#8a4fff',
    'faces': '#ff00ff',
    'tetrahedra': '#ff4081',
    'qkd_links': '#9400d3',
    'filtration': '#00b8d4',
    'betti0': '#e63946',
    'betti1': '#1d3557',
    'betti2': '#06d6a0',
    'arrow': '#ffa400',
    'highlight': '#ffcc00',
    'jump_path': '#ff7f7f',
    'abandon_path': '#ff0000',
    'secure_zone': '#e7cba9',
    'networks': ['#d2b48c', '#c19a6b', '#9370db', '#8b0000', '#7fff00']
}

# Create custom colormaps for quantum visualizations
quantum_cmap = LinearSegmentedColormap.from_list('quantum',
                                                ['#0c0c2c', '#4b0082', '#9400d3', '#ff00ff'], N=256)
                                                
quantum_diverging = LinearSegmentedColormap.from_list('quantum_div',
                                                    ['#0c0c2c', '#4b0082', '#f0f0f8', '#ff00ff', '#ff4081'], N=256)

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

#########################
# Network Visualization Functions
#########################

def visualize_quantum_network(G, pos=None, with_labels=True, node_color=None, 
                             edge_color=None, title=None, ax=None, figsize=(10, 8),
                             node_size=300, width=1.5, alpha=0.8, show=True,
                             save_path=None, dark_mode=False):
    """
    Visualize a quantum network with enhanced aesthetics
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph to visualize
    pos : dict, optional
        Dictionary of node positions
    with_labels : bool, default=True
        Whether to show node labels
    node_color : str or list, optional
        Color(s) for nodes
    edge_color : str or list, optional
        Color(s) for edges
    title : str, optional
        Title for the visualization
    ax : matplotlib.axes.Axes, optional
        Axes to draw the graph on
    figsize : tuple, default=(10, 8)
        Figure size
    node_size : int or list, default=300
        Size(s) for nodes
    width : float or list, default=1.5
        Width(s) for edges
    alpha : float, default=0.8
        Transparency
    show : bool, default=True
        Whether to show the plot
    save_path : str, optional
        Path to save the figure
    dark_mode : bool, default=False
        Whether to use dark mode colors
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with network visualization
    ax : matplotlib.axes.Axes
        Axes with network visualization
    pos : dict
        Node positions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set background color
    if dark_mode:
        ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        label_color = 'white'
    else:
        ax.set_facecolor(QUANTUM_COLORS['background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        label_color = 'black'
    
    # Generate positions if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Set default colors if not provided
    if node_color is None:
        node_color = QUANTUM_COLORS['nodes']
    if edge_color is None:
        edge_color = QUANTUM_COLORS['edges']
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=alpha, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=width, alpha=alpha, ax=ax)
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, font_color=label_color, font_family="sans-serif", ax=ax)
    
    # Set title
    if title:
        ax.set_title(title, color=label_color, fontsize=14)
    
    ax.set_axis_off()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    
    if show:
        plt.show()
    
    return fig, ax, pos

def visualize_quantum_network_with_topology(G, pos, betti_numbers, filtration_value=0.5, 
                                          title="Quantum Network with Topological Features",
                                          ax=None, figsize=(12, 10), show=True, save_path=None,
                                          dark_mode=False):
    """
    Visualize a quantum network with topological features highlighted
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph to visualize
    pos : dict
        Dictionary of node positions
    betti_numbers : dict
        Dictionary of Betti numbers
    filtration_value : float, default=0.5
        Current filtration value
    title : str, default="Quantum Network with Topological Features"
        Title for the visualization
    ax : matplotlib.axes.Axes, optional
        Axes to draw the graph on
    figsize : tuple, default=(12, 10)
        Figure size
    show : bool, default=True
        Whether to show the plot
    save_path : str, optional
        Path to save the figure
    dark_mode : bool, default=False
        Whether to use dark mode colors
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with network visualization
    ax : matplotlib.axes.Axes
        Axes with network visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set background color
    if dark_mode:
        ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        ax.set_facecolor(QUANTUM_COLORS['background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
    
    # Define node and edge attributes based on topological importance
    node_degree = dict(G.degree())
    
    # Calculate betweenness centrality for nodes and edges
    try:
        node_centrality = nx.betweenness_centrality(G)
        edge_centrality = nx.edge_betweenness_centrality(G)
    except:
        # Fallback to degree and weight if centrality calculation fails
        node_centrality = {n: d/max(dict(G.degree()).values()) for n, d in G.degree()}
        edge_centrality = {e: G.get_edge_data(*e).get('weight', 1.0) for e in G.edges()}
    
    # Scale node sizes based on centrality
    node_sizes = [300 + 1000 * node_centrality[n] for n in G.nodes()]
    
    # Create color map based on degree
    node_colors = [node_degree[n] for n in G.nodes()]
    
    # Create edge width based on centrality
    edge_widths = [1 + 5 * edge_centrality[e] for e in G.edges()]
    
    # Draw the network with enhanced features
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                 cmap=quantum_cmap, alpha=0.8, ax=ax)
    
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=[edge_centrality[e] for e in G.edges()], 
                                 edge_cmap=quantum_cmap, alpha=0.7, ax=ax)
    
    # Add a colorbar for node colors
    if len(G.nodes()) > 0:
        sm = ScalarMappable(cmap=quantum_cmap, norm=plt.Normalize(min(node_colors), max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Node Degree')
        cbar.set_label('Node Degree', color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=text_color)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color=text_color, ax=ax)
    
    # Add information about topological features
    betti_text = f"Betti Numbers: β₀={betti_numbers['betti0']}, β₁={betti_numbers['betti1']}, β₂={betti_numbers['betti2']}"
    
    text_box = ax.text(0.02, 0.02, betti_text, transform=ax.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                      color='black')
    
    ax.text(0.02, 0.96, f"Filtration value: {filtration_value:.2f}", transform=ax.transAxes,
           fontsize=12, color=text_color)
    
    # Set title and remove axis
    ax.set_title(title, fontsize=14, color=text_color)
    ax.set_axis_off()
    
    # Legend for topological features
    legend_elements = [
        Patch(facecolor=QUANTUM_COLORS['betti0'], label=f'Connected Components (β₀): {betti_numbers["betti0"]}'),
        Patch(facecolor=QUANTUM_COLORS['betti1'], label=f'Cycles/Holes (β₁): {betti_numbers["betti1"]}'),
        Patch(facecolor=QUANTUM_COLORS['betti2'], label=f'Voids (β₂): {betti_numbers["betti2"]}')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    
    if show:
        plt.show()
    
    return fig, ax

#########################
# Animation Functions
#########################

def create_quantum_topology_animation(networks_sequence, trust_matrices, node_security, filtration_values,
                                    betti_numbers_sequence, title="Quantum Topological Network Evolution",
                                    output_path="quantum_topology_animation.gif", fps=2, dark_mode=False):
    """
    Create an animation showing the evolution of quantum networks with topological features
    
    Parameters:
    -----------
    networks_sequence : list
        List of network states
    trust_matrices : list
        List of trust matrices for each state
    node_security : dict
        Dictionary of node security thresholds
    filtration_values : list
        List of filtration values to visualize
    betti_numbers_sequence : list
        List of Betti numbers for each state
    title : str, default="Quantum Topological Network Evolution"
        Title for the animation
    output_path : str, default="quantum_topology_animation.gif"
        Path to save the animation
    fps : int, default=2
        Frames per second
    dark_mode : bool, default=False
        Whether to use dark mode
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    """
    # Create figure and axes
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Main network visualization
    ax_network = plt.subplot(gs[0, 0])
    
    # Betti curves visualization
    ax_betti = plt.subplot(gs[0, 1])
    
    # Network statistics
    ax_stats = plt.subplot(gs[1, 0])
    
    # Persistence diagram
    ax_persistence = plt.subplot(gs[1, 1])
    
    # Set background color
    if dark_mode:
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        for ax in [ax_network, ax_betti, ax_stats, ax_persistence]:
            ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        for ax in [ax_network, ax_betti, ax_stats, ax_persistence]:
            ax.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
    
    # Create a consistent layout for all networks
    all_nodes = set()
    for networks in networks_sequence:
        for network in networks:
            all_nodes.update(network)
    
    G_full = nx.Graph()
    G_full.add_nodes_from(all_nodes)
    pos = nx.spring_layout(G_full, seed=42)
    
    def update(frame):
        # Clear all axes
        for ax in [ax_network, ax_betti, ax_stats, ax_persistence]:
            ax.clear()
        
        # Update network visualization
        networks = networks_sequence[frame]
        trust_matrix = trust_matrices[frame]
        betti_numbers = betti_numbers_sequence[frame]
        filtration_value = filtration_values[frame % len(filtration_values)]
        
        # Create the combined network for this frame
        G = nx.Graph()
        for network_idx, network in enumerate(networks):
            for node in network:
                G.add_node(node, network=network_idx)
        
        # Add edges based on trust matrix
        n_nodes = trust_matrix.shape[0]
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if i in G.nodes() and j in G.nodes():
                    avg_trust = (trust_matrix[i, j] + trust_matrix[j, i]) / 2
                    if avg_trust > 0.2:  # Only show significant trust relationships
                        G.add_edge(i, j, weight=avg_trust)
        
        # Color nodes by network
        network_colors = [QUANTUM_COLORS['networks'][G.nodes[n].get('network', 0) % len(QUANTUM_COLORS['networks'])] 
                         for n in G.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=network_colors, node_size=300, alpha=0.8, ax=ax_network)
        
        # Draw edges with varying width based on trust
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            avg_trust = G[u][v]['weight']
            edge_colors.append(avg_trust)
            edge_widths.append(1 + 3 * avg_trust)
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                              edge_cmap=quantum_cmap, alpha=0.6, ax=ax_network)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_color=text_color, ax=ax_network)
        
        # Add title with frame information
        ax_network.set_title(f"{title} - Step {frame + 1}/{len(networks_sequence)}", 
                           fontsize=14, color=text_color)
        
        # Add text for topological features
        betti_text = f"Betti Numbers: β₀={betti_numbers['betti0']}, β₁={betti_numbers['betti1']}, β₂={betti_numbers['betti2']}"
        ax_network.text(0.02, 0.02, betti_text, transform=ax_network.transAxes, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), color='black')
        
        ax_network.text(0.02, 0.96, f"Filtration value: {filtration_value:.2f}", 
                       transform=ax_network.transAxes, fontsize=12, color=text_color)
        
        # Remove axis
        ax_network.set_axis_off()
        
        # Plot Betti curves
        filtration_range = np.linspace(0, 1, 50)
        betti0_values = [max(0, betti_numbers['betti0'] - int(f > filtration_value) * (frame % 3)) for f in filtration_range]
        betti1_values = [max(0, int(f > 0.3) * betti_numbers['betti1'] - int(f > filtration_value) * (frame % 2)) for f in filtration_range]
        betti2_values = [max(0, int(f > 0.5) * betti_numbers['betti2']) for f in filtration_range]
        
        ax_betti.plot(filtration_range, betti0_values, '-', color=QUANTUM_COLORS['betti0'], lw=2, label='β₀ (Components)')
        ax_betti.plot(filtration_range, betti1_values, '-', color=QUANTUM_COLORS['betti1'], lw=2, label='β₁ (Cycles)')
        ax_betti.plot(filtration_range, betti2_values, '-', color=QUANTUM_COLORS['betti2'], lw=2, label='β₂ (Voids)')
        
        # Add vertical line for current filtration value
        ax_betti.axvline(x=filtration_value, color='red', linestyle='--', alpha=0.7)
        
        ax_betti.set_xlabel('Filtration Value', color=text_color)
        ax_betti.set_ylabel('Betti Number', color=text_color)
        ax_betti.set_title('Betti Curves', color=text_color)
        ax_betti.legend(loc='upper right')
        ax_betti.grid(True, alpha=0.3)
        
        # Set axis colors
        ax_betti.tick_params(axis='x', colors=text_color)
        ax_betti.tick_params(axis='y', colors=text_color)
        ax_betti.spines['bottom'].set_color(text_color)
        ax_betti.spines['top'].set_color(text_color)
        ax_betti.spines['left'].set_color(text_color)
        ax_betti.spines['right'].set_color(text_color)
        
        # Network statistics visualization
        # Calculate statistics for each network
        network_sizes = [len(net) for net in networks]
        network_labels = [f"Network {i}" for i in range(len(networks))]
        
        # Calculate average trust within each network
        network_trust = []
        for net_idx, network in enumerate(networks):
            if len(network) <= 1:
                network_trust.append(0)
                continue
                
            trust_sum = 0
            count = 0
            for i in network:
                for j in network:
                    if i != j:
                        trust_sum += trust_matrix[i, j]
                        count += 1
            
            if count > 0:
                network_trust.append(trust_sum / count)
            else:
                network_trust.append(0)
        
        # Plot network statistics
        bar_positions = np.arange(len(networks))
        bar_width = 0.4
        
        bars1 = ax_stats.bar(bar_positions - bar_width/2, network_sizes, bar_width, 
                            color=[QUANTUM_COLORS['networks'][i % len(QUANTUM_COLORS['networks'])] for i in range(len(networks))],
                            alpha=0.7, label='Network Size')
        
        # Create a second y-axis for trust values
        ax_stats_twin = ax_stats.twinx()
        bars2 = ax_stats_twin.bar(bar_positions + bar_width/2, network_trust, bar_width, 
                                color='lightgray', alpha=0.7, label='Avg Trust')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom', color=text_color, fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax_stats_twin.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                 f'{height:.2f}', ha='center', va='bottom', color=text_color, fontsize=9)
        
        ax_stats.set_xticks(bar_positions)
        ax_stats.set_xticklabels(network_labels, rotation=45, ha='right', color=text_color)
        ax_stats.set_ylabel('Network Size', color=text_color)
        ax_stats_twin.set_ylabel('Average Trust', color=text_color)
        ax_stats.set_title('Network Statistics', color=text_color)
        
        # Set axis colors
        ax_stats.tick_params(axis='x', colors=text_color)
        ax_stats.tick_params(axis='y', colors=text_color)
        ax_stats_twin.tick_params(axis='y', colors=text_color)
        
        # Legend with two dummy artists for the two bar types
        dummy_bars = [plt.Rectangle((0,0), 1, 1, fc=QUANTUM_COLORS['networks'][0], ec="none"),
                     plt.Rectangle((0,0), 1, 1, fc='lightgray', ec="none")]
        ax_stats.legend(dummy_bars, ['Network Size', 'Avg Trust'], loc='upper right')
        
        # Simulated persistence diagram (birth/death pairs)
        # Create simulated persistence points based on Betti numbers
        birth_death_pairs = []
        
        # Components (β₀)
        for i in range(betti_numbers['betti0']):
            birth = 0
            death = 0.3 + 0.5 * (i / max(1, betti_numbers['betti0']))
            birth_death_pairs.append((0, birth, death))
        
        # Cycles (β₁)
        for i in range(betti_numbers['betti1']):
            birth = 0.2 + 0.3 * (i / max(1, betti_numbers['betti1']))
            death = 0.6 + 0.3 * (i / max(1, betti_numbers['betti1']))
            birth_death_pairs.append((1, birth, death))
        
        # Voids (β₂)
        for i in range(betti_numbers['betti2']):
            birth = 0.4 + 0.2 * (i / max(1, betti_numbers['betti2']))
            death = 0.7 + 0.3 * (i / max(1, betti_numbers['betti2']))
            birth_death_pairs.append((2, birth, death))
        
        # Plot the persistence diagram
        ax_persistence.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)  # Diagonal
        
        # Different colors and markers for different dimensions
        markers = ['o', 's', '^']
        colors = [QUANTUM_COLORS['betti0'], QUANTUM_COLORS['betti1'], QUANTUM_COLORS['betti2']]
        labels = ['H₀', 'H₁', 'H₂']
        
        # Plot points by homology dimension
        for dim in range(3):
            points = [(b, d) for (dimension, b, d) in birth_death_pairs if dimension == dim]
            if points:
                births, deaths = zip(*points)
                ax_persistence.scatter(births, deaths, color=colors[dim], marker=markers[dim], 
                                    s=100, alpha=0.7, label=labels[dim])
        
        # Add vertical line for current filtration value
        ax_persistence.axvline(x=filtration_value, color='red', linestyle='--', alpha=0.7)
        ax_persistence.axhline(y=filtration_value, color='red', linestyle='--', alpha=0.7)
        
        ax_persistence.set_xlabel('Birth', color=text_color)
        ax_persistence.set_ylabel('Death', color=text_color)
        ax_persistence.set_title('Persistence Diagram', color=text_color)
        ax_persistence.set_xlim(0, 1)
        ax_persistence.set_ylim(0, 1)
        ax_persistence.legend(loc='lower right')
        ax_persistence.grid(True, alpha=0.3)
        
        # Set axis colors
        ax_persistence.tick_params(axis='x', colors=text_color)
        ax_persistence.tick_params(axis='y', colors=text_color)
        ax_persistence.spines['bottom'].set_color(text_color)
        ax_persistence.spines['top'].set_color(text_color)
        ax_persistence.spines['left'].set_color(text_color)
        ax_persistence.spines['right'].set_color(text_color)
        
        plt.tight_layout()
        return ax_network, ax_betti, ax_stats, ax_persistence
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(networks_sequence), blit=False)
    
    # Save animation
    anim.save(output_path, writer=animation.PillowWriter(fps=fps), 
             dpi=100, savefig_kwargs={'facecolor': fig.get_facecolor()})
    
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return anim

def create_quantum_filtration_animation(G, pos, distance_matrix, title="Quantum Filtration Evolution",
                                      output_path="quantum_filtration_animation.gif", fps=2,
                                      max_filtration=1.0, steps=10, dark_mode=False):
    """
    Create an animation showing the evolution of simplicial complex as filtration value changes
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph to visualize
    pos : dict
        Dictionary of node positions
    distance_matrix : numpy.ndarray
        Distance matrix between nodes
    title : str, default="Quantum Filtration Evolution"
        Title for the animation
    output_path : str, default="quantum_filtration_animation.gif"
        Path to save the animation
    fps : int, default=2
        Frames per second
    max_filtration : float, default=1.0
        Maximum filtration value
    steps : int, default=10
        Number of filtration steps
    dark_mode : bool, default=False
        Whether to use dark mode
        
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    if dark_mode:
        ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        ax.set_facecolor(QUANTUM_COLORS['background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
      # Set up simplicial complex computation using gudhi
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # Compute persistence once before animation
    simplex_tree.compute_persistence()
    
    filtration_values = np.linspace(0, max_filtration, steps)
    
    def update(frame):
        ax.clear()
        current_filtration = filtration_values[frame]
        
        # Get simplices at current filtration
        simplices = []
        for simplex, filtration in simplex_tree.get_filtration():
            if filtration <= current_filtration:
                simplices.append(simplex)
        
        # Create a graph with edges at the current filtration
        G_current = nx.Graph()
        G_current.add_nodes_from(G.nodes())
        
        # Add 1-simplices (edges)
        for simplex in simplices:
            if len(simplex) == 2:  # Edge
                G_current.add_edge(simplex[0], simplex[1])
        
        # Draw nodes
        nx.draw_networkx_nodes(G_current, pos, node_color=QUANTUM_COLORS['nodes'], 
                              node_size=300, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G_current, pos, edge_color=QUANTUM_COLORS['edges'], 
                              width=2, alpha=0.7, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G_current, pos, font_size=10, font_color=text_color, ax=ax)
          # Draw 2-simplices (triangles)
        for simplex in simplices:
            if len(simplex) == 3:  # Triangle
                triangle = np.array([pos[simplex[0]], pos[simplex[1]], pos[simplex[2]]])
                ax.fill(triangle[:, 0], triangle[:, 1], alpha=0.3, color=QUANTUM_COLORS['faces'])
                
        # Compute Betti numbers at this filtration
        try:
            betti_numbers = simplex_tree.persistent_betti_numbers(0, current_filtration)
            
            # Make sure we have all three Betti numbers, filling with zeros if needed
            b0 = betti_numbers[0] if len(betti_numbers) > 0 else 0
            b1 = betti_numbers[1] if len(betti_numbers) > 1 else 0
            b2 = betti_numbers[2] if len(betti_numbers) > 2 else 0
        except Exception as e:
            print(f"Error computing Betti numbers: {e}")
            b0, b1, b2 = 0, 0, 0
        
        # Add title with filtration value and Betti numbers
        ax.set_title(f"{title} - Filtration: {current_filtration:.2f}", fontsize=14, color=text_color)
        
        # Add text for topological features
        betti_text = f"Betti Numbers: β₀={b0}, β₁={b1}, β₂={b2}"
        ax.text(0.02, 0.02, betti_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), color='black')
        
        # Add progress indicator
        ax.text(0.02, 0.96, f"Step {frame + 1}/{len(filtration_values)}", 
               transform=ax.transAxes, fontsize=12, color=text_color)
        
        ax.set_axis_off()
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(filtration_values), blit=False)
    
    # Save animation
    anim.save(output_path, writer=animation.PillowWriter(fps=fps), 
             dpi=100, savefig_kwargs={'facecolor': fig.get_facecolor()})
    
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return anim

#########################
# Stability Analysis
#########################

def plot_quantum_stability_analysis(networks_history, trust_matrices, 
                                   perturbation_levels=None, betti_histories=None,
                                   title="Quantum Network Stability Analysis",
                                   output_path="quantum_stability_analysis.png", dark_mode=False):
    """
    Create stability analysis plots showing how networks and topological features respond to perturbations
    
    Parameters:
    -----------
    networks_history : list
        List of network sequences under different perturbation levels
    trust_matrices : list
        List of final trust matrices for each perturbation level
    perturbation_levels : list, optional
        List of perturbation levels
    betti_histories : list, optional
        List of Betti number sequences for each perturbation level
    title : str, default="Quantum Network Stability Analysis"
        Title for the plot
    output_path : str, default="quantum_stability_analysis.png"
        Path to save the figure
    dark_mode : bool, default=False
        Whether to use dark mode
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with stability analysis
    """
    if perturbation_levels is None:
        perturbation_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    # Network structure stability
    ax_network = plt.subplot(gs[0, 0])
    
    # Betti number stability
    ax_betti = plt.subplot(gs[0, 1])
    
    # Trust matrix heatmap
    ax_trust = plt.subplot(gs[1, 0])
    
    # Individual network size stability
    ax_sizes = plt.subplot(gs[1, 1])
    
    # Set background color
    if dark_mode:
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        for ax in [ax_network, ax_betti, ax_trust, ax_sizes]:
            ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        for ax in [ax_network, ax_betti, ax_trust, ax_sizes]:
            ax.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
    
    # Network structure stability plot
    # Measure of structure: number of networks and average size
    num_networks = [len(networks[-1]) for networks in networks_history]
    avg_sizes = [np.mean([len(net) for net in networks[-1]]) for networks in networks_history]
    
    ax_network.plot(perturbation_levels, num_networks, 'o-', color=QUANTUM_COLORS['betti0'], 
                   label='Number of Networks', linewidth=2, markersize=8)
    
    ax_network_twin = ax_network.twinx()
    ax_network_twin.plot(perturbation_levels, avg_sizes, 's--', color=QUANTUM_COLORS['betti1'], 
                        label='Avg Network Size', linewidth=2, markersize=8)
    
    ax_network.set_xlabel('Perturbation Level', color=text_color)
    ax_network.set_ylabel('Number of Networks', color=QUANTUM_COLORS['betti0'])
    ax_network_twin.set_ylabel('Avg Network Size', color=QUANTUM_COLORS['betti1'])
    ax_network.set_title('Network Structure Stability', color=text_color)
    ax_network.grid(True, alpha=0.3)
    
    # Add legends for both y-axes
    lines1, labels1 = ax_network.get_legend_handles_labels()
    lines2, labels2 = ax_network_twin.get_legend_handles_labels()
    ax_network.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set axis colors
    ax_network.tick_params(axis='x', colors=text_color)
    ax_network.tick_params(axis='y', colors=QUANTUM_COLORS['betti0'])
    ax_network_twin.tick_params(axis='y', colors=QUANTUM_COLORS['betti1'])
      # Betti number stability plot if betti_histories is available
    if betti_histories:
        # Extract final Betti numbers for each perturbation level
        betti0_values = [betti[-1]['betti0'] for betti in betti_histories]
        betti1_values = [betti[-1]['betti1'] for betti in betti_histories]
        betti2_values = [betti[-1]['betti2'] for betti in betti_histories]
        
        ax_betti.plot(perturbation_levels, betti0_values, 'o-', color=QUANTUM_COLORS['betti0'], 
                     label='β₀ (Components)', linewidth=2, markersize=8)
        ax_betti.plot(perturbation_levels, betti1_values, 's-', color=QUANTUM_COLORS['betti1'], 
                     label='β₁ (Cycles)', linewidth=2, markersize=8)
        ax_betti.plot(perturbation_levels, betti2_values, '^-', color=QUANTUM_COLORS['betti2'], 
                     label='β₂ (Voids)', linewidth=2, markersize=8)
        
        ax_betti.set_xlabel('Perturbation Level', color=text_color)
        ax_betti.set_ylabel('Betti Number', color=text_color)
        ax_betti.set_title('Topological Feature Stability', color=text_color)
        ax_betti.legend(loc='upper right')
        ax_betti.grid(True, alpha=0.3)
        
        # Set axis colors
        ax_betti.tick_params(axis='x', colors=text_color)
        ax_betti.tick_params(axis='y', colors=text_color)
    else:
        # If no Betti histories, show a message
        ax_betti.text(0.5, 0.5, "Topological analysis not used", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax_betti.transAxes, fontsize=14, color=text_color)
        ax_betti.set_title('Topological Feature Stability (Disabled)', color=text_color)
        ax_betti.set_axis_off()
    
    # Trust matrix heatmap (use the middle perturbation level)
    mid_idx = len(perturbation_levels) // 2
    trust_matrix = trust_matrices[mid_idx]
    
    im = ax_trust.imshow(trust_matrix, cmap=quantum_diverging, interpolation='nearest', 
                        vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_trust)
    cbar.set_label('Trust Value', color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=text_color)
    
    # Add grid to show cell boundaries
    ax_trust.set_xticks(np.arange(trust_matrix.shape[1]) - 0.5, minor=True)
    ax_trust.set_yticks(np.arange(trust_matrix.shape[0]) - 0.5, minor=True)
    ax_trust.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
    
    ax_trust.set_title('Trust Matrix Heatmap', color=text_color)
    ax_trust.set_xlabel('Node ID', color=text_color)
    ax_trust.set_ylabel('Node ID', color=text_color)
    
    # Set tick colors
    ax_trust.tick_params(axis='x', colors=text_color)
    ax_trust.tick_params(axis='y', colors=text_color)
    
    # Individual network size stability
    # Get sizes of networks across perturbation levels (up to 5 networks)
    max_networks = 5
    network_sizes = []
    
    for level_idx, networks in enumerate(networks_history):
        final_networks = networks[-1]
        sizes = [len(net) for net in final_networks[:max_networks]]
        # Pad with zeros if fewer than max_networks
        sizes.extend([0] * (max_networks - len(sizes)))
        network_sizes.append(sizes)
    
    network_sizes = np.array(network_sizes).T  # Transpose for stacking
    
    # Plot stacked bars for each network's size
    bottom = np.zeros(len(perturbation_levels))
    
    for i in range(max_networks):
        ax_sizes.bar(perturbation_levels, network_sizes[i], bottom=bottom, 
                    color=QUANTUM_COLORS['networks'][i % len(QUANTUM_COLORS['networks'])],
                    label=f'Network {i+1}', alpha=0.7)
        bottom += network_sizes[i]
    
    ax_sizes.set_xlabel('Perturbation Level', color=text_color)
    ax_sizes.set_ylabel('Network Size', color=text_color)
    ax_sizes.set_title('Individual Network Size Stability', color=text_color)
    ax_sizes.legend(loc='upper right')
    ax_sizes.grid(True, alpha=0.3)
    
    # Set axis colors
    ax_sizes.tick_params(axis='x', colors=text_color)
    ax_sizes.tick_params(axis='y', colors=text_color)
    
    # Set overall title
    fig.suptitle(title, fontsize=16, color=text_color)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"Stability analysis saved to {output_path}")
    return fig

#########################
# Helper Functions
#########################

def plot_betti_curves(filtration_values, betti_numbers, title="Quantum Betti Curves",
                     output_path=None, dark_mode=False):
    """
    Plot Betti curves showing how Betti numbers change with filtration value
    
    Parameters:
    -----------
    filtration_values : list
        List of filtration values
    betti_numbers : list
        List of dictionaries with betti0, betti1, betti2 keys
    title : str, default="Quantum Betti Curves"
        Title for the plot
    output_path : str, optional
        Path to save the figure
    dark_mode : bool, default=False
        Whether to use dark mode
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with Betti curves
    """
    fig, ax = plt.subplots(figsize=(10, 8))
      # Set background color
    if dark_mode:
        ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        ax.set_facecolor(QUANTUM_COLORS['background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
        
    # Extract Betti numbers for each dimension
    try:
        betti0 = [b.get('betti0', 0) for b in betti_numbers]
        betti1 = [b.get('betti1', 0) for b in betti_numbers]
        betti2 = [b.get('betti2', 0) for b in betti_numbers]
    except Exception as e:
        print(f"Error extracting Betti numbers: {e}")
        betti0 = betti1 = betti2 = [0] * len(filtration_values)
    
    # Plot the curves
    ax.plot(filtration_values, betti0, 'o-', color=QUANTUM_COLORS['betti0'], 
           label='β₀ (Components)', linewidth=2, markersize=6)
    ax.plot(filtration_values, betti1, 's-', color=QUANTUM_COLORS['betti1'], 
           label='β₁ (Cycles)', linewidth=2, markersize=6)
    ax.plot(filtration_values, betti2, '^-', color=QUANTUM_COLORS['betti2'], 
           label='β₂ (Voids)', linewidth=2, markersize=6)
    
    ax.set_xlabel('Filtration Value', color=text_color)
    ax.set_ylabel('Betti Number', color=text_color)
    ax.set_title(title, color=text_color)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
      # Set axis colors
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
        print(f"Betti curves saved to {output_path}")
    
    return fig, ax

def plot_persistence_diagram(persistence_pairs, title="Quantum Persistence Diagram", 
                            current_filtration=None, output_path=None, dark_mode=False):
    """
    Plot a persistence diagram showing birth/death pairs of topological features
    
    Parameters:
    -----------
    persistence_pairs : list
        List of (dimension, birth, death) tuples
    title : str, default="Quantum Persistence Diagram"
        Title for the plot
    current_filtration : float, optional
        Current filtration value to highlight
    output_path : str, optional
        Path to save the figure
    dark_mode : bool, default=False
        Whether to use dark mode
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with persistence diagram
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set background color
    if dark_mode:
        ax.set_facecolor(QUANTUM_COLORS['dark_background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['dark_background'])
        text_color = 'white'
    else:
        ax.set_facecolor(QUANTUM_COLORS['background'])
        fig.patch.set_facecolor(QUANTUM_COLORS['background'])
        text_color = 'black'
    
    # Plot the diagonal
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    
    # Different colors and markers for different dimensions
    markers = ['o', 's', '^']
    colors = [QUANTUM_COLORS['betti0'], QUANTUM_COLORS['betti1'], QUANTUM_COLORS['betti2']]
    labels = ['H₀ (Components)', 'H₁ (Cycles)', 'H₂ (Voids)']
    
    # Group points by dimension
    dim_points = {0: [], 1: [], 2: []}
    for dim, birth, death in persistence_pairs:
        if dim <= 2:  # Only plot up to dimension 2
            dim_points[dim].append((birth, death))
    
    # Plot points by dimension
    for dim in range(3):
        points = dim_points[dim]
        if points:
            births, deaths = zip(*points)
            ax.scatter(births, deaths, color=colors[dim], marker=markers[dim], 
                      s=100, alpha=0.7, label=labels[dim])
    
    # Add lines for current filtration if provided
    if current_filtration is not None:
        ax.axvline(x=current_filtration, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=current_filtration, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Birth', color=text_color)
    ax.set_ylabel('Death', color=text_color)
    ax.set_title(title, color=text_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set axis colors
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
        print(f"Persistence diagram saved to {output_path}")
    
    return fig, ax

if __name__ == "__main__":
    # Simple demonstration if run directly
    print("This module provides visualization functions for quantum topological trust partitioning.")
    print("Import it into your main script to use its functionality.")

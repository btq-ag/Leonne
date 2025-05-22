#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkCommunities.py

This script creates animated visualizations of different network community types,
inspired by the network community typology from neuroscience. The visualizations
match the style shown in the reference image and demonstrate the distinctive 
structural properties of each network model.

Network types included:
1. Random (Erdős–Rényi model) - No structure
2. Community structure (Stochastic block model) - Clustered groups
3. Small-world (Watts–Strogatz model) - Efficient communication
4. Hub structure (Barabási–Albert model) - Heavy-tailed degree distribution
5. Spatially embedded (Spatial model) - Physically constrained

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
from scipy.spatial import Voronoi
from matplotlib import cm
import os
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
# Remove the line that creates a subdirectory

# Set random seed for reproducibility
np.random.seed(42)

# Color schemes to match the reference image
COLOR_SCHEMES = {
    'random': {'nodes': 'black', 'edges': 'black', 'background': '#f5e8d0'},
    'community': {'nodes': 'black', 'edges': 'black', 'background': '#f5e8d0', 
                 'communities': ['#e7cba9', '#d2b48c', '#c19a6b']},
    'small_world': {'nodes': 'black', 'edges': 'black', 'background': '#f5e8d0', 
                   'shortcut': '#ff7f7f'},
    'hub': {'nodes': 'black', 'edges': 'black', 'background': '#f5e8d0', 
           'hub_node': '#8b0000'},
    'spatial': {'nodes': 'black', 'edges': 'black', 'background': '#f5e8d0', 
               'region': '#e6ceff'}
}

def save_animation(anim, filename):
    """Save animation to file"""
    filepath = os.path.join(output_dir, filename)
    anim.save(filepath, writer='pillow', fps=8, dpi=120)
    print(f"Animation saved to {filepath}")
    plt.close()

#########################
# 1. Random Network (Erdős–Rényi Model)
#########################

def create_random_network_animation(n=25, p_final=0.15, n_frames=30):
    """Create animation showing random network formation (Erdős–Rényi model)"""
    
    # Initialize graph with nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Set up the figure with the beige background from reference
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor(COLOR_SCHEMES['random']['background'])
    
    # Add border to match reference image
    border_color = '#d2b48c'  # Darker beige border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Get uniformly distributed positions across the figure
    pos = {}
    for i in range(n):
        angle = 2 * np.pi * i / n
        r = 0.65 + 0.2 * np.random.random()  # Slightly randomize radius
        pos[i] = (r * np.cos(angle), r * np.sin(angle))
    
    # Pre-compute edge appearances
    all_possible_edges = list(nx.non_edges(G))
    np.random.shuffle(all_possible_edges)
    
    edges_per_frame = []
    total_edges = int(p_final * n * (n-1) / 2)
    edges_to_add = [int(i * total_edges / (n_frames-1)) for i in range(n_frames)]
    
    for i in range(n_frames):
        if i == 0:
            edges_per_frame.append([])
        else:
            start_idx = edges_to_add[i-1]
            end_idx = edges_to_add[i]
            edges_per_frame.append(all_possible_edges[start_idx:end_idx])
    
    def update(frame):
        ax.clear()
        # Redraw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        
        # Create frame graph
        G_frame = nx.Graph()
        G_frame.add_nodes_from(range(n))
        for i in range(frame+1):
            G_frame.add_edges_from(edges_per_frame[i])
        
        # Draw nodes as black circles
        nx.draw_networkx_nodes(G_frame, pos, 
                              node_color=COLOR_SCHEMES['random']['nodes'],
                              node_size=250,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw edges as thick black lines
        nx.draw_networkx_edges(G_frame, pos,
                              edge_color=COLOR_SCHEMES['random']['edges'],
                              width=2.5)
        
        # Remove labels and set title
        if frame == 0:
            ax.set_title("Random\n(Erdős–Rényi)", fontsize=16, pad=20)
        else:
            ax.set_title("Random\n(Erdős–Rényi)", fontsize=16, pad=20)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    return anim

#########################
# 2. Community Structure (Stochastic Block Model)
#########################

def create_community_network_animation(communities=3, nodes_per_community=8, n_frames=30):
    """Create animation showing community structure network formation"""
    
    # Initialize parameters
    n = communities * nodes_per_community
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor(COLOR_SCHEMES['community']['background'])
    
    # Add border
    border_color = '#d2b48c'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Assign community membership
    node_community = {}
    community_nodes = {}
    for i in range(communities):
        nodes = list(range(i*nodes_per_community, (i+1)*nodes_per_community))
        community_nodes[i] = nodes
        for node in nodes:
            node_community[node] = i
    
    # Generate positions - communities arranged in a circle
    pos = {}
    for comm_idx, nodes in community_nodes.items():
        # Position the community
        comm_angle = 2 * np.pi * comm_idx / communities
        comm_x = np.cos(comm_angle) * 0.6
        comm_y = np.sin(comm_angle) * 0.6
        
        # Position nodes within community
        for i, node in enumerate(nodes):
            node_angle = 2 * np.pi * i / len(nodes)
            node_r = 0.25  # Community radius
            pos[node] = (comm_x + node_r * np.cos(node_angle), 
                         comm_y + node_r * np.sin(node_angle))
    
    # Pre-compute edge appearances
    # First add intra-community edges, then inter-community
    intra_community_edges = []
    inter_community_edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            if node_community[i] == node_community[j]:
                intra_community_edges.append((i, j))
            else:
                # Add fewer inter-community edges
                if np.random.random() < 0.2:  # Only 20% of possible inter-edges
                    inter_community_edges.append((i, j))
    
    np.random.shuffle(intra_community_edges)
    np.random.shuffle(inter_community_edges)
    
    # Determine edges per frame
    intra_per_frame = [[] for _ in range(n_frames)]
    inter_per_frame = [[] for _ in range(n_frames)]
    
    # First half of frames: add intra-community edges
    intra_frames = n_frames // 2
    intra_per_stage = len(intra_community_edges) // intra_frames
    
    for i in range(intra_frames):
        start_idx = i * intra_per_stage
        end_idx = start_idx + intra_per_stage
        intra_per_frame[i] = intra_community_edges[start_idx:end_idx]
    
    # Second half: add inter-community edges
    inter_frames = n_frames - intra_frames
    inter_per_stage = len(inter_community_edges) // inter_frames
    
    for i in range(inter_frames):
        frame_idx = i + intra_frames
        start_idx = i * inter_per_stage
        end_idx = start_idx + inter_per_stage
        inter_per_frame[frame_idx] = inter_community_edges[start_idx:end_idx]
    
    def update(frame):
        ax.clear()
        # Redraw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        
        # Create frame graph
        G_frame = nx.Graph()
        G_frame.add_nodes_from(range(n))
        
        for i in range(frame+1):
            G_frame.add_edges_from(intra_per_frame[i])
            G_frame.add_edges_from(inter_per_frame[i])
        
        # Draw community backgrounds
        for i, nodes in community_nodes.items():
            node_positions = np.array([pos[n] for n in nodes])
            centroid = np.mean(node_positions, axis=0)
            
            # Create an ellipse for each community
            community_color = COLOR_SCHEMES['community']['communities'][i % len(COLOR_SCHEMES['community']['communities'])]
            ellipse = Ellipse(centroid, width=0.6, height=0.6,
                             fill=True, alpha=0.3, color=community_color)
            ax.add_patch(ellipse)
        
        # Draw edges
        nx.draw_networkx_edges(G_frame, pos,
                              edge_color='black',
                              width=2.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_frame, pos,
                              node_color='black',
                              node_size=250,
                              edgecolors='black',
                              linewidths=2)
        
        # Set title and limits
        ax.set_title("Community Structure\n(Stochastic Block Model)", fontsize=16, pad=20)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    return anim

#########################
# 3. Small-World Network (Watts–Strogatz)
#########################

def create_small_world_animation(n=20, k=4, p_rewire=0.2, n_frames=30):
    """Create animation showing small-world network formation"""
    
    # Create initial ring lattice
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add ring lattice connections
    for i in range(n):
        for j in range(1, k//2 + 1):
            G.add_edge(i, (i+j) % n)
            G.add_edge(i, (i-j) % n)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor(COLOR_SCHEMES['small_world']['background'])
    
    # Add border
    border_color = '#d2b48c'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Generate positions (perfect circle)
    pos = {}
    for i in range(n):
        angle = 2 * np.pi * i / n
        pos[i] = (np.cos(angle), np.sin(angle))
    
    # Store original edges
    original_edges = list(G.edges())
    
    # Pre-compute rewirings
    edges_to_rewire = []
    for u, v in original_edges:
        if u < v and np.random.random() < p_rewire:  # Only consider each edge once
            potential_targets = [w for w in range(n) if w != u and w != v and not G.has_edge(u, w)]
            if potential_targets:
                new_target = np.random.choice(potential_targets)
                edges_to_rewire.append((u, v, new_target))  # (from, old_to, new_to)
    
    # Distribute rewirings across frames
    rewirings_per_frame = []
    rewirings_per_frame.append([])  # First frame has no rewirings
    
    # Rest of frames have progressive rewirings
    rewirings_per_stage = max(1, len(edges_to_rewire) // (n_frames-1))
    for i in range(1, n_frames):
        start_idx = (i-1) * rewirings_per_stage
        end_idx = min(start_idx + rewirings_per_stage, len(edges_to_rewire))
        if start_idx < len(edges_to_rewire):
            rewirings_per_frame.append(edges_to_rewire[start_idx:end_idx])
        else:
            rewirings_per_frame.append([])
    
    def update(frame):
        ax.clear()
        # Redraw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        
        # Apply rewirings up to this frame
        G_frame = nx.Graph()
        G_frame.add_nodes_from(range(n))
        G_frame.add_edges_from(original_edges)
        
        # Keep track of rewired edges and new shortcut edges
        rewired_edges = set()
        shortcuts = []
        
        for i in range(1, frame+1):
            for u, old_v, new_v in rewirings_per_frame[i]:
                if G_frame.has_edge(u, old_v):
                    G_frame.remove_edge(u, old_v)
                    rewired_edges.add((u, old_v))
                    rewired_edges.add((old_v, u))
                
                G_frame.add_edge(u, new_v)
                shortcuts.append((u, new_v))
        
        # Draw regular ring lattice edges
        regular_edges = [(u, v) for u, v in G_frame.edges() 
                         if (u, v) not in shortcuts and (v, u) not in shortcuts]
        
        nx.draw_networkx_edges(G_frame, pos,
                              edgelist=regular_edges,
                              edge_color='black',
                              width=2.5)
        
        # Draw shortcut edges in a different color/style
        nx.draw_networkx_edges(G_frame, pos,
                              edgelist=shortcuts,
                              edge_color=COLOR_SCHEMES['small_world']['shortcut'],
                              width=2.5,
                              style='dashed')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_frame, pos,
                              node_color='black',
                              node_size=250,
                              edgecolors='black',
                              linewidths=2)
        
        # Set title and limits
        ax.set_title("Small-world\n(Watts–Strogatz)", fontsize=16, pad=20)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    return anim

#########################
# 4. Hub Structure (Barabási–Albert)
#########################

def create_hub_network_animation(n=25, m=1, n_frames=30):
    """Create animation showing hub structure network formation"""
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor(COLOR_SCHEMES['hub']['background'])
    
    # Add border
    border_color = '#d2b48c'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Generate the full network evolution
    G = nx.barabasi_albert_graph(n, m, seed=42)
    
    # Compute positions with the central hub in the middle
    # Identify the hub (node with highest degree)
    degrees = dict(G.degree())
    hub_node = max(degrees, key=degrees.get)
    
    # Custom layout: hub in center, others in a circle
    pos = {}
    pos[hub_node] = (0, 0)  # Hub at center
    
    # Position other nodes in a circle around the hub
    other_nodes = [node for node in G.nodes() if node != hub_node]
    for i, node in enumerate(other_nodes):
        angle = 2 * np.pi * i / len(other_nodes)
        r = 0.8  # Radius of circle
        pos[node] = (r * np.cos(angle), r * np.sin(angle))
    
    # Precompute graph evolution
    graphs = []
    for i in range(m+1, n+1):
        g = nx.barabasi_albert_graph(i, m, seed=42)
        graphs.append(g)
    
    # Distribute evolution across frames
    if len(graphs) > n_frames:
        indices = np.linspace(0, len(graphs)-1, n_frames, dtype=int)
        graphs = [graphs[i] for i in indices]
    else:
        # Pad with duplicates of the last graph
        last_graph = graphs[-1]
        while len(graphs) < n_frames:
            graphs.append(last_graph)
    
    def update(frame):
        ax.clear()
        # Redraw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        
        G_frame = graphs[frame]
        
        # Get node sizes based on degree
        frame_degrees = dict(G_frame.degree())
        node_sizes = {}
        for node in G_frame.nodes():
            # Base size with bonus for degree
            node_sizes[node] = 150 + 30 * frame_degrees[node]
        
        # Draw edges
        nx.draw_networkx_edges(G_frame, pos,
                              edge_color='black',
                              width=2.5)
        
        # Draw nodes with size proportional to degree
        # Regular nodes
        regular_nodes = [n for n in G_frame.nodes() if frame_degrees[n] < max(frame_degrees.values())]
        nx.draw_networkx_nodes(G_frame, pos, 
                              nodelist=regular_nodes,
                              node_color='black',
                              node_size=[node_sizes[n] for n in regular_nodes],
                              edgecolors='black',
                              linewidths=2)
        
        # Hub nodes (nodes with max degree)
        hub_nodes = [n for n in G_frame.nodes() if frame_degrees[n] == max(frame_degrees.values())]
        if hub_nodes:
            nx.draw_networkx_nodes(G_frame, pos, 
                                  nodelist=hub_nodes,
                                  node_color=COLOR_SCHEMES['hub']['hub_node'],
                                  node_size=[node_sizes[n] for n in hub_nodes],
                                  edgecolors='black',
                                  linewidths=2)
        
        # Set title and limits
        ax.set_title("Hub Structure\n(Barabási–Albert)", fontsize=16, pad=20)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    return anim

#########################
# 5. Spatial Network
#########################

def create_spatial_network_animation(n=25, radius_final=0.5, n_frames=30):
    """Create animation showing spatially embedded network formation"""
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor(COLOR_SCHEMES['spatial']['background'])
    
    # Add border
    border_color = '#d2b48c'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Generate brain-like shape
    positions = []
    while len(positions) < n:
        # Generate points biased toward an ellipsoid brain shape
        x = np.random.normal(0, 0.4)  # Mean 0, std 0.4
        y = np.random.normal(0, 0.3)  # Mean 0, std 0.3 (narrower)
        
        # Scale x to make it more ellipsoidal
        x = x * 1.5
        
        # Only keep points within reasonable bounds
        if abs(x) < 0.9 and abs(y) < 0.7:
            positions.append((x, y))
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Store positions
    pos = {i: positions[i] for i in range(n)}
    
    # Create radius thresholds for each frame
    radius_values = np.linspace(0, radius_final, n_frames)
    
    # Pre-compute edges for each frame
    edges_by_frame = []
    
    for radius in radius_values:
        frame_edges = []
        for i in range(n):
            for j in range(i+1, n):
                # Calculate Euclidean distance
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= radius:
                    frame_edges.append((i, j))
        edges_by_frame.append(frame_edges)
    
    def update(frame):
        ax.clear()
        # Redraw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        
        # Create brain-shaped background
        brain_shape = plt.Polygon([(-1, -0.7), (-0.5, -0.85), (0, -0.9), 
                                 (0.5, -0.85), (1, -0.7), (1.1, -0.5), 
                                 (1.1, 0.1), (1, 0.4), (0.7, 0.6),
                                 (0.3, 0.7), (-0.3, 0.7), (-0.7, 0.6),
                                 (-1, 0.4), (-1.1, 0.1), (-1.1, -0.5)],
                                 closed=True, alpha=0.2, 
                                 color=COLOR_SCHEMES['spatial']['region'])
        ax.add_patch(brain_shape)
        
        # Create frame graph
        G_frame = nx.Graph()
        G_frame.add_nodes_from(range(n))
        G_frame.add_edges_from(edges_by_frame[frame])
        
        # Draw edges
        nx.draw_networkx_edges(G_frame, pos,
                              edge_color='black',
                              width=2.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_frame, pos,
                              node_color='black',
                              node_size=250,
                              edgecolors='black',
                              linewidths=2)
        
        # Set title and limits
        ax.set_title("Spatially Embedded\n(Spatial Model)", fontsize=16, pad=20)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    return anim

#########################
# Combined visualization of all network types
#########################

def create_combined_visualization():
    """Create a static figure comparing all five network types side by side"""
    
    # Set up figure with 5 subplots in a row
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor='white')
    
    # Style settings for all plots
    border_color = '#d2b48c'
    
    # 1. Random (Erdős–Rényi)
    G_random = nx.gnp_random_graph(25, 0.15, seed=42)
    pos_random = nx.spring_layout(G_random, seed=42)
    
    axes[0].set_facecolor(COLOR_SCHEMES['random']['background'])
    # Add border
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    nx.draw_networkx_nodes(G_random, pos_random, ax=axes[0],
                         node_color='black',
                         node_size=200,
                         edgecolors='black',
                         linewidths=2)
    nx.draw_networkx_edges(G_random, pos_random, ax=axes[0],
                         edge_color='black',
                         width=2.5)
    axes[0].set_title("Random\n(Erdős–Rényi)", fontsize=16, pad=20)
    axes[0].axis('off')
    
    # 2. Community (Stochastic Block Model)
    communities = 3
    nodes_per_community = 8
    n = communities * nodes_per_community
    
    # Create block model with high intra-community probability and low inter-community
    sizes = [nodes_per_community] * communities
    p = np.zeros((communities, communities))
    for i in range(communities):
        for j in range(communities):
            if i == j:  # Intra-community
                p[i][j] = 0.7
            else:  # Inter-community
                p[i][j] = 0.05
    
    G_community = nx.stochastic_block_model(sizes, p, seed=42)
    
    # Position nodes in community clusters
    node_community = {}
    community_nodes = {}
    for i in range(communities):
        nodes = list(range(i*nodes_per_community, (i+1)*nodes_per_community))
        community_nodes[i] = nodes
        for node in nodes:
            node_community[node] = i
    
    pos_community = {}
    for comm_idx, nodes in community_nodes.items():
        # Position the community
        comm_angle = 2 * np.pi * comm_idx / communities
        comm_x = np.cos(comm_angle) * 0.6
        comm_y = np.sin(comm_angle) * 0.6
        
        # Position nodes within community
        for i, node in enumerate(nodes):
            node_angle = 2 * np.pi * i / len(nodes)
            node_r = 0.25  # Community radius
            pos_community[node] = (comm_x + node_r * np.cos(node_angle), 
                                 comm_y + node_r * np.sin(node_angle))
    
    axes[1].set_facecolor(COLOR_SCHEMES['community']['background'])
    # Add border
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Draw community backgrounds
    for i, nodes in community_nodes.items():
        node_positions = np.array([pos_community[n] for n in nodes])
        centroid = np.mean(node_positions, axis=0)
        
        # Create an ellipse for each community
        community_color = COLOR_SCHEMES['community']['communities'][i % len(COLOR_SCHEMES['community']['communities'])]
        ellipse = Ellipse(centroid, width=0.6, height=0.6,
                         fill=True, alpha=0.3, color=community_color)
        axes[1].add_patch(ellipse)
    
    nx.draw_networkx_edges(G_community, pos_community, ax=axes[1],
                         edge_color='black',
                         width=2.5)
    nx.draw_networkx_nodes(G_community, pos_community, ax=axes[1],
                         node_color='black',
                         node_size=200,
                         edgecolors='black',
                         linewidths=2)
    axes[1].set_title("Community Structure\n(Stochastic Block Model)", fontsize=16, pad=20)
    axes[1].axis('off')
    
    # 3. Small-World (Watts–Strogatz)
    G_small_world = nx.watts_strogatz_graph(20, 4, 0.2, seed=42)
    pos_small_world = nx.circular_layout(G_small_world)
    
    axes[2].set_facecolor(COLOR_SCHEMES['small_world']['background'])
    # Add border
    for spine in axes[2].spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Identify shortcut edges vs regular lattice edges
    # Calculate path lengths in the ring
    ring_dists = {}
    for u, v in G_small_world.edges():
        ring_dist = min((u - v) % 20, (v - u) % 20)
        ring_dists[(u, v)] = ring_dist
    
    # Edges with ring_dist > k/2 are shortcuts
    shortcuts = [(u, v) for (u, v), dist in ring_dists.items() if dist > 2]
    regular_edges = [(u, v) for (u, v), dist in ring_dists.items() if dist <= 2]
    
    nx.draw_networkx_edges(G_small_world, pos_small_world, ax=axes[2],
                         edgelist=regular_edges,
                         edge_color='black',
                         width=2.5)
    nx.draw_networkx_edges(G_small_world, pos_small_world, ax=axes[2],
                         edgelist=shortcuts,
                         edge_color=COLOR_SCHEMES['small_world']['shortcut'],
                         width=2.5,
                         style='dashed')
    nx.draw_networkx_nodes(G_small_world, pos_small_world, ax=axes[2],
                         node_color='black',
                         node_size=200,
                         edgecolors='black',
                         linewidths=2)
    axes[2].set_title("Small-world\n(Watts–Strogatz)", fontsize=16, pad=20)
    axes[2].axis('off')
    
    # 4. Hub Structure (Barabási–Albert)
    G_hub = nx.barabasi_albert_graph(25, 1, seed=42)
    
    # Identify the hub
    degrees = dict(G_hub.degree())
    hub_node = max(degrees, key=degrees.get)
    
    # Custom layout: hub in center, others in a circle
    pos_hub = {}
    pos_hub[hub_node] = (0, 0)  # Hub at center
    
    # Position other nodes in a circle around the hub
    other_nodes = [node for node in G_hub.nodes() if node != hub_node]
    for i, node in enumerate(other_nodes):
        angle = 2 * np.pi * i / len(other_nodes)
        r = 0.8  # Radius of circle
        pos_hub[node] = (r * np.cos(angle), r * np.sin(angle))
    
    axes[3].set_facecolor(COLOR_SCHEMES['hub']['background'])
    # Add border
    for spine in axes[3].spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Get node sizes based on degree
    node_sizes = {}
    for node in G_hub.nodes():
        # Base size with bonus for degree
        node_sizes[node] = 150 + 30 * degrees[node]
    
    # Draw edges
    nx.draw_networkx_edges(G_hub, pos_hub, ax=axes[3],
                         edge_color='black',
                         width=2.5)
    
    # Draw regular nodes
    regular_nodes = [n for n in G_hub.nodes() if degrees[n] < max(degrees.values())]
    nx.draw_networkx_nodes(G_hub, pos_hub, ax=axes[3],
                         nodelist=regular_nodes,
                         node_color='black',
                         node_size=[node_sizes[n] for n in regular_nodes],
                         edgecolors='black',
                         linewidths=2)
    
    # Draw hub nodes
    hub_nodes = [n for n in G_hub.nodes() if degrees[n] == max(degrees.values())]
    nx.draw_networkx_nodes(G_hub, pos_hub, ax=axes[3],
                         nodelist=hub_nodes,
                         node_color=COLOR_SCHEMES['hub']['hub_node'],
                         node_size=[node_sizes[n] for n in hub_nodes],
                         edgecolors='black',
                         linewidths=2)
    
    axes[3].set_title("Hub Structure\n(Barabási–Albert)", fontsize=16, pad=20)
    axes[3].axis('off')
    
    # 5. Spatial Network
    # Generate brain-like positions
    positions = []
    while len(positions) < 25:
        # Generate points biased toward an ellipsoid brain shape
        x = np.random.normal(0, 0.4)
        y = np.random.normal(0, 0.3)
        
        # Scale x to make it more ellipsoidal
        x = x * 1.5
        
        # Only keep points within reasonable bounds
        if abs(x) < 0.9 and abs(y) < 0.7:
            positions.append((x, y))
    
    G_spatial = nx.random_geometric_graph(25, 0.5, pos=dict(enumerate(positions)), seed=42)
    pos_spatial = {i: positions[i] for i in range(25)}
    
    axes[4].set_facecolor(COLOR_SCHEMES['spatial']['background'])
    # Add border
    for spine in axes[4].spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Create brain-shaped background
    brain_shape = plt.Polygon([(-1, -0.7), (-0.5, -0.85), (0, -0.9), 
                             (0.5, -0.85), (1, -0.7), (1.1, -0.5), 
                             (1.1, 0.1), (1, 0.4), (0.7, 0.6),
                             (0.3, 0.7), (-0.3, 0.7), (-0.7, 0.6),
                             (-1, 0.4), (-1.1, 0.1), (-1.1, -0.5)],
                             closed=True, alpha=0.2, 
                             color=COLOR_SCHEMES['spatial']['region'])
    axes[4].add_patch(brain_shape)
    
    nx.draw_networkx_edges(G_spatial, pos_spatial, ax=axes[4],
                         edge_color='black',
                         width=2.5)
    nx.draw_networkx_nodes(G_spatial, pos_spatial, ax=axes[4],
                         node_color='black',
                         node_size=200,
                         edgecolors='black',
                         linewidths=2)
    axes[4].set_title("Spatially Embedded\n(Spatial Model)", fontsize=16, pad=20)
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_network_types.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {os.path.join(output_dir, 'all_network_types.png')}")
    return

#########################
# Main function to generate all animations
#########################

def generate_all_animations():
    """Generate and save all network community type animations"""
    
    print("Generating random network animation...")
    random_anim = create_random_network_animation(n=25, p_final=0.15)
    save_animation(random_anim, "random_network.gif")
    
    print("Generating community structure network animation...")
    community_anim = create_community_network_animation(communities=3, nodes_per_community=8)
    save_animation(community_anim, "community_network.gif")
    
    print("Generating small-world network animation...")
    small_world_anim = create_small_world_animation(n=20, k=4, p_rewire=0.2)
    save_animation(small_world_anim, "small_world_network.gif")
    
    print("Generating hub structure network animation...")
    # Draw nodes colored by community
    node_colors = []
    for node in G_community.nodes():
        # Determine which community this node belongs to
        for i, community in enumerate(nx.community.greedy_modularity_communities(G_community)):
            if node in community:
                node_colors.append(COLOR_SCHEMES['community']['communities'][i % len(COLOR_SCHEMES['community']['communities'])])
                break
    
    nx.draw_networkx_nodes(G_community, pos_community, ax=axes[1],
                         node_color=node_colors,
                         node_size=60, alpha=0.9)
    nx.draw_networkx_edges(G_community, pos_community, ax=axes[1],
                         edge_color='gray',
                         width=0.8, alpha=0.6)
    
    axes[1].set_title("Community Structure\n(Stochastic Block Model)", fontsize=14)
    axes[1].axis('off')
    
    # 3. Small-World (Watts–Strogatz)
    G_small_world = nx.watts_strogatz_graph(40, 4, 0.3, seed=42)
    pos_small_world = nx.circular_layout(G_small_world)
    
    axes[2].set_facecolor(COLOR_SCHEMES['small_world']['background'])
    nx.draw_networkx_nodes(G_small_world, pos_small_world, ax=axes[2],
                         node_color=COLOR_SCHEMES['small_world']['nodes'],
                         node_size=60, alpha=0.9)
    nx.draw_networkx_edges(G_small_world, pos_small_world, ax=axes[2],
                         edge_color=COLOR_SCHEMES['small_world']['edges'],
                         width=0.8, alpha=0.6)
    axes[2].set_title("Small-World\n(Watts–Strogatz)", fontsize=14)
    axes[2].axis('off')
    
    # 4. Hub Structure (Barabási–Albert)
    G_hub = nx.barabasi_albert_graph(40, 2, seed=42)
    pos_hub = nx.spring_layout(G_hub, seed=42)
    
    # Calculate node sizes based on degree
    degrees = dict(G_hub.degree())
    node_sizes = [20 + 60 * (degrees[node] / max(degrees.values())) for node in G_hub.nodes()]
    
    # Color nodes based on degree
    threshold = max(degrees.values()) * 0.7
    node_colors = [COLOR_SCHEMES['hub']['hub'] if degrees[node] > threshold 
                  else COLOR_SCHEMES['hub']['nodes'] for node in G_hub.nodes()]
    
    axes[3].set_facecolor(COLOR_SCHEMES['hub']['background'])
    nx.draw_networkx_nodes(G_hub, pos_hub, ax=axes[3],
                         node_color=node_colors,
                         node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G_hub, pos_hub, ax=axes[3],
                         edge_color=COLOR_SCHEMES['hub']['edges'],
                         width=0.8, alpha=0.6)
    axes[3].set_title("Hub Structure\n(Barabási–Albert)", fontsize=14)
    axes[3].axis('off')
    
    # 5. Spatial Network
    # Generate brain-like positions
    positions = []
    while len(positions) < 50:
        r = np.random.uniform(0, 1) ** 0.5
        theta = np.random.uniform(0, 2 * np.pi)
        
        x = r * np.cos(theta) * 1.5
        y = r * np.sin(theta)
        
        positions.append((x, y))
    
    G_spatial = nx.random_geometric_graph(50, 0.5, pos=dict(enumerate(positions)), seed=42)
    pos_spatial = {i: positions[i] for i in range(50)}
    
    axes[4].set_facecolor(COLOR_SCHEMES['spatial']['background'])
    
    # Create Voronoi tesselation for background
    vor = Voronoi(positions)
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[j] for j in region]
            if len(polygon) > 2:
                axes[4].fill(*zip(*polygon), alpha=0.15, color=COLOR_SCHEMES['spatial']['regions'])
    
    nx.draw_networkx_nodes(G_spatial, pos_spatial, ax=axes[4],
                         node_color=COLOR_SCHEMES['spatial']['nodes'],
                         node_size=60, alpha=0.9)
    nx.draw_networkx_edges(G_spatial, pos_spatial, ax=axes[4],
                         edge_color=COLOR_SCHEMES['spatial']['edges'],
                         width=0.8, alpha=0.7)
    axes[4].set_title("Spatially Embedded\n(Spatial Model)", fontsize=14)
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_network_types.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {os.path.join(output_dir, 'all_network_types.png')}")
    return

#########################
# Execute script
#########################

if __name__ == "__main__":
    print("Generating network community visualizations...")
    
    # Generate static comparison of all network types
    print("Creating combined visualization of all network types...")
    create_combined_visualization()
    
    # Generate animations for each network type
    generate_all_animations()
    
    print("All visualizations completed successfully!")
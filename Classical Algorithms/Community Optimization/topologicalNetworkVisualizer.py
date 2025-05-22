#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topologicalNetworkVisualizer.py

This script creates animations showing topological aspects of community networks
that are directly related to Topological Consensus Networks. It visualizes concepts
like simplicial complexes, persistent homology, and consensus formation across
different network topologies.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from itertools import combinations
from matplotlib.animation import FuncAnimation, PillowWriter

# Create output directory directly in the Community Optimization folder
output_dir = os.path.dirname(os.path.abspath(__file__))

# Set random seed for reproducibility
np.random.seed(42)

# Color schemes for visualizations
COLORS = {
    'node': '#1f77b4',
    'edge': '#7f7f7f',
    'simplex': '#ff7f0e',
    'highlight': '#ff5555',
    'validator': '#2ca02c',
    'neutral': '#1f77b4',
    'consensus': '#d62728',
    'background': '#f5f5f5',
    'trust_high': '#2ca02c',
    'trust_medium': '#ffbf00',
    'trust_low': '#d62728'
}

#########################
# Topological Community Networks
#########################

def create_topological_consensus_animation(n_frames=40, n_nodes=25):
    """Create animation showing consensus formation in a network with topological features"""
    
    # Create a small-world network for the base structure (good for consensus)
    G = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=42)
    
    # Get positions using a spring layout which positions nodes nicely
    pos = nx.spring_layout(G, seed=42)
    
    # Identify key nodes - use betweenness centrality to find important nodes
    centrality = nx.betweenness_centrality(G)
    validator_nodes = [n for n, c in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    # For better presentation, remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Initialize node consensus states (0 = undecided, 1 = consensus)
    consensus_state = {n: 0 for n in G.nodes()}
    # Initialize validators to have consensus
    for n in validator_nodes:
        consensus_state[n] = 1
        
    # Define node spread - validators propagate their consensus to neighbors
    def spread_consensus(states, network, validators):
        new_states = states.copy()
        
        # Keep track of which nodes get influenced in this round
        newly_influenced = set()
        
        # Loop through all nodes
        for node in network.nodes():
            # Skip nodes that already have consensus
            if states[node] == 1:
                continue
                
            # Check neighbors for influence
            neighbors = list(network.neighbors(node))
            consensus_neighbors = sum(1 for n in neighbors if states[n] == 1)
            
            # Node is influenced if more than 50% of neighbors have consensus
            if consensus_neighbors > len(neighbors) / 2:
                new_states[node] = 1
                newly_influenced.add(node)
                
        return new_states, newly_influenced
    
    # Pre-compute states for each frame
    states_by_frame = []
    highlighted_nodes_by_frame = []
    simplices_by_frame = []
    
    # Initial state
    states_by_frame.append(consensus_state.copy())
    highlighted_nodes_by_frame.append(set(validator_nodes))  # Highlight validators in first frame
    simplices_by_frame.append([])  # No simplices in first frame
    
    # Compute consensus spread and simplices over time
    for frame in range(1, n_frames):
        # Spread consensus
        new_states, newly_influenced = spread_consensus(states_by_frame[-1], G, validator_nodes)
        states_by_frame.append(new_states)
        highlighted_nodes_by_frame.append(newly_influenced)
        
        # Find 2-simplices (triangles) among nodes with consensus
        consensus_nodes = [n for n, state in new_states.items() if state == 1]
        triangles = []
        
        # Look for triangles in the network
        for a, b, c in combinations(consensus_nodes, 3):
            if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(a, c):
                triangles.append((a, b, c))
                
        simplices_by_frame.append(triangles)
    
    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        current_states = states_by_frame[frame]
        highlighted = highlighted_nodes_by_frame[frame]
        
        # Draw simplices (triangles) first so they're behind nodes
        for triangle in simplices_by_frame[frame]:
            points = [pos[n] for n in triangle]
            polygon = Polygon(points, closed=True, alpha=0.3, facecolor=COLORS['simplex'], edgecolor=None)
            ax.add_patch(polygon)
        
        # Draw edges
        for u, v in G.edges():
            # Determine edge color based on consensus state
            if current_states[u] == 1 and current_states[v] == 1:
                edge_color = COLORS['consensus']
                alpha = 0.8
                linewidth = 1.5
            else:
                edge_color = COLORS['edge']
                alpha = 0.5
                linewidth = 1.0
                
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   color=edge_color, alpha=alpha, linewidth=linewidth)
        
        # Draw nodes with different colors based on state
        for node in G.nodes():
            if node in validator_nodes:
                node_color = COLORS['validator']
                node_size = 300
            elif current_states[node] == 1:
                node_color = COLORS['consensus']
                node_size = 250
            else:
                node_color = COLORS['neutral']
                node_size = 200
                
            # Add highlight for newly influenced nodes
            if node in highlighted:
                ax.scatter(pos[node][0], pos[node][1], s=node_size+100, color=COLORS['highlight'], alpha=0.3)
                
            ax.scatter(pos[node][0], pos[node][1], s=node_size, color=node_color, edgecolors='black', linewidth=1.5)
        
        # Add title showing consensus progress
        consensus_count = sum(1 for state in current_states.values() if state == 1)
        ax.set_title(f"Topological Consensus Formation\nConsensus: {consensus_count}/{n_nodes} nodes, Frame: {frame+1}/{n_frames}", 
                    fontsize=14, pad=20)
        
        return ax
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    # Save animation
    output_path = os.path.join(output_dir, 'topological_consensus_evolution.gif')
    print(f"Creating topological consensus animation...")
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return output_path

#########################
# Trust-Based Simplicial Complex
#########################

def create_trust_simplicial_animation(n_frames=40, n_nodes=30):
    """Create animation showing trust-based simplicial complex formation"""
    
    # Generate a network with trust values on edges
    G = nx.watts_strogatz_graph(n_nodes, 6, 0.2, seed=42)
    
    # Generate trust values for edges (random for this example)
    np.random.seed(42)
    nx.set_edge_attributes(G, {e: {'trust': np.random.uniform(0.1, 1.0)} for e in G.edges()})
    
    # Get positions using layout that spreads nodes nicely
    pos = nx.spring_layout(G, seed=42)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    # For better presentation, remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # For each frame, calculate which simplices exist based on trust threshold
    def get_simplices(graph, trust_threshold):
        # Get edges above threshold
        high_trust_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['trust'] >= trust_threshold]
        
        # Create subgraph with only high-trust edges
        trust_graph = nx.Graph()
        trust_graph.add_nodes_from(graph.nodes())
        trust_graph.add_edges_from(high_trust_edges)
        
        # Find 1-simplices (edges)
        edges = list(trust_graph.edges())
        
        # Find 2-simplices (triangles)
        triangles = []
        for a, b, c in combinations(graph.nodes(), 3):
            if (trust_graph.has_edge(a, b) and 
                trust_graph.has_edge(b, c) and 
                trust_graph.has_edge(a, c)):
                triangles.append((a, b, c))
        
        return edges, triangles
    
    # Pre-compute simplices for each frame
    trust_thresholds = np.linspace(0.9, 0.2, n_frames)  # Decreasing threshold over time
    edges_by_frame = []
    triangles_by_frame = []
    
    for threshold in trust_thresholds:
        edges, triangles = get_simplices(G, threshold)
        edges_by_frame.append(edges)
        triangles_by_frame.append(triangles)
    
    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        current_threshold = trust_thresholds[frame]
        edges = edges_by_frame[frame]
        triangles = triangles_by_frame[frame]
        
        # Draw all edges with transparency according to trust
        for u, v, data in G.edges(data=True):
            trust = data['trust']
            if (u, v) in edges or (v, u) in edges:
                # High trust edge
                if trust > 0.7:
                    edge_color = COLORS['trust_high']
                    linewidth = 2.0
                elif trust > 0.4:
                    edge_color = COLORS['trust_medium']
                    linewidth = 1.5
                else:
                    edge_color = COLORS['trust_low']
                    linewidth = 1.0
                alpha = 0.9
            else:
                # Low trust edge
                edge_color = COLORS['edge']
                alpha = 0.1
                linewidth = 0.5
            
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   color=edge_color, alpha=alpha, linewidth=linewidth)
        
        # Draw triangles (2-simplices)
        for triangle in triangles:
            points = [pos[n] for n in triangle]
            polygon = Polygon(points, closed=True, alpha=0.3, facecolor=COLORS['simplex'], edgecolor=None)
            ax.add_patch(polygon)
          # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=COLORS['node'], node_size=200, 
                              edgecolors='black', linewidths=1.5)
        
        # Add title showing trust threshold
        ax.set_title(f"Trust-Based Simplicial Complex\nTrust Threshold: {current_threshold:.2f}, Frame: {frame+1}/{n_frames}", 
                    fontsize=14, pad=20)
        
        return ax
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    # Save animation
    output_path = os.path.join(output_dir, 'trust_simplicial_complex.gif')
    print(f"Creating trust-based simplicial complex animation...")
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return output_path

#########################
# Community Detection via Filtration
#########################

def create_community_filtration_animation(n_frames=40, n_nodes=50):
    """Create animation showing community detection through persistent homology filtration"""
    
    # Create a network with clear community structure
    communities = 3
    nodes_per_community = n_nodes // communities
    
    # Create network using stochastic block model
    sizes = [nodes_per_community] * communities
    
    # More connections within communities, fewer between
    p_intra = 0.7  # Probability of connection within community
    p_inter = 0.05  # Probability of connection between communities
    
    # Create probability matrix
    p = np.zeros((communities, communities))
    for i in range(communities):
        for j in range(communities):
            if i == j:  # Same community
                p[i][j] = p_intra
            else:  # Different communities
                p[i][j] = p_inter
    
    # Generate the graph
    G = nx.stochastic_block_model(sizes, p, seed=42)
    
    # Generate positions with community structure
    pos = {}
    for i in range(communities):
        # Position communities in a circle
        community_angle = 2 * np.pi * i / communities
        community_radius = 5
        community_x = community_radius * np.cos(community_angle)
        community_y = community_radius * np.sin(community_angle)
        
        # Position nodes within communities in a smaller circle
        for j in range(nodes_per_community):
            node_idx = i * nodes_per_community + j
            angle = 2 * np.pi * j / nodes_per_community
            node_radius = 2
            pos[node_idx] = (
                community_x + node_radius * np.cos(angle),
                community_y + node_radius * np.sin(angle)
            )
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    # For better presentation, remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Compute edge distances based on positions
    edge_distance = {}
    for u, v in G.edges():
        dist = np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
        edge_distance[(u, v)] = dist
        edge_distance[(v, u)] = dist
    
    # Pre-compute filtration for each frame
    # Start with high distance threshold to show only intra-community edges,
    # then gradually lower to reveal inter-community edges
    distance_thresholds = np.linspace(3.0, 10.0, n_frames)
    
    edges_by_frame = []
    community_edges_by_frame = []
    
    for threshold in distance_thresholds:
        # Edges with distance less than threshold
        edges = [(u, v) for (u, v) in G.edges() if edge_distance[(u, v)] <= threshold]
        
        # Separate intra-community and inter-community edges
        intra_edges = []
        inter_edges = []
        
        for u, v in edges:
            u_comm = u // nodes_per_community
            v_comm = v // nodes_per_community
            
            if u_comm == v_comm:
                intra_edges.append((u, v))
            else:
                inter_edges.append((u, v))
        
        edges_by_frame.append(edges)
        community_edges_by_frame.append((intra_edges, inter_edges))
    
    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        current_threshold = distance_thresholds[frame]
        intra_edges, inter_edges = community_edges_by_frame[frame]
        
        # Draw intra-community edges
        for u, v in intra_edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   color=COLORS['trust_high'], alpha=0.7, linewidth=1.0)
        
        # Draw inter-community edges
        for u, v in inter_edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   color=COLORS['highlight'], alpha=0.5, linewidth=1.0)
        
        # Draw nodes with community colors
        for i in range(communities):
            start_idx = i * nodes_per_community
            end_idx = start_idx + nodes_per_community
            
            # Get different colors for different communities
            if i == 0:
                color = COLORS['validator']
            elif i == 1:
                color = COLORS['consensus']
            else:
                color = COLORS['neutral']
            
            # Draw nodes for this community
            nodes = list(range(start_idx, end_idx))
            xs = [pos[n][0] for n in nodes]
            ys = [pos[n][1] for n in nodes]
            ax.scatter(xs, ys, color=color, s=150, edgecolors='black', linewidth=1.0)
        
        # Add title
        intra_count = len(intra_edges)
        inter_count = len(inter_edges)
        total_edges = intra_count + inter_count
        
        ax.set_title(f"Community Detection via Distance Filtration\n" +
                    f"Distance Threshold: {current_threshold:.1f}, " +
                    f"Intra: {intra_count}, Inter: {inter_count}", 
                    fontsize=14, pad=20)
        
        return ax
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    # Save animation
    output_path = os.path.join(output_dir, 'community_filtration.gif')
    print(f"Creating community filtration animation...")
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return output_path

#########################
# Main function
#########################

def main():
    print("Generating topological network visualizations for Topological Consensus Networks...")
    print("Current working directory:", os.getcwd())
    print("Output directory:", output_dir)
    
    try:
        # Create all animations
        print("Creating topological consensus animation...")
        consensus_animation = create_topological_consensus_animation(n_frames=30)
        
        print("Creating trust-based simplicial complex animation...")
        trust_animation = create_trust_simplicial_animation(n_frames=30)
        
        print("Creating community filtration animation...")
        community_animation = create_community_filtration_animation(n_frames=30)
        
        print("All topological network visualizations completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

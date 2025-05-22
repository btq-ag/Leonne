#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkVisualizer.py

This script creates visualizations that are more relevant to topological consensus networks,
including community structures, network evolution, and statistical properties.
It integrates concepts from network theory, topological data analysis, and consensus algorithms.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import PillowWriter
from itertools import combinations
import os
from tqdm import tqdm
from scipy.spatial import Voronoi

# Output directory set to current directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Set random seed for reproducibility
np.random.seed(42)

# Define colors for different consensus states and node types
COLORS = {
    'consensus': '#4CAF50',  # Green
    'dissent': '#F44336',    # Red
    'neutral': '#2196F3',    # Blue
    'validator': '#FFC107',  # Yellow/Gold for validators
    'background': '#f5f5f5',
    'edge': '#555555',
    'highlight': '#9C27B0'   # Purple for highlighting
}

def create_topological_consensus_animation(n_nodes=30, n_frames=40, network_type='small_world'):
    """
    Create an animation showing consensus formation in a network using topological properties.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    n_frames : int
        Number of animation frames
    network_type : str
        Type of network ('random', 'small_world', 'scale_free', 'community')
    """
    # Create the base network based on type
    if network_type == 'random':
        G = nx.erdos_renyi_graph(n_nodes, 0.15, seed=42)
        title = "Random Network Consensus Evolution"
    elif network_type == 'small_world':
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.2, seed=42)
        title = "Small-World Network Consensus Evolution"
    elif network_type == 'scale_free':
        G = nx.barabasi_albert_graph(n_nodes, 2, seed=42)
        title = "Scale-Free Network Consensus Evolution"
    elif network_type == 'community':
        # Create a community structure with 3 communities
        communities = 3
        nodes_per_community = n_nodes // communities
        sizes = [nodes_per_community] * communities
        p = np.zeros((communities, communities))
        for i in range(communities):
            for j in range(communities):
                if i == j:  # Intra-community
                    p[i][j] = 0.7
                else:  # Inter-community
                    p[i][j] = 0.1
        G = nx.stochastic_block_model(sizes, p, seed=42)
        title = "Community Structure Consensus Evolution"
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Position nodes using spring layout for a nice visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Select validators (nodes with highest centrality)
    betweenness = nx.betweenness_centrality(G)
    validators = sorted(betweenness, key=betweenness.get, reverse=True)[:3]
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    # Initial state: all nodes are neutral except validators which have consensus
    node_states = {}
    for node in G.nodes():
        if node in validators:
            node_states[node] = 'consensus'
        else:
            node_states[node] = 'neutral'
    
    # Pre-compute consensus evolution (consensus spreads from validators)
    consensus_evolution = []
    consensus_evolution.append(node_states.copy())
    
    current_states = node_states.copy()
    for frame in range(1, n_frames):
        new_states = current_states.copy()
        
        # For each node, check its neighbors' states
        for node in G.nodes():
            if current_states[node] != 'consensus':
                neighbors = list(G.neighbors(node))
                consensus_neighbors = sum(1 for n in neighbors if current_states[n] == 'consensus')
                
                # If more than half of neighbors have consensus, adopt consensus
                if consensus_neighbors > len(neighbors) / 2:
                    new_states[node] = 'consensus'
                # Occasionally introduce some dissent
                elif np.random.random() < 0.05 and frame < n_frames / 2:
                    new_states[node] = 'dissent'
        
        current_states = new_states
        consensus_evolution.append(current_states.copy())
    
    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        
        # Get node states for this frame
        frame_states = consensus_evolution[frame]
        
        # Create node color mapping
        node_colors = [COLORS[frame_states[node]] for node in G.nodes()]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.7, width=1.5, edge_color=COLORS['edge'])
        
        # Draw nodes with different colors based on state
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=[n for n in G.nodes() if n not in validators],
                             node_color=[COLORS[frame_states[n]] for n in G.nodes() if n not in validators],
                             node_size=300, alpha=0.9)
        
        # Draw validator nodes
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=validators,
                             node_color=[COLORS[frame_states[n]] for n in validators],
                             node_size=500, alpha=0.9, edgecolors='black', linewidths=2)
        
        # Add title with consensus percentage
        consensus_pct = sum(1 for v in frame_states.values() if v == 'consensus') / n_nodes * 100
        ax.set_title(f"{title}\nConsensus: {consensus_pct:.1f}%", fontsize=16)
        
        # Remove axis
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        return ax
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    
    # Save animation
    filename = f"consensus_evolution_{network_type}.gif"
    filepath = os.path.join(output_dir, filename)
    anim.save(filepath, writer='pillow', fps=6, dpi=120)
    
    print(f"Saved consensus evolution animation to {filepath}")
    return anim

def create_topological_simplices_animation(n_nodes=50, n_frames=40):
    """
    Create a 3D animation showing the evolution of simplices in topological data analysis.
    This is more directly related to topological consensus networks.
    """
    # Generate two communities with different characteristics
    np.random.seed(42)
    
    # First community (denser)
    A_nodes = 25
    A = nx.watts_strogatz_graph(A_nodes, 4, 0.2, seed=42)
    
    # Second community (sparser)
    B_nodes = 25
    B = nx.erdos_renyi_graph(B_nodes, 0.1, seed=43)
    
    # Position nodes in 2D space
    scale = 8
    offset_x = scale + 2
    pos2d = {}
    
    # Position first community in a more structured layout
    for i, node in enumerate(A.nodes()):
        angle = 2 * np.pi * i / A_nodes
        r = 0.8 + 0.2 * np.random.random()
        pos2d[node] = ((r * np.cos(angle) - 0.5) * scale, (r * np.sin(angle) - 0.5) * scale)
    
    # Position second community with more randomness
    for j, node in enumerate(B.nodes()):
        angle = 2 * np.pi * j / B_nodes
        r = 0.8 + 0.4 * np.random.random()
        pos2d[node + A_nodes] = ((r * np.cos(angle) - 0.5) * scale + offset_x, (r * np.sin(angle) - 0.5) * scale)
    
    # Compute inter-community edges (representing consensus bridges)
    cross_pairs = []
    for u in A.nodes():
        x1, y1 = pos2d[u]
        for v in B.nodes():
            vid = v + A_nodes
            x2, y2 = pos2d[vid]
            dist = np.hypot(x2 - x1, y2 - y1)
            cross_pairs.append((u, vid, dist))
    cross_pairs.sort(key=lambda x: x[2])
    
    # Create the merged graph for final state
    final_graph = nx.disjoint_union(A, B)
    
    # Add edges between communities (top 20% closest pairs)
    n_cross_edges = int(0.1 * len(cross_pairs))
    for u, vid, _ in cross_pairs[:n_cross_edges]:
        final_graph.add_edge(u, vid)
    
    # Setup figure for 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    
    # Base height for visualization
    zs = 0
    
    # Function to grow graph over time
    def grow(G, t, T):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        edges = list(G.edges())
        count = int(len(edges) * (t + 1) / T)
        H.add_edges_from(edges[:count])
        return H
    
    # Function to merge communities progressively
    def merge_step(t):
        G = nx.disjoint_union(A, B)
        # Calculate how many cross edges to add at this step
        count = int(n_cross_edges * t / (n_frames // 3))
        for u, vid, _ in cross_pairs[:count]:
            G.add_edge(u, vid)
        return G
    
    # Function to draw 2-simplices (triangles) representing higher-order relationships
    def draw_simplices(G):
        # Find all triangles (3-cliques) in the graph
        triangles = []
        for trio in combinations(G.nodes(), 3):
            if (G.has_edge(trio[0], trio[1]) and 
                G.has_edge(trio[1], trio[2]) and 
                G.has_edge(trio[0], trio[2])):
                # Create 3D coordinates for the triangle
                pts = [(*pos2d[n], zs) for n in trio]
                triangles.append(pts)
        
        # Draw triangles if any exist
        if triangles:
            # Color triangles based on whether they span communities or not
            colors = []
            for trio in triangles:
                nodes = [i for i, _ in enumerate(trio)]
                # Check if this triangle spans both communities
                has_A = any(n < A_nodes for n in nodes)
                has_B = any(n >= A_nodes for n in nodes)
                if has_A and has_B:
                    colors.append(COLORS['highlight'])  # Cross-community triangles
                else:
                    colors.append('lightblue')  # Within-community triangles
            
            poly = Poly3DCollection(triangles, facecolors=colors, alpha=0.6, linewidths=0.5, edgecolors='gray')
            ax.add_collection3d(poly)
    
    # Function to draw the network
    def draw_graph(G):
        # Draw simplices first (as background)
        draw_simplices(G)
        
        # Draw edges
        for u, v in G.edges():
            x1, y1 = pos2d[u]
            x2, y2 = pos2d[v]
            
            # Check if this is a cross-community edge
            if (u < A_nodes and v >= A_nodes) or (u >= A_nodes and v < A_nodes):
                color = COLORS['highlight']  # Highlight cross edges
                linewidth = 2.0
            else:
                color = COLORS['edge']
                linewidth = 1.0
                
            ax.plot([x1, x2], [y1, y2], [zs, zs], color=color, linewidth=linewidth)
          # Draw nodes - different colors for different communities
        # Get nodes for each community
        nodes_A = [n for n in G.nodes() if n < A_nodes]
        nodes_B = [n for n in G.nodes() if n >= A_nodes]
        
        # Draw nodes from community A if there are any
        if nodes_A:
            xs_A, ys_A = zip(*(pos2d[n] for n in nodes_A))
            ax.scatter(xs_A, ys_A, zs, c=COLORS['neutral'], s=80, edgecolors='black', linewidth=1)
        
        # Draw nodes from community B if there are any
        if nodes_B:
            xs_B, ys_B = zip(*(pos2d[n] for n in nodes_B))
            ax.scatter(xs_B, ys_B, zs, c=COLORS['validator'], s=80, edgecolors='black', linewidth=1)
    
    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_facecolor('white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        
        # First phase: grow community A
        if frame < n_frames // 3:
            H = grow(A, frame, n_frames // 3)
            H = nx.disjoint_union(H, nx.Graph())  # Empty B community
            step_desc = f"Building Community A: {frame+1}/{n_frames//3}"
            
        # Second phase: grow community B
        elif frame < 2 * (n_frames // 3):
            H_A = A.copy()  # A is fully grown
            k = frame - n_frames // 3
            H_B = grow(B, k, n_frames // 3)
            
            # Combine the communities
            H = nx.disjoint_union(H_A, H_B)
            step_desc = f"Building Community B: {k+1}/{n_frames//3}"
            
        # Third phase: add cross-community edges
        else:
            t = frame - 2 * (n_frames // 3)
            H = merge_step(t)
            step_desc = f"Building Cross-Community Links: {t+1}/{n_frames//3}"
        
        # Draw the current state of the graph
        draw_graph(H)
        
        # Count higher-order structures (triangles)
        triangle_count = 0
        for trio in combinations(H.nodes(), 3):
            if (H.has_edge(trio[0], trio[1]) and 
                H.has_edge(trio[1], trio[2]) and 
                H.has_edge(trio[0], trio[2])):
                triangle_count += 1
        
        # Add annotations
        ax.set_title(f"Topological Consensus Network Evolution\n{step_desc}", fontsize=14, color='black')
        
        # Add information about 2-simplices (triangles)
        ax.text2D(0.05, 0.95, f"2-Simplices (Triangles): {triangle_count}", 
                 transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
        
        # Set view angle and limits
        ax.view_init(elev=30, azim=frame % 360)  # Rotate view
        ax.set_xlim(-scale, scale + offset_x + 2)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200)
    
    # Save animation
    filename = "network_evolution_3d_simplices_v5.gif"
    output_path = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=8)
    anim.save(output_path, writer=writer)
    
    print(f"Saved 3D simplices animation to {output_path}")
    return anim

def create_consensus_landscape_animation(n_points=30, n_frames=30):
    """
    Create an animation showing a network complex with consensus landscape.
    This combines topological data analysis with consensus dynamics.
    """
    # Generate random point cloud
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * 2 - 1
    
    # Compute pairwise distances
    edges = []
    for i, j in combinations(range(n_points), 2):
        dist = np.linalg.norm(points[i] - points[j])
        edges.append((i, j, dist))
    edges.sort(key=lambda x: x[2])
    
    # Setup Union-Find for tracking connected components
    parent = list(range(n_points))
    rank = [0] * n_points
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True
    
    # Initialize consensus states with some seed nodes
    consensus_states = np.zeros(n_points)
    
    # Select 3 random seed nodes to have initial consensus
    seed_nodes = np.random.choice(n_points, 3, replace=False)
    consensus_states[seed_nodes] = 1.0
    
    # Persistence bars (birth-death pairs)
    bars = []
    for i, j, dist in edges:
        if union(i, j):
            # When components merge, record bars
            bars.append((0.0, dist))
    
    # Radii and persistence landscape
    radii = np.linspace(0, 1.0, n_frames)
    f_vals = np.zeros((len(bars), len(radii)))
    for idx, (birth, death) in enumerate(bars):
        for j, t in enumerate(radii):
            f_vals[idx, j] = max(0, min(t, death - t))
    
    # Compute first persistence landscape
    lam1 = f_vals.max(axis=0)
    
    # Setup figure with three subplots
    fig, (ax_pts, ax_bar, ax_land) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Prepare animation writer
    filename = "network_complex_with_landscape_v2.gif"
    output_path = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=10)
    
    # Update function for animation
    def update(frame):
        radius = radii[frame]
        ax_pts.clear()
        ax_bar.clear()
        ax_land.clear()
        
        # Update consensus propagation
        if frame > 0:
            # At each step, consensus spreads from node to node
            new_consensus = consensus_states.copy()
            for i, j, dist in edges:
                if dist <= radius and consensus_states[i] > 0:
                    # Transfer some consensus to neighbor
                    new_consensus[j] = max(new_consensus[j], consensus_states[i] * (1.0 - dist/1.5))
                if dist <= radius and consensus_states[j] > 0:
                    new_consensus[i] = max(new_consensus[i], consensus_states[j] * (1.0 - dist/1.5))
            consensus_states[:] = new_consensus
        
        # Draw network complex with consensus state reflected in node colors
        node_colors = []
        for i in range(n_points):
            # Gradient from blue (0) to green (1)
            if consensus_states[i] <= 0.2:
                color = COLORS['dissent']  # No consensus
            elif consensus_states[i] >= 0.8:
                color = COLORS['consensus']  # Full consensus
            else:
                color = COLORS['neutral']  # Partial consensus
            node_colors.append(color)
        
        # Draw nodes
        ax_pts.scatter(points[:, 0], points[:, 1], c=node_colors, s=100, edgecolors='black')
        
        # Draw 2-simplices (triangles)
        triangles = []
        for u, v, w in combinations(range(n_points), 3):
            if (np.linalg.norm(points[u]-points[v]) <= radius and
                np.linalg.norm(points[v]-points[w]) <= radius and
                np.linalg.norm(points[u]-points[w]) <= radius):
                triangles.append([points[u], points[v], points[w]])
        
        # Draw triangles
        if triangles:
            coll = PolyCollection(triangles, facecolors='lightgray', alpha=0.3, edgecolors='none')
            ax_pts.add_collection(coll)
        
        # Draw edges
        for i, j, dist in edges:
            if dist <= radius:
                ax_pts.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                         color='gray', alpha=0.7, linewidth=1.0)
        
        # Title and formatting for network plot
        ax_pts.set_title(f"Consensus Network Complex (r = {radius:.2f})")
        ax_pts.set_aspect('equal')
        ax_pts.set_xlim(-1.2, 1.2)
        ax_pts.set_ylim(-1.2, 1.2)
        ax_pts.set_axis_off()
        
        # Compute consensus percentage
        consensus_pct = np.sum(consensus_states >= 0.8) / n_points * 100
        ax_pts.text(-1.0, -1.0, f"Consensus: {consensus_pct:.1f}%", fontsize=12)
        
        # Draw persistence barcode
        for idx, (birth, death) in enumerate(bars):
            if radius >= birth:
                end = min(radius, death)
                ax_bar.hlines(idx, birth, end, linewidth=4, color='purple')
        
        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_ylim(-1, len(bars))
        ax_bar.set_title("H0 Persistence Barcode")
        ax_bar.set_xlabel("Filtration Parameter (ε)")
        ax_bar.set_yticks([])
        
        # Draw persistence landscape
        ax_land.plot(radii[:frame+1], lam1[:frame+1], color='darkblue', linewidth=2)
        ax_land.set_xlim(0, 1.0)
        ax_land.set_ylim(0, lam1.max() * 1.05)
        ax_land.set_title("Persistence Landscape λ1")
        ax_land.set_xlabel("Filtration Parameter (ε)")
        ax_land.set_ylabel("λ1 (First Landscape Function)")
        
        # Add connectivity information
        components = len(set(find(i) for i in range(n_points)))
        ax_bar.text(0.5, -0.9, f"Connected Components: {components}", fontsize=10, ha='center')
    
    # Create and save animation
    with writer.saving(fig, output_path, dpi=100):
        for frame in range(n_frames):
            update(frame)
            writer.grab_frame()
    
    # Save a frame preview
    frame_path = os.path.join(output_dir, "network_complex_with_landscape_v2_frame.png") 
    fig.savefig(frame_path)
    
    print(f"Saved network complex with landscape animation to {output_path}")
    print(f"Saved frame preview to {frame_path}")
    
    return fig

# Main function to generate all visualizations with different network configurations
def main():
    print("Generating network visualizations for Topological Consensus Networks...")
    
    try:
        print("\nCreating consensus landscape animation...")
        create_consensus_landscape_animation(n_points=30, n_frames=30)
        
        # Generate consensus evolution animations for a single network type
        print("\nCreating consensus evolution animation for small_world network...")
        create_topological_consensus_animation(n_nodes=30, n_frames=30, network_type='small_world')
        
        print("\nAll visualizations completed!")
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

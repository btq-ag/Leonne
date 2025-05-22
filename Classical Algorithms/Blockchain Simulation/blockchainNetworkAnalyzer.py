#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blockchainNetworkAnalyzer.py

This script enhances the blockchain visualization by incorporating network analysis
concepts from other algorithms in the project. It creates animated visualizations
of different network types on blockchain structures, and analyzes their topological
and trust properties across the blockchain.

Features:
1. Multiple network models for blockchain blocks (random, small-world, scale-free, 
   community structure, spatial)
2. Trust metrics between blocks in the chain
3. Network community analysis across the blockchain
4. Different visualization styles for various network properties
5. Multiple configurations for different types of blockchains

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from itertools import combinations
import random

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

#########################
# Network Generation Functions
#########################

def create_random_network(nodes, p=0.3):
    """Create a random (Erdős–Rényi) network"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, j in combinations(nodes, 2):
        if np.random.random() < p:
            G.add_edge(i, j)
    return G

def create_small_world_network(nodes, k=4, p=0.1):
    """Create a small-world (Watts-Strogatz) network"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    
    # Connect each node to k nearest neighbors
    k = min(k, n-1)
    for i, node in enumerate(nodes):
        for j in range(1, k//2 + 1):
            G.add_edge(node, nodes[(i+j) % n])
            G.add_edge(node, nodes[(i-j) % n])
    
    # Rewire edges with probability p
    for i, node in enumerate(nodes):
        for j in range(1, k//2 + 1):
            if np.random.random() < p:
                # Remove one connection
                neighbor = nodes[(i+j) % n]
                if G.has_edge(node, neighbor):
                    G.remove_edge(node, neighbor)
                    # Add a random different connection
                    available = [n for n in nodes if n != node and n != neighbor and not G.has_edge(node, n)]
                    if available:
                        new_neighbor = np.random.choice(available)
                        G.add_edge(node, new_neighbor)
    
    return G

def create_scale_free_network(nodes, m=2):
    """Create a scale-free (Barabási-Albert) network"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    
    if n <= m:
        # For small networks, connect everything
        for i, j in combinations(nodes, 2):
            G.add_edge(i, j)
        return G
    
    # Create initial fully connected network with m nodes
    for i, j in combinations(nodes[:m], 2):
        G.add_edge(i, j)
    
    # Add remaining nodes with preferential attachment
    for i in range(m, n):
        node = nodes[i]
        # Calculate probabilities based on degree
        degrees = [G.degree(nodes[j]) for j in range(i)]
        total_degree = sum(degrees) if sum(degrees) > 0 else 1
        probs = [d/total_degree for d in degrees]
        
        # Connect to m existing nodes
        targets = np.random.choice(range(i), size=min(m, i), replace=False, p=probs)
        for target in targets:
            G.add_edge(node, nodes[target])
    
    return G

def create_community_network(nodes, n_communities=2, p_intra=0.7, p_inter=0.05):
    """Create a network with community structure"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    
    # Create approximately equal-sized communities
    communities = [[] for _ in range(n_communities)]
    for i, node in enumerate(nodes):
        communities[i % n_communities].append(node)
    
    # Add intra-community edges with high probability
    for community in communities:
        for i, j in combinations(community, 2):
            if np.random.random() < p_intra:
                G.add_edge(i, j)
    
    # Add inter-community edges with low probability
    for c1_idx in range(n_communities):
        for c2_idx in range(c1_idx+1, n_communities):
            for i in communities[c1_idx]:
                for j in communities[c2_idx]:
                    if np.random.random() < p_inter:
                        G.add_edge(i, j)
    
    return G, communities

def create_spatial_network(nodes, node_positions, radius=0.3):
    """Create a spatially embedded network"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # Connect nodes based on physical proximity
    for i, j in combinations(nodes, 2):
        pos_i = node_positions[i]
        pos_j = node_positions[j]
        distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
        if distance < radius:
            G.add_edge(i, j)
    
    return G

#########################
# Trust Functions
#########################

def calculate_trust_matrix(G, nodes):
    """Calculate trust matrix based on network structure"""
    n = len(nodes)
    trust_matrix = np.zeros((n, n))
    
    # Use shortest path distance to calculate trust (closer = more trust)
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                trust_matrix[i, j] = 1.0  # Self-trust is maximum
            elif G.has_edge(node_i, node_j):
                trust_matrix[i, j] = 0.8  # Direct connections have high trust
            else:
                try:
                    # Calculate trust based on shortest path distance
                    path_length = nx.shortest_path_length(G, node_i, node_j)
                    trust_matrix[i, j] = max(0.1, 1.0 / (path_length + 1))
                except nx.NetworkXNoPath:
                    trust_matrix[i, j] = 0.0  # No path means no trust
    
    return trust_matrix

def calculate_inter_block_trust(block_trust_matrices, inter_block_edges):
    """Calculate trust between blocks based on connections"""
    n_blocks = len(block_trust_matrices)
    inter_block_trust = np.zeros((n_blocks, n_blocks))
    
    # Self-trust of blocks is high
    for i in range(n_blocks):
        inter_block_trust[i, i] = 1.0
    
    # Calculate trust between different blocks based on inter-block edges
    for i in range(n_blocks):
        for j in range(i+1, n_blocks):
            # Count edges between blocks i and j
            edges_between = sum(1 for u, v in inter_block_edges if 
                               (u // 20 == i and v // 20 == j) or 
                               (u // 20 == j and v // 20 == i))
            
            # Calculate trust based on number of connections
            max_possible = 20  # maximum reasonable number of inter-block edges
            trust_value = min(0.9, edges_between / max_possible)
            
            # Make trust symmetric
            inter_block_trust[i, j] = trust_value
            inter_block_trust[j, i] = trust_value
    
    return inter_block_trust

#########################
# Blockchain Network Visualization
#########################

def create_blockchain_network_visualization(n_blocks=5, n_nodes_per_block=20, n_frames=40, 
                                          network_type='mixed', filename='blockchain_network_analysis.gif'):
    """
    Create visualization of blockchain networks with different network types and analysis
    
    Args:
        n_blocks: Number of blocks in the blockchain
        n_nodes_per_block: Number of nodes per block
        n_frames: Number of animation frames
        network_type: Type of networks to create ('mixed', 'small_world', 'scale_free', 
                     'community', 'random', 'spatial')
        filename: Output filename for the animation
    """
    # Set up the figure with 3D projection
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Define the blockchain structure
    block_height = 0.5
    block_radius = 2.0
    spacing = 1.2 * block_height
    
    # Generate positions for nodes on each block
    block_nodes = {}
    node_positions_3d = {}
    block_graphs = {}
    block_trust_matrices = {}
    communities_by_block = {}
    
    for block_idx in range(n_blocks):
        # Define nodes for this block
        block_nodes[block_idx] = list(range(block_idx * n_nodes_per_block, 
                                          (block_idx + 1) * n_nodes_per_block))
        
        # Position nodes in a circle on the cylinder
        z_position = block_idx * (block_height + spacing)
        
        # Create 2D and 3D positions for each node
        positions_2d = {}
        for i, node_idx in enumerate(block_nodes[block_idx]):
            angle = 2 * np.pi * i / n_nodes_per_block
            x = block_radius * np.cos(angle)
            y = block_radius * np.sin(angle)
            z = z_position + block_height / 2
            
            positions_2d[node_idx] = (x, y)
            node_positions_3d[node_idx] = (x, y, z)
        
        # Determine network type for this block
        if network_type == 'mixed':
            current_type = block_idx % 5
        else:
            type_mapping = {
                'random': 0,
                'small_world': 1,
                'scale_free': 2,
                'community': 3,
                'spatial': 4
            }
            current_type = type_mapping.get(network_type, 0)
        
        # Create the appropriate network for this block
        if current_type == 0:  # Random network
            block_graphs[block_idx] = create_random_network(block_nodes[block_idx], p=0.3)
            communities_by_block[block_idx] = [block_nodes[block_idx]]  # One community
            
        elif current_type == 1:  # Small-world
            block_graphs[block_idx] = create_small_world_network(block_nodes[block_idx], k=4, p=0.1)
            communities_by_block[block_idx] = [block_nodes[block_idx]]  # One community
            
        elif current_type == 2:  # Scale-free
            block_graphs[block_idx] = create_scale_free_network(block_nodes[block_idx], m=2)
            communities_by_block[block_idx] = [block_nodes[block_idx]]  # One community
            
        elif current_type == 3:  # Community structure
            block_graphs[block_idx], communities = create_community_network(
                block_nodes[block_idx], n_communities=3, p_intra=0.7, p_inter=0.05)
            communities_by_block[block_idx] = communities
            
        else:  # Spatial network
            block_graphs[block_idx] = create_spatial_network(
                block_nodes[block_idx], positions_2d, radius=0.7)
            communities_by_block[block_idx] = [block_nodes[block_idx]]  # One community
        
        # Calculate trust matrix for this block
        block_trust_matrices[block_idx] = calculate_trust_matrix(
            block_graphs[block_idx], block_nodes[block_idx])
    
    # Create inter-block connections
    inter_block_edges = []
    for block_idx in range(n_blocks - 1):
        upper_nodes = block_nodes[block_idx]
        lower_nodes = block_nodes[block_idx + 1]
        
        # Number of inter-block connections depends on trust between blocks
        n_connections = np.random.randint(5, 15)
        
        # Connect some nodes between adjacent blocks
        for _ in range(n_connections):
            upper_node = np.random.choice(upper_nodes)
            lower_node = np.random.choice(lower_nodes)
            inter_block_edges.append((upper_node, lower_node))
    
    # Calculate trust between blocks
    inter_block_trust = calculate_inter_block_trust(block_trust_matrices, inter_block_edges)
    
    # Define colors for different network types
    network_colors = {
        0: '#ffb347',  # Random - orange
        1: '#87ceeb',  # Small-world - sky blue
        2: '#ff6961',  # Scale-free - salmon
        3: '#77dd77',  # Community - light green
        4: '#b19cd9'   # Spatial - light purple
    }
    
    # Create animation with additional pause frames
    pause_frames = 5
    total_frames = n_frames + pause_frames
    
    # Create colormap for trust values
    trust_cmap = plt.cm.YlOrRd
    
    def update(frame):
        ax.clear()
        ax.set_facecolor('black')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, n_blocks * (block_height + spacing))
        ax.set_axis_off()
        
        # For the pause frames, use the last frame's state
        data_frame = min(frame, n_frames - 1)
        
        # Determine how many blocks to show based on frame
        if data_frame < n_frames // 2:
            # First half of frames: progressively show more blocks
            blocks_to_show = max(1, int(data_frame / (n_frames // 2) * n_blocks))
            edges_progress = 1.0  # Show all edges for revealed blocks
        else:
            # Second half: all blocks visible, progressively show analysis
            blocks_to_show = n_blocks
            edges_progress = min(1.0, (data_frame - n_frames // 2) / (n_frames // 2 - 1))
        
        # Draw blocks as cylinders
        for block_idx in range(min(blocks_to_show, n_blocks)):
            z = block_idx * (block_height + spacing)
            
            # Determine network type for color
            if network_type == 'mixed':
                current_type = block_idx % 5
            else:
                type_mapping = {
                    'random': 0,
                    'small_world': 1,
                    'scale_free': 2,
                    'community': 3,
                    'spatial': 4
                }
                current_type = type_mapping.get(network_type, 0)
            
            cylinder_color = network_colors[current_type]
            
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 32)
            x = block_radius * np.cos(theta)
            y = block_radius * np.sin(theta)
            
            # Draw bottom circle
            ax.plot(x, y, [z] * len(theta), color='white', alpha=0.5, linewidth=1.5)
            # Draw top circle
            ax.plot(x, y, [z + block_height] * len(theta), color='white', alpha=0.5, linewidth=1.5)
            
            # Create the sides of the cylinder with network type color
            for i in range(len(theta)-1):
                ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], color='white', alpha=0.5, linewidth=1)
            
            # Fill cylinder with transparent color
            alpha = 0.2  # Transparency level
            ax.plot_surface(
                np.outer(x, np.ones(2)),
                np.outer(y, np.ones(2)),
                np.outer(np.ones(len(theta)), [z, z+block_height]),
                color=cylinder_color, alpha=alpha
            )
            
            # Add network type label
            type_names = ['Random', 'Small-World', 'Scale-Free', 'Community', 'Spatial']
            label_x = 0
            label_y = 0
            label_z = z + block_height + 0.2
            ax.text(label_x, label_y, label_z, type_names[current_type], 
                  color=cylinder_color, fontsize=10, ha='center', va='center')
            
            # Draw communities if applicable
            if current_type == 3 and edges_progress > 0.5:  # Community network
                communities = communities_by_block[block_idx]
                community_colors = ['#ff5733', '#33ff57', '#3357ff']
                
                for comm_idx, community in enumerate(communities):
                    if comm_idx < len(community_colors):
                        # Draw a colored transparent surface for each community
                        comm_nodes = [node for node in community]
                        comm_x = [node_positions_3d[node][0] for node in comm_nodes]
                        comm_y = [node_positions_3d[node][1] for node in comm_nodes]
                        
                        # Calculate centroid
                        centroid_x = np.mean(comm_x)
                        centroid_y = np.mean(comm_y)
                        
                        # Draw a radius from centroid
                        radius = 0.6
                        theta = np.linspace(0, 2*np.pi, 20)
                        circle_x = centroid_x + radius * np.cos(theta)
                        circle_y = centroid_y + radius * np.sin(theta)
                        circle_z = z + block_height/2
                        
                        ax.plot(circle_x, circle_y, [circle_z] * len(theta), 
                              color=community_colors[comm_idx], alpha=0.7, linewidth=2)
            
            # Draw nodes for this block
            for node_idx in block_nodes[block_idx]:
                node_pos = node_positions_3d[node_idx]
                
                # Node size based on centrality for scale-free networks
                if current_type == 2 and edges_progress > 0.5:  # Scale-free
                    degree = block_graphs[block_idx].degree(node_idx)
                    node_size = 30 + degree * 5
                    node_color = 'red' if degree > 3 else 'white'
                else:
                    node_size = 30
                    node_color = 'white'
                
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                         color=node_color, s=node_size, edgecolors='black', alpha=0.8)
            
            # Draw intra-block edges for this block with trust-based coloring
            if edges_progress > 0.2:
                for edge in block_graphs[block_idx].edges():
                    u, v = edge
                    pos_u = node_positions_3d[u]
                    pos_v = node_positions_3d[v]
                    
                    # For small-world networks, highlight long-range connections
                    if current_type == 1:
                        u_idx = block_nodes[block_idx].index(u)
                        v_idx = block_nodes[block_idx].index(v)
                        dist = min(abs(u_idx - v_idx), n_nodes_per_block - abs(u_idx - v_idx))
                        
                        if dist > 2:  # Long-range connection
                            edge_color = '#ff7f7f'  # Light red for long-range
                            alpha = 0.9
                            width = 2
                        else:  # Local connection
                            edge_color = 'white'
                            alpha = 0.6
                            width = 1
                    
                    # For community networks, color edges by community
                    elif current_type == 3 and edges_progress > 0.5:
                        # Find which communities u and v belong to
                        u_comm = v_comm = -1
                        for comm_idx, community in enumerate(communities_by_block[block_idx]):
                            if u in community:
                                u_comm = comm_idx
                            if v in community:
                                v_comm = comm_idx
                        
                        if u_comm == v_comm:  # Same community
                            edge_color = community_colors[u_comm % len(community_colors)]
                            alpha = 0.7
                            width = 1.5
                        else:  # Between communities
                            edge_color = 'white'
                            alpha = 0.3
                            width = 0.5
                    
                    # Otherwise color by trust
                    else:
                        u_local_idx = block_nodes[block_idx].index(u)
                        v_local_idx = block_nodes[block_idx].index(v)
                        trust = block_trust_matrices[block_idx][u_local_idx, v_local_idx]
                        edge_color = trust_cmap(trust)
                        alpha = 0.6
                        width = 1
                    
                    ax.plot([pos_u[0], pos_v[0]], 
                           [pos_u[1], pos_v[1]], 
                           [pos_u[2], pos_v[2]], 
                           color=edge_color, alpha=alpha, linewidth=width)
        
        # Draw inter-block connections in second half of animation
        if edges_progress > 0.3:
            visible_inter_block_edges = []
            for u, v in inter_block_edges:
                block_u = u // n_nodes_per_block
                block_v = v // n_nodes_per_block
                if block_u < blocks_to_show and block_v < blocks_to_show:
                    visible_inter_block_edges.append((u, v))
            
            # Only show a fraction of edges based on edge_progress
            scale_factor = (edges_progress - 0.3) / 0.7  # Rescale to 0-1
            num_edges_to_show = int(len(visible_inter_block_edges) * min(1.0, scale_factor))
            
            for idx, (u, v) in enumerate(visible_inter_block_edges):
                if idx < num_edges_to_show:
                    pos_u = node_positions_3d[u]
                    pos_v = node_positions_3d[v]
                    
                    # Color based on inter-block trust
                    block_u = u // n_nodes_per_block
                    block_v = v // n_nodes_per_block
                    trust = inter_block_trust[block_u, block_v]
                    
                    if trust > 0.7:
                        edge_color = 'cyan'
                    elif trust > 0.4:
                        edge_color = 'yellow'
                    else:
                        edge_color = 'white'
                    
                    ax.plot([pos_u[0], pos_v[0]], 
                           [pos_u[1], pos_v[1]], 
                           [pos_u[2], pos_v[2]], 
                           color=edge_color, alpha=0.8, linewidth=1.5)
        
        # Add title and analysis information
        if data_frame < n_frames // 3:
            title = f"Blockchain Network Structure - Building Blocks: {blocks_to_show}/{n_blocks}"
            title_color = 'white'
        elif data_frame < 2 * n_frames // 3:
            # Show network types
            title = "Blockchain with Multiple Network Topologies"
            title_color = 'white'
        else:
            # Show analysis
            title = "Blockchain Network Analysis: Trust and Community Structure"
            title_color = '#77dd77'  # light green
        
        ax.set_title(title, color=title_color, y=0.98, fontsize=14)
        
        # Add legend for completed frames
        if data_frame >= n_frames - 5:
            # Create legend elements based on network types in the blockchain
            legend_elements = []
            
            # Basic elements
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            label='Node', markerfacecolor='white', markersize=8))
            
            # Add network types that are present
            if network_type == 'mixed':
                for type_idx, name in enumerate(['Random', 'Small-World', 'Scale-Free', 
                                               'Community', 'Spatial']):
                    color = network_colors[type_idx]
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                                    label=f'{name} Network'))
            else:
                # Add just the specific network type
                type_idx = {'random': 0, 'small_world': 1, 'scale_free': 2, 
                          'community': 3, 'spatial': 4}.get(network_type, 0)
                color = network_colors[type_idx]
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                              label=f'{network_type.replace("_", "-").title()} Network'))
            
            # Add connection types
            legend_elements.append(plt.Line2D([0], [0], color='white', lw=1.5, 
                                            label='Intra-Block Connection'))
            legend_elements.append(plt.Line2D([0], [0], color='cyan', lw=1.5, 
                                            label='High-Trust Cross-Block Connection'))
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.0, 0.9), fontsize=10, frameon=True, 
                     facecolor='black', edgecolor='white', labelcolor='white')
        
        # Rotate view for better 3D perception
        if frame < n_frames:
            angle = 20 + (frame % 15) * 2
        else:
            # Use the final frame's angle for all pause frames
            angle = 20 + ((n_frames - 1) % 15) * 2
            
        ax.view_init(elev=20, azim=angle)
        
        return ax
    
    # Create animation with additional pause frames
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    
    # Save animation
    anim.save(os.path.join(output_dir, filename), 
              writer='pillow', fps=5, dpi=120)
    plt.close(fig)
    
    print(f"Blockchain network visualization saved to {os.path.join(output_dir, filename)}")
    return anim

#########################
# Main function
#########################

def main():
    """Create different blockchain network visualizations"""
    # Create blockchain network visualizations with different configurations
    print("Creating blockchain network visualizations...")
    
    # 1. Mixed network types (default)
    print("\n1. Creating mixed network types blockchain...")
    create_blockchain_network_visualization(
        n_blocks=5, 
        n_nodes_per_block=20, 
        n_frames=30,
        network_type='mixed',
        filename='blockchain_mixed_networks.gif'
    )
    
    # 2. Small-world blockchain
    print("\n2. Creating small-world blockchain...")
    create_blockchain_network_visualization(
        n_blocks=4, 
        n_nodes_per_block=15, 
        n_frames=25,
        network_type='small_world',
        filename='blockchain_small_world.gif'
    )
    
    # 3. Scale-free blockchain
    print("\n3. Creating scale-free blockchain...")
    create_blockchain_network_visualization(
        n_blocks=4, 
        n_nodes_per_block=18, 
        n_frames=25,
        network_type='scale_free',
        filename='blockchain_scale_free.gif'
    )
    
    # 4. Community structure blockchain
    print("\n4. Creating community structure blockchain...")
    create_blockchain_network_visualization(
        n_blocks=3, 
        n_nodes_per_block=24, 
        n_frames=25,
        network_type='community',
        filename='blockchain_community.gif'
    )
    
    print("\nAll blockchain visualizations completed successfully!")

if __name__ == "__main__":
    main()

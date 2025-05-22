#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blockchainVisualizer.py

This script creates animated visualizations of networks over blockchain structures
represented as cylinders. Each block in the blockchain can have different network
topologies (random, small-world, scale-free, community, spatial).

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from itertools import combinations

# Set output directory to the current directory rather than creating a subfolder
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

#########################
# Blockchain Network Visualization
#########################

def create_blockchain_network_visualization(n_blocks=5, n_nodes_per_block=10, n_frames=40):
    """Create visualization of networks over blockchain represented as cylinders"""
    # Set up the figure with 3D projection
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Define the blockchain structure
    block_height = 0.5
    block_radius = 2.0
    spacing = 1.2 * block_height
    
    # Generate positions for nodes on each block
    block_nodes = {}
    node_positions_3d = {}
    
    for block_idx in range(n_blocks):
        block_nodes[block_idx] = list(range(block_idx * n_nodes_per_block, 
                                          (block_idx + 1) * n_nodes_per_block))
        
        # Position nodes in a circle on the cylinder
        z_position = block_idx * (block_height + spacing)
        
        for i, node_idx in enumerate(block_nodes[block_idx]):
            angle = 2 * np.pi * i / n_nodes_per_block
            x = block_radius * np.cos(angle)
            y = block_radius * np.sin(angle)
            z = z_position + block_height / 2
            
            node_positions_3d[node_idx] = (x, y, z)
    
    # Create connections: intra-block (within same block) and inter-block (between blocks)
    intra_block_edges = []
    inter_block_edges = []
    
    # Intra-block connections (more dense)
    for block_idx in range(n_blocks):
        nodes = block_nodes[block_idx]
        # Make each block a different graph type
        if block_idx % 5 == 0:  # Random network
            p = 0.3
            for i, j in combinations(nodes, 2):
                if np.random.random() < p:
                    intra_block_edges.append((i, j))
        elif block_idx % 5 == 1:  # Small-world
            # Connect to k nearest neighbors
            k = min(4, n_nodes_per_block - 1)
            for i, node in enumerate(nodes):
                for j in range(1, k//2 + 1):
                    intra_block_edges.append((node, nodes[(i+j) % n_nodes_per_block]))
                    intra_block_edges.append((node, nodes[(i-j) % n_nodes_per_block]))
        elif block_idx % 5 == 2:  # Scale-free (simplified)
            # Preferential attachment
            degrees = {node: 0 for node in nodes}
            for i in range(1, len(nodes)):
                source = nodes[i]
                # Choose target with probability proportional to degree + 1
                probs = [degrees[target] + 1 for target in nodes[:i]]
                target_idx = np.random.choice(range(i), p=np.array(probs)/sum(probs))
                target = nodes[target_idx]
                intra_block_edges.append((source, target))
                degrees[source] += 1
                degrees[target] += 1
        elif block_idx % 5 == 3:  # Community structure
            # Two communities per block
            comm1 = nodes[:n_nodes_per_block//2]
            comm2 = nodes[n_nodes_per_block//2:]
            
            # Dense connections within communities
            for i, j in combinations(comm1, 2):
                if np.random.random() < 0.7:
                    intra_block_edges.append((i, j))
            
            for i, j in combinations(comm2, 2):
                if np.random.random() < 0.7:
                    intra_block_edges.append((i, j))
            
            # Sparse connections between communities
            for i in comm1:
                for j in comm2:
                    if np.random.random() < 0.1:
                        intra_block_edges.append((i, j))
        else:  # Spatial (geometric)
            # Connect if distance on cylinder surface is below threshold
            for i, j in combinations(nodes, 2):
                pos_i = node_positions_3d[i]
                pos_j = node_positions_3d[j]
                # Calculate angular distance
                angle_i = np.arctan2(pos_i[1], pos_i[0])
                angle_j = np.arctan2(pos_j[1], pos_j[0])
                angle_diff = min(abs(angle_i - angle_j), 2*np.pi - abs(angle_i - angle_j))
                arc_distance = block_radius * angle_diff
                
                if arc_distance < block_radius * 0.5:
                    intra_block_edges.append((i, j))
    
    # Inter-block connections (cross-block edges, sparser)
    for block_idx in range(n_blocks - 1):
        upper_nodes = block_nodes[block_idx]
        lower_nodes = block_nodes[block_idx + 1]
        
        # Connect some nodes between adjacent blocks
        for upper_node in upper_nodes:
            # Each upper node connects to 1-2 nodes in the block below
            n_connections = np.random.randint(1, 3)
            targets = np.random.choice(lower_nodes, size=n_connections, replace=False)
            
            for target in targets:
                inter_block_edges.append((upper_node, target))
    
    # Add 5 pause frames at the end
    pause_frames = 5
    total_frames = n_frames + pause_frames
    
    # Create animation showing blockchain growth
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
            # Second half: all blocks visible, progressively show cross-block connections
            blocks_to_show = n_blocks
            edges_progress = min(1.0, (data_frame - n_frames // 2) / (n_frames // 2 - 1))
        
        # Draw blocks as cylinders
        for block_idx in range(min(blocks_to_show, n_blocks)):
            z = block_idx * (block_height + spacing)
            
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 32)
            x = block_radius * np.cos(theta)
            y = block_radius * np.sin(theta)
            
            # Draw bottom circle
            ax.plot(x, y, [z] * len(theta), color='white', alpha=0.5, linewidth=1.5)
            # Draw top circle
            ax.plot(x, y, [z + block_height] * len(theta), color='white', alpha=0.5, linewidth=1.5)
            
            # Create the sides of the cylinder with a gradient color
            rgba_colors = []
            for _ in range(len(theta)-1):
                if block_idx % 5 == 0:  # Random network - beige
                    rgba_colors.append([0.95, 0.91, 0.82, 0.3])
                elif block_idx % 5 == 1:  # Small-world - light orange
                    rgba_colors.append([0.91, 0.80, 0.66, 0.3])
                elif block_idx % 5 == 2:  # Scale-free - darker beige
                    rgba_colors.append([0.76, 0.60, 0.42, 0.3])
                elif block_idx % 5 == 3:  # Community - tan
                    rgba_colors.append([0.82, 0.71, 0.55, 0.3])
                else:  # Spatial - lavender
                    rgba_colors.append([0.90, 0.81, 1.0, 0.3])
            
            # Draw cylinder sides
            for i in range(len(theta)-1):
                ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], color='white', alpha=0.5, linewidth=1)
            
            # Draw nodes for this block
            for node_idx in block_nodes[block_idx]:
                node_pos = node_positions_3d[node_idx]
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                           color='white', s=30, edgecolors='black', alpha=0.8)
            
            # Draw intra-block edges for this block
            edges_for_block = [(u, v) for (u, v) in intra_block_edges 
                               if u in block_nodes[block_idx] and v in block_nodes[block_idx]]
            
            for u, v in edges_for_block:
                pos_u = node_positions_3d[u]
                pos_v = node_positions_3d[v]
                ax.plot([pos_u[0], pos_v[0]], 
                        [pos_u[1], pos_v[1]], 
                        [pos_u[2], pos_v[2]], 
                        color='white', alpha=0.6, linewidth=1)
        
        # Draw inter-block connections based on edge_progress
        visible_inter_block_edges = []
        for u, v in inter_block_edges:
            block_u = u // n_nodes_per_block
            block_v = v // n_nodes_per_block
            if block_u < blocks_to_show and block_v < blocks_to_show:
                visible_inter_block_edges.append((u, v))
        
        # Only show a fraction of edges based on edge_progress
        num_edges_to_show = int(len(visible_inter_block_edges) * edges_progress)
        for idx, (u, v) in enumerate(visible_inter_block_edges):
            if idx < num_edges_to_show:
                pos_u = node_positions_3d[u]
                pos_v = node_positions_3d[v]
                ax.plot([pos_u[0], pos_v[0]], 
                        [pos_u[1], pos_v[1]], 
                        [pos_u[2], pos_v[2]], 
                        color='cyan', alpha=0.8, linewidth=1.5)
        
        # Add title
        if data_frame < n_frames // 2:
            title = f"Blockchain Network Structure - Building Blocks: {blocks_to_show}/{n_blocks}"
        else:
            # During the pause frames, keep the title fixed at 100%
            percentage = 100 if edges_progress >= 1.0 else int(edges_progress * 100)
            title = f"Blockchain Network Structure - Connecting Cross-Block Links: {percentage}%"
        
        ax.set_title(title, color='white', y=0.98, fontsize=14)
        
        # Add legend for completed frames or during pause
        if data_frame >= n_frames - 1:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Node', 
                           markerfacecolor='white', markersize=8),
                plt.Line2D([0], [0], color='white', lw=1.5, label='Intra-Block Connection'),
                plt.Line2D([0], [0], color='cyan', lw=1.5, label='Cross-Block Connection')
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.0, 0.8), fontsize=12, frameon=True, 
                     facecolor='black', edgecolor='white', labelcolor='white')
        
        # Rotate view for better 3D perception
        # Keep the same rotation angle for pause frames to avoid jitter
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
    anim.save(os.path.join(output_dir, 'blockchain_network_visualization.gif'), 
              writer='pillow', fps=5, dpi=120)
    plt.close(fig)
    
    print(f"Blockchain network visualization saved to {os.path.join(output_dir, 'blockchain_network_visualization.gif')}")
    return anim

#########################
# Main function
#########################

def main():
    # Create blockchain network visualization
    print("Creating blockchain network visualization...")
    create_blockchain_network_visualization(n_blocks=5, n_nodes_per_block=12, n_frames=30)
    
    print("Blockchain visualization completed successfully!")

if __name__ == "__main__":
    main()

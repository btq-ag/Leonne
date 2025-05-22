#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
permutation_visualizer.py

This script creates animated visualizations of generalized permutations
represented as networks with different configurations.

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

# Import from our generalized permutations module
from generalizedPermutations import (
    FisherYates, 
    permutationOrbitFisherYates, 
    symmetricNDFisherYates, 
    consensusFisherYates,
    consensusShuffle
)

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Define colors for different permutation states
COLORS = {
    'permutation': '#4CAF50',  # Green
    'orbit': '#F44336',        # Red
    'neutral': '#2196F3',      # Blue
    'symmetric': '#FFC107',    # Yellow/Gold 
    'background': '#111111',
    'edge': '#555555',
    'highlight': '#9C27B0'     # Purple
}

def create_permutation_network(n=10, edge_probability=0.3, network_type='random'):
    """Create a network to visualize permutations on"""
    if network_type == 'random':
        G = nx.erdos_renyi_graph(n, edge_probability)
    elif network_type == 'small_world':
        G = nx.watts_strogatz_graph(n, 4, 0.3)
    elif network_type == 'scale_free':
        G = nx.barabasi_albert_graph(n, 2)
    else:  # default to complete graph
        G = nx.complete_graph(n)
    return G

def visualize_permutation_steps(shuffle_algorithm, input_set, network, filename, extraInfo=False):
    """
    Create a visualization showing each step of the permutation algorithm on a network.
    
    Parameters:
    -----------
    shuffle_algorithm : function
        The permutation algorithm to visualize
    input_set : list
        The initial set to permute
    network : networkx.Graph
        The network to visualize on
    filename : str
        Base name for output files
    """
    # Make deep copy of input so we don't modify the original
    working_set = input_set.copy()
    steps = []
    
    # Record initial state
    steps.append(working_set.copy())
    
    # Track the algorithm steps by monkey patching the algorithm
    original_algorithm = shuffle_algorithm
    
    # Define a wrapper to capture states
    def wrapper(input_set, extraInfo=False):
        nonlocal steps, working_set
        
        # Initialize working array and steps list
        working_set = input_set.copy()
        steps = [working_set.copy()]  # Record initial state
        
        # Execute the algorithm (gets the final state)
        result = original_algorithm(input_set, extraInfo)
        
        # Add the final state to be sure
        if isinstance(result, tuple):
            steps.append(result[0])
            return result
        else:
            steps.append(result)
            return result
    
    # Replace the algorithm with our wrapper
    wrapper.__name__ = original_algorithm.__name__
    
    # Run the algorithm to capture steps
    result = wrapper(input_set, extraInfo)
    
    # Create a visualization of each step
    fig = plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
    
    # Create static plots for each step
    for i, step_state in enumerate(steps):
        pos = nx.spring_layout(network)
        plt.clf()
        plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
        
        # Map permutation elements to node colors
        node_colors = []
        for node in network.nodes():
            if node < len(step_state) and node in step_state:
                idx = step_state.index(node)
                if idx == node:  # Fixed point
                    node_colors.append(COLORS['symmetric'])
                else:
                    node_colors.append(COLORS['permutation'])
            else:
                node_colors.append(COLORS['neutral'])
        
        nx.draw(network, pos, with_labels=True, node_color=node_colors, 
                edge_color=COLORS['edge'], width=1.5, font_color='white',
                node_size=500, font_size=10)
        
        plt.title(f"{shuffle_algorithm.__name__} - Step {i}", color='white')
        
        # Save the figure for this step
        plt.savefig(os.path.join(output_dir, f"{filename}_step_{i}.png"), 
                   dpi=300, facecolor=COLORS['background'], bbox_inches='tight')
        plt.close()
    
    # Create an animation
    create_permutation_animation(shuffle_algorithm, input_set, network, filename)
    
    return steps

def create_permutation_animation(shuffle_algorithm, input_set, network, filename, frames=30):
    """
    Create an animation of the permutation algorithm process on a network.
    """
    G = network
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    def animate(i):
        ax.clear()
        
        # For the first frame, show the initial state
        if i == 0:
            current_set = input_set.copy()
        # For the last frame, show the final permutation
        elif i == frames - 1:
            result = shuffle_algorithm(input_set.copy(), False)
            if isinstance(result, tuple):
                current_set = result[0]
            else:
                current_set = result
        # For intermediate frames, interpolate between initial and final
        else:
            # Run the algorithm to get the final state
            result = shuffle_algorithm(input_set.copy(), False)
            if isinstance(result, tuple):
                final_set = result[0]
            else:
                final_set = result
                
            # Create a partial permutation for this frame
            current_set = input_set.copy()
            fraction = i / (frames - 1)
            
            # Swap some elements based on the frame number
            for j in range(len(input_set)):
                if j < int(fraction * len(input_set)):
                    current_set[j] = final_set[j]
        
        # Map permutation elements to node colors
        node_colors = []
        for node in G.nodes():
            if node < len(current_set) and node in current_set:
                idx = current_set.index(node)
                if idx == node:  # Fixed point
                    node_colors.append(COLORS['symmetric'])
                else:
                    node_colors.append(COLORS['permutation'])
            else:
                node_colors.append(COLORS['neutral'])
        
        # Draw the network with the current state
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                edge_color=COLORS['edge'], width=1.5, font_color='white',
                node_size=500, font_size=10)
        
        # Add title with algorithm name and frame number
        ax.set_title(f"{shuffle_algorithm.__name__} - Frame {i+1}/{frames}", color='white')
        
        return ax,
    
    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
    
    # Save the animation
    anim.save(os.path.join(output_dir, f"{filename}.gif"), writer='pillow', fps=5, 
              dpi=100, savefig_kwargs={'facecolor': COLORS['background']})
    
    plt.close()

def visualize_permutation_networks(n_elements=5):
    """
    Create visualizations for different permutation algorithms on various network types.
    """
    # Define input sets
    input_set = list(range(n_elements))
    
    # Create different network types
    network_types = ['random', 'small_world', 'scale_free']
    networks = {
        'random': create_permutation_network(n_elements, 0.4, 'random'),
        'small_world': create_permutation_network(n_elements, 0.4, 'small_world'),
        'scale_free': create_permutation_network(n_elements, 0.4, 'scale_free')
    }
    
    # Define permutation algorithms
    algorithms = [
        FisherYates,
        permutationOrbitFisherYates,
        symmetricNDFisherYates,
        consensusFisherYates
    ]
    
    # Create visualizations for each algorithm on each network type
    for algorithm in algorithms:
        for network_type, network in networks.items():
            print(f"Creating visualization for {algorithm.__name__} on {network_type} network...")
            filename = f"{algorithm.__name__}_{network_type}_network"
            visualize_permutation_steps(algorithm, input_set, network, filename)

def visualize_edge_permutations():
    """
    Create visualizations for edge-based permutation algorithms.
    """
    # Define edge sets
    edge_sets = {
        'test': np.array([[1,2],[2,1],[2,3],[3,2],[4,4]]),
        'large': np.array([[1,1],[1,2],[2,1],[3,2],[3,3],[4,1],[5,2]]),
        'trivial': np.array([[1,1],[2,2],[3,3]])
    }
    
    # Create a bipartite graph to visualize
    for name, edge_set in edge_sets.items():
        # Create bipartite graph from edge set
        G = nx.Graph()
        
        # Add nodes for left and right sides
        left_nodes = set(edge_set[:,0])
        right_nodes = set(edge_set[:,1])
        
        # Add nodes with bipartite attribute
        for node in left_nodes:
            G.add_node(f"L{node}", bipartite=0)
        for node in right_nodes:
            G.add_node(f"R{node}", bipartite=1)
        
        # Add edges
        for edge in edge_set:
            G.add_edge(f"L{edge[0]}", f"R{edge[1]}")
        
        # Create visualization
        print(f"Creating visualization for edge permutation on {name} edge set...")
        plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
        
        # Use bipartite layout
        pos = nx.bipartite_layout(G, nodes=[f"L{node}" for node in left_nodes])
        
        # Draw the network
        nx.draw(G, pos, with_labels=True, 
                node_color=[COLORS['permutation'] if node.startswith('L') else COLORS['orbit'] for node in G.nodes()],
                edge_color=COLORS['edge'], width=1.5, font_color='white',
                node_size=500, font_size=10)
        
        plt.title(f"Edge Set: {name}", color='white')
        
        # Save the static visualization
        plt.savefig(os.path.join(output_dir, f"edge_set_{name}.png"), 
                   dpi=300, facecolor=COLORS['background'], bbox_inches='tight')
        plt.close()
        
        # Create animation for edge permutation
        create_edge_permutation_animation(edge_set, name)

def create_edge_permutation_animation(edge_set, name, frames=30):
    """
    Create animation showing edge permutation process.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create bipartite graph from edge set
    G = nx.Graph()
    
    # Add nodes for left and right sides
    left_nodes = set(edge_set[:,0])
    right_nodes = set(edge_set[:,1])
    
    # Add nodes with bipartite attribute
    for node in left_nodes:
        G.add_node(f"L{node}", bipartite=0)
    for node in right_nodes:
        G.add_node(f"R{node}", bipartite=1)
    
    # Use bipartite layout
    pos = nx.bipartite_layout(G, nodes=[f"L{node}" for node in left_nodes])
    
    # Get final permutation
    final_edge_set, _ = consensusShuffle(edge_set.copy(), False)
    
    def animate(i):
        ax.clear()
        
        # Create a partial permutation for this frame
        current_edge_set = edge_set.copy()
        fraction = i / (frames - 1)
        
        # Interpolate between initial and final state
        if i > 0 and i < frames - 1:
            # Swap some edges based on the frame number
            swap_count = int(fraction * len(edge_set))
            for j in range(swap_count):
                current_edge_set[j] = final_edge_set[j]
        
        if i == frames - 1:
            current_edge_set = final_edge_set
        
        # Update graph with current edge set
        G.clear()
        left_nodes = set(current_edge_set[:,0])
        right_nodes = set(current_edge_set[:,1])
        
        # Add nodes
        for node in left_nodes:
            G.add_node(f"L{node}", bipartite=0)
        for node in right_nodes:
            G.add_node(f"R{node}", bipartite=1)
        
        # Add edges
        for edge in current_edge_set:
            G.add_edge(f"L{edge[0]}", f"R{edge[1]}")
        
        # Draw the network
        nx.draw(G, pos, ax=ax, with_labels=True, 
                node_color=[COLORS['permutation'] if node.startswith('L') else COLORS['orbit'] for node in G.nodes()],
                edge_color=COLORS['edge'], width=1.5, font_color='white',
                node_size=500, font_size=10)
        
        ax.set_title(f"Edge Permutation - {name} - Frame {i+1}/{frames}", color='white')
        
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
    
    # Save the animation
    anim.save(os.path.join(output_dir, f"edge_permutation_{name}.gif"), writer='pillow', fps=5, 
              dpi=100, savefig_kwargs={'facecolor': COLORS['background']})
    
    plt.close()

if __name__ == "__main__":
    print("Generating permutation network visualizations...")
    visualize_permutation_networks(n_elements=7)
    
    print("Generating edge permutation visualizations...")
    visualize_edge_permutations()
    
    print("All visualizations complete!")

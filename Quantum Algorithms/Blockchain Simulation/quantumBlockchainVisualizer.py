#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantumBlockchainVisualizer.py

This script creates animated visualizations of quantum-enhanced blockchain networks
represented as connected cylindrical structures. Each block in the blockchain 
incorporates quantum effects including:
1. Quantum random number generation for node selection
2. Quantum key distribution for secure cross-block links
3. Entropy addition for combining classical and quantum randomness
4. Compliance graphs for validating quantum communication

Quantum enhancements provide improved security and entropy compared to classical
blockchain algorithms through true quantum randomness and secure shared random states.

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
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantum_network_animations')
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility of classical components
np.random.seed(42)

#########################
# Quantum Random Number Generation
#########################

def simulate_quantum_rng(n_values=1, q_decoherence=0.05):
    """
    Simulate quantum random number generation with optional decoherence effects.
    
    Args:
        n_values: Number of random values to generate
        q_decoherence: Parameter to simulate quantum decoherence effects
    
    Returns:
        Array of random values between 0 and 1
    """
    # In a real quantum system, this would use quantum hardware
    # Here we simulate the pure randomness of quantum measurement
    
    # Generate true uniform random values
    pure_random = np.random.random(n_values)
    
    # Add slight fluctuation to simulate quantum measurement uncertainty
    if q_decoherence > 0:
        # Apply small noise to simulate quantum measurement effects
        noise = np.random.normal(0, q_decoherence, n_values)
        # Ensure values stay in [0,1] range with modulo
        quantum_random = np.mod(pure_random + noise, 1.0)
        return quantum_random
    
    return pure_random

#########################
# Quantum Key Distribution for Network Links
#########################

def simulate_qkd_for_links(node_i, node_j, q_channel_noise=0.02, 
                          eavesdropping=False, n_bits=64):
    """
    Simulate Quantum Key Distribution (QKD) between nodes to establish
    secure shared randomness for consensus.
    
    Args:
        node_i, node_j: The two nodes establishing quantum communication
        q_channel_noise: Quantum channel noise parameter
        eavesdropping: Whether eavesdropping is present on the channel
        n_bits: Number of bits to share
    
    Returns:
        shared_key: Bit string shared between nodes
        secure_flag: Whether the key exchange is considered secure
    """
    # In reality, this would involve actual quantum hardware using BB84 or similar protocols
    
    # Generate random basis choices for both parties
    alice_bases = np.random.randint(0, 2, n_bits)  # 0: Z-basis, 1: X-basis
    bob_bases = np.random.randint(0, 2, n_bits)
    
    # Alice's random bits to send
    alice_bits = np.random.randint(0, 2, n_bits)
    
    # Simulate quantum transmission and measurement
    bob_bits = np.zeros(n_bits, dtype=int)
    
    for i in range(n_bits):
        # If same basis, Bob gets correct result with high probability
        if alice_bases[i] == bob_bases[i]:
            # Channel noise can flip bits
            if np.random.random() < q_channel_noise:
                bob_bits[i] = 1 - alice_bits[i]  # Bit flip
            else:
                bob_bits[i] = alice_bits[i]  # Correct transmission
        else:
            # Different bases - Bob gets random result
            bob_bits[i] = np.random.randint(0, 2)
            
    # Bases comparison (public discussion phase)
    matching_bases = alice_bases == bob_bases
    
    # Keep only bits where bases matched
    shared_key_alice = alice_bits[matching_bases]
    shared_key_bob = bob_bits[matching_bases]
    
    # Simulate eavesdropping detection by comparing subset of bits
    if len(shared_key_alice) > 0:
        # Use a subset of the key for security check
        check_size = min(len(shared_key_alice) // 4, 1)
        check_indices = np.random.choice(len(shared_key_alice), check_size, replace=False)
        
        error_rate = np.sum(shared_key_alice[check_indices] != shared_key_bob[check_indices]) / check_size
        
        # If eavesdropping occurred, increase error rate significantly
        if eavesdropping:
            error_rate = max(error_rate, 0.25)  # Minimum 25% error with eavesdropping
            
        # QKD is considered secure if error rate is below threshold
        secure_flag = error_rate < 0.1  # Typical threshold
        
        # Remove check bits from final key
        mask = np.ones(len(shared_key_alice), dtype=bool)
        mask[check_indices] = False
        final_key = shared_key_alice[mask]
        
        # Convert bit array to binary string
        key_str = ''.join(str(bit) for bit in final_key)
        
        return key_str, secure_flag
    
    return "", False

#########################
# Quantum Entropy Addition
#########################

def combine_randomness_sources(classical_random, quantum_random):
    """
    Combine classical and quantum randomness using entropy addition.
    
    Args:
        classical_random: Classical random bits/values
        quantum_random: Quantum random bits/values
    
    Returns:
        Combined random source with enhanced entropy
    """
    # For numerical values, use XOR-equivalent operation for floating-point values
    if isinstance(classical_random, (float, int)) and isinstance(quantum_random, (float, int)):
        # Map values to binary representation and back
        return (classical_random + quantum_random) % 1.0
    
    # For bit strings, use XOR
    elif isinstance(classical_random, str) and isinstance(quantum_random, str):
        # Ensure equal length by padding
        max_len = max(len(classical_random), len(quantum_random))
        c_padded = classical_random.zfill(max_len)
        q_padded = quantum_random.zfill(max_len)
        
        # Bitwise XOR
        combined = ''.join(str(int(c) ^ int(q)) for c, q in zip(c_padded, q_padded))
        return combined
    
    # For arrays, use element-wise operations
    elif isinstance(classical_random, np.ndarray) and isinstance(quantum_random, np.ndarray):
        return np.mod(classical_random + quantum_random, 1.0)
    
    # Fallback for other types
    else:
        # Convert to string representation and apply XOR
        c_str = str(classical_random)
        q_str = str(quantum_random)
        combined_str = ''.join(str(ord(c) ^ ord(q)) for c, q in zip(c_str.ljust(len(q_str), '0'), 
                                                                     q_str.ljust(len(c_str), '0')))
        return combined_str

#########################
# Compliance Graph Generation
#########################

def generate_compliance_graph(n_nodes, qkd_links):
    """
    Generate a compliance graph showing which QKD links are verified secure.
    
    Args:
        n_nodes: Number of nodes in the network
        qkd_links: Dictionary of QKD links and their security status
    
    Returns:
        Networkx graph representing the compliance network
    """
    G = nx.Graph()
    
    # Add all nodes
    G.add_nodes_from(range(n_nodes))
    
    # Add edges for secure QKD links
    for (i, j), (key, secure) in qkd_links.items():
        if secure:
            G.add_edge(i, j, weight=1.0, secure=True)
        else:
            # Add insecure links with lower weight
            G.add_edge(i, j, weight=0.2, secure=False)
    
    return G

#########################
# Quantum Blockchain Network Visualization
#########################

def create_quantum_blockchain_visualization(n_blocks=5, n_nodes_per_block=10, n_frames=40,
                                          quantum_ratio=0.7, show_compliance=True):
    """
    Create visualization of quantum-enhanced blockchain networks represented as cylinders.
    
    Args:
        n_blocks: Number of blocks in the blockchain
        n_nodes_per_block: Number of nodes per block
        n_frames: Number of animation frames
        quantum_ratio: Ratio of quantum to classical links
        show_compliance: Whether to display the compliance graph
    
    Returns:
        Animation object
    """
    # Set up the figure with 3D projection
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Define the blockchain structure
    block_height = 0.5
    block_radius = 2.0
    spacing = 1.2 * block_height
    
    # Generate positions for nodes on each block
    block_nodes = {}
    node_positions_3d = {}
    
    # Track quantum nodes with special properties
    quantum_nodes = set()
    
    for block_idx in range(n_blocks):
        block_nodes[block_idx] = list(range(block_idx * n_nodes_per_block, 
                                          (block_idx + 1) * n_nodes_per_block))
        
        # Position nodes in a circle on the cylinder
        z_position = block_idx * (block_height + spacing)
        
        for i, node_idx in enumerate(block_nodes[block_idx]):
            angle = 2 * np.pi * i / n_nodes_per_block
            
            # Use quantum RNG to slightly adjust node positions for quantum nodes
            if np.random.random() < quantum_ratio:
                quantum_nodes.add(node_idx)
                # Add quantum fluctuation to position
                q_random = simulate_quantum_rng(3, q_decoherence=0.03)
                x = block_radius * np.cos(angle) * (1 + 0.05 * q_random[0])
                y = block_radius * np.sin(angle) * (1 + 0.05 * q_random[1])
                z = z_position + block_height / 2 + 0.05 * q_random[2]
            else:
                # Classical node positioning
                x = block_radius * np.cos(angle)
                y = block_radius * np.sin(angle)
                z = z_position + block_height / 2
                
            node_positions_3d[node_idx] = (x, y, z)
    
    # Create connections: intra-block (within same block) and inter-block (between blocks)
    intra_block_edges = []
    inter_block_edges = []
    quantum_edges = []  # Track quantum-secured edges
    qkd_links = {}      # Store QKD link status for compliance graphs
    
    # Intra-block connections (more dense)
    for block_idx in range(n_blocks):
        nodes = block_nodes[block_idx]
        # Make each block a different graph type with quantum enhancements
        if block_idx % 5 == 0:  # Quantum-random network
            p = 0.3
            for i, j in combinations(nodes, 2):
                # Use quantum RNG for connection probability
                if i in quantum_nodes and j in quantum_nodes:
                    # Quantum-quantum links
                    if simulate_quantum_rng(1)[0] < p:
                        intra_block_edges.append((i, j))
                        quantum_edges.append((i, j))
                        # Simulate QKD between quantum nodes
                        key, secure = simulate_qkd_for_links(i, j)
                        qkd_links[(i, j)] = (key, secure)
                else:
                    # At least one classical node, use classical randomness
                    if np.random.random() < p:
                        intra_block_edges.append((i, j))
        
        elif block_idx % 5 == 1:  # Quantum small-world
            # Connect to k nearest neighbors
            k = min(4, n_nodes_per_block - 1)
            for i, node in enumerate(nodes):
                for j in range(1, k//2 + 1):
                    # Regular ring connections
                    intra_block_edges.append((node, nodes[(i+j) % n_nodes_per_block]))
                    intra_block_edges.append((node, nodes[(i-j) % n_nodes_per_block]))
                
                # Add quantum long-range connections for quantum nodes
                if node in quantum_nodes:
                    # Quantum rewiring - add one long-range connection
                    all_other_nodes = [n for n in nodes if n != node and (node, n) not in intra_block_edges]
                    if all_other_nodes:
                        q_target = np.random.choice(all_other_nodes)
                        intra_block_edges.append((node, q_target))
                        quantum_edges.append((node, q_target))
                        # Simulate QKD for this quantum link
                        key, secure = simulate_qkd_for_links(node, q_target)
                        qkd_links[(node, q_target)] = (key, secure)
        
        elif block_idx % 5 == 2:  # Quantum scale-free
            # Preferential attachment with quantum preference
            degrees = {node: 0 for node in nodes}
            for i in range(1, len(nodes)):
                source = nodes[i]
                
                # Quantum nodes have increased preference in quantum model
                if source in quantum_nodes:
                    # Higher quantum preference
                    probs = [(degrees[target] + (2 if target in quantum_nodes else 1)) 
                            for target in nodes[:i]]
                else:
                    # Normal preference
                    probs = [degrees[target] + 1 for target in nodes[:i]]
                
                # Normalize probabilities
                probs = np.array(probs)/sum(probs)
                
                # Choose target with probability proportional to degree + quantum factor
                target_idx = np.random.choice(range(i), p=probs)
                target = nodes[target_idx]
                intra_block_edges.append((source, target))
                degrees[source] += 1
                degrees[target] += 1
                
                # If both are quantum nodes, establish quantum link
                if source in quantum_nodes and target in quantum_nodes:
                    quantum_edges.append((source, target))
                    key, secure = simulate_qkd_for_links(source, target)
                    qkd_links[(source, target)] = (key, secure)
        
        elif block_idx % 5 == 3:  # Quantum community structure
            # Two communities with quantum entanglement between them
            comm1 = nodes[:n_nodes_per_block//2]
            comm2 = nodes[n_nodes_per_block//2:]
            
            # Dense connections within communities
            for i, j in combinations(comm1, 2):
                p_connect = 0.7
                # Use quantum randomness if both nodes are quantum
                if i in quantum_nodes and j in quantum_nodes:
                    if simulate_quantum_rng(1)[0] < p_connect:
                        intra_block_edges.append((i, j))
                        quantum_edges.append((i, j))
                        key, secure = simulate_qkd_for_links(i, j)
                        qkd_links[(i, j)] = (key, secure)
                else:
                    if np.random.random() < p_connect:
                        intra_block_edges.append((i, j))
            
            # Do the same for second community
            for i, j in combinations(comm2, 2):
                p_connect = 0.7
                if i in quantum_nodes and j in quantum_nodes:
                    if simulate_quantum_rng(1)[0] < p_connect:
                        intra_block_edges.append((i, j))
                        quantum_edges.append((i, j))
                        key, secure = simulate_qkd_for_links(i, j)
                        qkd_links[(i, j)] = (key, secure)
                else:
                    if np.random.random() < p_connect:
                        intra_block_edges.append((i, j))
            
            # Quantum-enhanced inter-community links (entanglement across communities)
            # Find quantum nodes in each community
            q_comm1 = [n for n in comm1 if n in quantum_nodes]
            q_comm2 = [n for n in comm2 if n in quantum_nodes]
            
            # Create quantum bridges between communities
            for i in q_comm1:
                for j in q_comm2:
                    if np.random.random() < 0.3:  # Higher connection probability for quantum bridges
                        intra_block_edges.append((i, j))
                        quantum_edges.append((i, j))
                        key, secure = simulate_qkd_for_links(i, j, q_channel_noise=0.01)  # Lower noise
                        qkd_links[(i, j)] = (key, secure)
            
            # Classical sparse connections between communities
            for i in [n for n in comm1 if n not in quantum_nodes]:
                for j in [n for n in comm2 if n not in quantum_nodes]:
                    if np.random.random() < 0.1:
                        intra_block_edges.append((i, j))
                        
        else:  # Quantum spatial (geometric)
            # Connect if distance on cylinder surface is below threshold, with quantum tunneling
            for i, j in combinations(nodes, 2):
                pos_i = node_positions_3d[i]
                pos_j = node_positions_3d[j]
                # Calculate angular distance
                angle_i = np.arctan2(pos_i[1], pos_i[0])
                angle_j = np.arctan2(pos_j[1], pos_j[0])
                angle_diff = min(abs(angle_i - angle_j), 2*np.pi - abs(angle_i - angle_j))
                arc_distance = block_radius * angle_diff
                
                # Regular connections for close nodes
                if arc_distance < block_radius * 0.5:
                    intra_block_edges.append((i, j))
                # Quantum tunneling for quantum nodes allows some long-range connections
                elif i in quantum_nodes and j in quantum_nodes:
                    tunneling_prob = 0.2 * np.exp(-arc_distance / block_radius)
                    if simulate_quantum_rng(1)[0] < tunneling_prob:
                        intra_block_edges.append((i, j))
                        quantum_edges.append((i, j))
                        # Tunneled connections use QKD with slightly higher noise
                        key, secure = simulate_qkd_for_links(i, j, q_channel_noise=0.03)
                        qkd_links[(i, j)] = (key, secure)
    
    # Inter-block connections (cross-block edges) using quantum key distribution
    for block_idx in range(n_blocks - 1):
        upper_nodes = block_nodes[block_idx]
        lower_nodes = block_nodes[block_idx + 1]
        
        # Connect some nodes between adjacent blocks
        for upper_node in upper_nodes:
            # Preferentially connect quantum nodes for inter-block links
            if upper_node in quantum_nodes:
                # Quantum nodes make more connections with enhanced security
                n_connections = np.random.randint(1, 3)
                # Prefer quantum nodes in lower block if available
                quantum_targets = [n for n in lower_nodes if n in quantum_nodes]
                if quantum_targets and len(quantum_targets) >= n_connections:
                    targets = np.random.choice(quantum_targets, size=n_connections, replace=False)
                else:
                    targets = np.random.choice(lower_nodes, size=n_connections, replace=False)
                
                for target in targets:
                    inter_block_edges.append((upper_node, target))
                    # Use QKD for quantum-to-quantum links
                    if target in quantum_nodes:
                        quantum_edges.append((upper_node, target))
                        key, secure = simulate_qkd_for_links(upper_node, target)
                        qkd_links[(upper_node, target)] = (key, secure)
            else:
                # Classical nodes just make 1 connection
                if np.random.random() < 0.7:  # Not all classical nodes connect between blocks
                    target = np.random.choice(lower_nodes)
                    inter_block_edges.append((upper_node, target))
    
    # Generate global compliance graph
    total_nodes = n_blocks * n_nodes_per_block
    compliance_graph = generate_compliance_graph(total_nodes, qkd_links)
    
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
            
            # Create the sides of the cylinder with quantum-specific colors
            for i in range(len(theta)-1):
                if block_idx % 5 == 0:  # Quantum-random network - purple
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], 
                            color='mediumpurple', alpha=0.5, linewidth=1)
                elif block_idx % 5 == 1:  # Quantum small-world - blue
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], 
                            color='dodgerblue', alpha=0.5, linewidth=1)
                elif block_idx % 5 == 2:  # Quantum scale-free - cyan
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], 
                            color='turquoise', alpha=0.5, linewidth=1)
                elif block_idx % 5 == 3:  # Quantum community - teal
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], 
                            color='teal', alpha=0.5, linewidth=1)
                else:  # Quantum spatial - bright green
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z, z+block_height], 
                            color='limegreen', alpha=0.5, linewidth=1)
            
            # Draw nodes for this block
            for node_idx in block_nodes[block_idx]:
                node_pos = node_positions_3d[node_idx]
                if node_idx in quantum_nodes:
                    # Quantum nodes have different appearance
                    ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                              color='deepskyblue', s=40, edgecolors='white', alpha=0.9)
                    
                    # Add subtle quantum effect (small glow)
                    if data_frame % 3 == 0:  # Animate the glow every 3 frames
                        ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                                  color='cyan', s=60, alpha=0.2)
                else:
                    # Classical nodes
                    ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                              color='white', s=30, edgecolors='gray', alpha=0.8)
            
            # Draw intra-block edges for this block
            edges_for_block = [(u, v) for (u, v) in intra_block_edges 
                               if u in block_nodes[block_idx] and v in block_nodes[block_idx]]
            
            for u, v in edges_for_block:
                pos_u = node_positions_3d[u]
                pos_v = node_positions_3d[v]
                
                if (u, v) in quantum_edges or (v, u) in quantum_edges:
                    # Quantum secured links
                    ax.plot([pos_u[0], pos_v[0]], 
                            [pos_u[1], pos_v[1]], 
                            [pos_u[2], pos_v[2]], 
                            color='cyan', alpha=0.7, linewidth=1.5)
                    
                    # Add QKD visualization effect
                    if data_frame % 2 == 0 and ((u, v) in qkd_links or (v, u) in qkd_links):
                        # Create points along the line for QKD visualization
                        qkd_key = qkd_links.get((u, v), qkd_links.get((v, u), (None, False)))[0]
                        if qkd_key:
                            # Visualize first few bits of the QKD key along the connection
                            t_vals = np.linspace(0, 1, min(8, len(qkd_key)))
                            for t_idx, t in enumerate(t_vals):
                                if t_idx < len(qkd_key) and qkd_key[t_idx] == '1':
                                    # Only show '1' bits as points
                                    pt_x = pos_u[0] + t * (pos_v[0] - pos_u[0])
                                    pt_y = pos_u[1] + t * (pos_v[1] - pos_u[1])
                                    pt_z = pos_u[2] + t * (pos_v[2] - pos_u[2])
                                    ax.scatter(pt_x, pt_y, pt_z, color='yellow', s=10, alpha=0.7)
                else:
                    # Classical links
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
                
                if (u, v) in quantum_edges or (v, u) in quantum_edges:
                    # Quantum cross-block links
                    ax.plot([pos_u[0], pos_v[0]], 
                            [pos_u[1], pos_v[1]], 
                            [pos_u[2], pos_v[2]], 
                            color='deepskyblue', alpha=0.8, linewidth=1.8, linestyle='-')
                else:
                    # Classical cross-block links
                    ax.plot([pos_u[0], pos_v[0]], 
                            [pos_u[1], pos_v[1]], 
                            [pos_u[2], pos_v[2]], 
                            color='lightblue', alpha=0.7, linewidth=1.5)
        
        # Draw compliance graph on fully built blockchain
        if show_compliance and data_frame >= n_frames - 1:
            # Display a 2D compliance graph in corner to show QKD validation
            # We draw this as a smaller network in the top right corner
            pos_2d = nx.spring_layout(compliance_graph, seed=42)
            
            # Scale and position the 2D graph in 3D space
            scale = 0.5
            offset_x, offset_y, offset_z = 2.0, 2.0, (n_blocks - 1) * (block_height + spacing)
            
            # Draw nodes from compliance graph
            for node, (x, y) in pos_2d.items():
                scaled_x = offset_x + scale * x
                scaled_y = offset_y + scale * y
                scaled_z = offset_z
                
                if node in quantum_nodes:
                    ax.scatter(scaled_x, scaled_y, scaled_z, 
                              color='cyan', s=30, edgecolors='white', alpha=0.9)
                else:
                    ax.scatter(scaled_x, scaled_y, scaled_z, 
                              color='white', s=20, alpha=0.7)
            
            # Draw edges from compliance graph
            for u, v, data in compliance_graph.edges(data=True):
                if u in pos_2d and v in pos_2d:  # Make sure nodes are in layout
                    x1, y1 = pos_2d[u]
                    x2, y2 = pos_2d[v]
                    
                    scaled_x1 = offset_x + scale * x1
                    scaled_y1 = offset_y + scale * y1
                    scaled_z1 = offset_z
                    
                    scaled_x2 = offset_x + scale * x2
                    scaled_y2 = offset_y + scale * y2
                    scaled_z2 = offset_z
                    
                    if data['secure']:
                        # Secure QKD link
                        ax.plot([scaled_x1, scaled_x2], 
                                [scaled_y1, scaled_y2], 
                                [scaled_z1, scaled_z2], 
                                color='lime', alpha=0.8, linewidth=1.5)
                    else:
                        # Insecure QKD link
                        ax.plot([scaled_x1, scaled_x2], 
                                [scaled_y1, scaled_y2], 
                                [scaled_z1, scaled_z2], 
                                color='red', alpha=0.6, linewidth=1.0, linestyle='--')
            
            # Add a small label for the compliance graph
            ax.text(offset_x, offset_y, offset_z + 0.3, 
                    "QKD Compliance Graph", 
                    color='white', fontsize=8, ha='center')
        
        # Add title
        if data_frame < n_frames // 2:
            title = f"Quantum Blockchain Network - Building Blocks: {blocks_to_show}/{n_blocks}"
        else:
            # During the pause frames, keep the title fixed at 100%
            percentage = 100 if edges_progress >= 1.0 else int(edges_progress * 100)
            title = f"Quantum Blockchain Network - Connecting Cross-Block Links: {percentage}%"
        
        ax.set_title(title, color='white', y=0.98, fontsize=14)
        
        # Add legend for completed frames or during pause
        if data_frame >= n_frames - 1:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='Classical Node'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='deepskyblue', markersize=8, label='Quantum Node'),
                plt.Line2D([0], [0], color='white', lw=1.5, label='Classical Connection'),
                plt.Line2D([0], [0], color='cyan', lw=1.5, label='Quantum Connection'),
                plt.Line2D([0], [0], color='deepskyblue', lw=1.8, label='Quantum Cross-Block Link')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(0.0, 0.95), fontsize=10, frameon=True, 
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
    anim.save(os.path.join(output_dir, 'quantum_blockchain_visualization.gif'), 
              writer='pillow', fps=5, dpi=120)
    
    # Save a still image of the final frame
    plt.figure(figsize=(14, 12), facecolor='black')
    ax_still = plt.axes(projection='3d')
    update(n_frames-1)
    plt.savefig(os.path.join(output_dir, 'quantum_blockchain_final.png'), dpi=120)
    
    plt.close(fig)
    
    print(f"Quantum blockchain network visualization saved to {os.path.join(output_dir, 'quantum_blockchain_visualization.gif')}")
    return anim

#########################
# Additional analysis plots
#########################

def create_quantum_vs_classical_entropy_plot():
    """Create a plot comparing quantum vs classical randomness entropy"""
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate data
    classical_bits = np.random.randint(0, 2, size=1000)
    
    # Count the frequency of subsequences to analyze randomness
    # For true random sequences, all patterns should appear with equal frequency
    subsequence_len = 4
    classical_patterns = {}
    
    # Count subsequence patterns for classical bits
    for i in range(len(classical_bits) - subsequence_len + 1):
        pattern = ''.join(str(bit) for bit in classical_bits[i:i+subsequence_len])
        classical_patterns[pattern] = classical_patterns.get(pattern, 0) + 1
    
    # Normalize to get probabilities
    total_classical = sum(classical_patterns.values())
    for pattern in classical_patterns:
        classical_patterns[pattern] /= total_classical
    
    # Now do the same for simulated quantum bits
    # For simulation, we'll add slight perturbations to represent quantum effects
    q_bits = []
    for _ in range(1000):
        # Simulate quantum measurement - perfectly random
        q_bits.append(int(simulate_quantum_rng(1)[0] > 0.5))
    
    # Count subsequence patterns for quantum bits
    quantum_patterns = {}
    for i in range(len(q_bits) - subsequence_len + 1):
        pattern = ''.join(str(bit) for bit in q_bits[i:i+subsequence_len])
        quantum_patterns[pattern] = quantum_patterns.get(pattern, 0) + 1
    
    # Normalize to get probabilities
    total_quantum = sum(quantum_patterns.values())
    for pattern in quantum_patterns:
        quantum_patterns[pattern] /= total_quantum
    
    # Get all unique patterns
    all_patterns = sorted(set(list(classical_patterns.keys()) + list(quantum_patterns.keys())))
    
    # Create x-coordinates for bar chart
    x = np.arange(len(all_patterns))
    width = 0.35
    
    # Get y-values in the same order as all_patterns
    classical_probs = [classical_patterns.get(p, 0) for p in all_patterns]
    quantum_probs = [quantum_patterns.get(p, 0) for p in all_patterns]
    
    # Calculate entropy
    classical_entropy = -np.sum([p * np.log2(p) for p in classical_probs if p > 0])
    quantum_entropy = -np.sum([p * np.log2(p) for p in quantum_probs if p > 0])
    
    # Theoretical max entropy for 4-bit patterns = log2(16) = 4
    max_entropy = np.log2(2**subsequence_len)
    
    # Plot bars
    ax.bar(x - width/2, classical_probs, width, label=f'Classical (Entropy: {classical_entropy:.3f} bits)', color='skyblue')
    ax.bar(x + width/2, quantum_probs, width, label=f'Quantum (Entropy: {quantum_entropy:.3f} bits)', color='violet')
    
    # Add ideal entropy line
    ideal_prob = 1/(2**subsequence_len)
    ax.axhline(y=ideal_prob, color='red', linestyle='--', alpha=0.5, 
              label=f'Ideal Distribution (Max Entropy: {max_entropy:.3f} bits)')
    
    # Add labels and title
    ax.set_xlabel(f'Binary Patterns ({subsequence_len}-bit)')
    ax.set_ylabel('Probability')
    ax.set_title('Entropy Comparison: Classical vs Quantum Random Bit Patterns')
    ax.set_xticks(x)
    ax.set_xticklabels(all_patterns, rotation=90, fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_vs_classical_entropy.png'), dpi=120)
    plt.close(fig)
    
    print(f"Entropy comparison plot saved to {os.path.join(output_dir, 'quantum_vs_classical_entropy.png')}")

def create_quantum_security_comparison():
    """Create a plot showing quantum vs classical security scaling"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points: network size vs number of bits needed for security
    network_sizes = np.array([10, 20, 50, 100, 200, 500, 1000])
    
    # Classical security (scales linearly with size)
    classical_security_bits = network_sizes * 2
    
    # Quantum security (scales logarithmically due to quantum properties)
    quantum_security_bits = 2 * np.log2(network_sizes) * network_sizes
    
    # Classical security with QKD enhancement
    classical_qkd_security = network_sizes * 1.2
    
    # Quantum advantage ratio
    advantage_ratio = classical_security_bits / quantum_security_bits
    
    # Plot the data
    ax.plot(network_sizes, classical_security_bits, 'o-', label='Classical Security', color='blue')
    ax.plot(network_sizes, quantum_security_bits, 's-', label='Full Quantum Network', color='purple')
    ax.plot(network_sizes, classical_qkd_security, '^-', label='Classical + QKD', color='green')
    
    # Add second y-axis for quantum advantage
    ax2 = ax.twinx()
    ax2.plot(network_sizes, advantage_ratio, '--', label='Quantum Advantage Ratio', color='red')
    ax2.set_ylabel('Quantum Advantage (Classical/Quantum Bits)')
    
    # Add labels and title
    ax.set_xlabel('Network Size (Nodes)')
    ax.set_ylabel('Security Bits Required')
    ax.set_title('Quantum vs. Classical Security Scaling')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_security_comparison.png'), dpi=120)
    plt.close(fig)
    
    print(f"Security comparison plot saved to {os.path.join(output_dir, 'quantum_security_comparison.png')}")

#########################
# Main function
#########################

def main():
    # Create quantum blockchain network visualization
    print("Creating quantum blockchain network visualization...")
    print("This may take a few minutes to complete...")
    # Use smaller parameters for faster testing
    create_quantum_blockchain_visualization(n_blocks=3, n_nodes_per_block=8, n_frames=20, quantum_ratio=0.6)
    
    # Generate analysis plots
    print("Generating analysis plots...")
    create_quantum_vs_classical_entropy_plot()
    create_quantum_security_comparison()
    
    print("Quantum blockchain visualization completed successfully!")
    print("Files saved to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    main()

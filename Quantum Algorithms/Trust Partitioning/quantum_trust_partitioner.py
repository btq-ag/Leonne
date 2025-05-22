#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_trust_partitioner.py

Core implementation of quantum trust partitioning algorithm with topological features.
This file contains the main algorithms for quantum-enhanced trust partitioning,
including persistent homology analysis and topological features.

Key components:
1. Quantum-enhanced trust metrics
2. Trust-driven partitioning algorithm
3. Persistent homology computation
4. Betti numbers analysis
5. Stability analysis

This is a consolidated implementation that incorporates functionality from:
- quantumTrustPartitioner.py
- quantumBettiPartitioner.py
- quantum_trust_topology.py

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import time
from tqdm import tqdm
import gudhi as gd
from itertools import combinations
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

#########################
# Quantum Random Number Generation
#########################

def simulate_quantum_random_bit():
    """
    Simulate a quantum random bit using classical randomness
    This simulates the output of a quantum circuit that creates superposition
    and then measures in the computational basis
    
    Returns:
    --------
    int
        0 or 1 with equal probability
    """
    return random.randint(0, 1)

def generate_quantum_random_number(bits=8):
    """
    Generate a quantum-inspired random number
    
    Parameters:
    -----------
    bits : int, default=8
        Number of bits for the random number
        
    Returns:
    --------
    int
        Random number between 0 and 2^bits - 1
    """
    result = 0
    for i in range(bits):
        result = (result << 1) | simulate_quantum_random_bit()
    return result

#########################
# Quantum Trust Matrix Functions
#########################

def create_initial_quantum_trust_matrix(n_nodes, network_type='random'):
    """
    Create an initial trust matrix for a quantum-enhanced network
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    network_type : str, default='random'
        Type of network ('random', 'small_world', 'scale_free', 'community')
        
    Returns:
    --------
    trust_matrix : numpy.ndarray
        Initial trust matrix
    """
    # Create base network structure
    G = None
    
    if network_type == 'random':
        # Erdős-Rényi random graph
        p = 0.3  # Probability of edge creation
        G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
        
    elif network_type == 'small_world':
        # Watts-Strogatz small-world graph
        k = 4  # Each node is connected to k nearest neighbors
        p = 0.3  # Probability of rewiring each edge
        G = nx.watts_strogatz_graph(n_nodes, k, p, seed=42)
        
    elif network_type == 'scale_free':
        # Barabási-Albert scale-free graph
        m = 2  # Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
        
    elif network_type == 'community':
        # Generate network with community structure
        n_communities = max(2, n_nodes // 10)
        sizes = [n_nodes // n_communities] * n_communities
        sizes[-1] += n_nodes % n_communities  # Add remainder to last community
        p_in = 0.7  # Probability of edge within community
        p_out = 0.1  # Probability of edge between communities
        G = nx.random_partition_graph(sizes, p_in, p_out, seed=42)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Generate trust matrix from graph
    adj_matrix = nx.to_numpy_array(G)
    
    # Convert adjacency to trust values
    trust_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                trust_matrix[i, j] = 1.0  # Self-trust
            elif adj_matrix[i, j] == 1:
                # Connected nodes have higher trust with quantum fluctuation
                base_trust = 0.7 + 0.3 * np.random.random()
                quantum_factor = 1.0 + 0.1 * (generate_quantum_random_number() / 255.0 - 0.5)
                trust_matrix[i, j] = min(1.0, base_trust * quantum_factor)
            else:
                # Non-connected nodes have lower trust with quantum fluctuation
                base_trust = 0.1 + 0.2 * np.random.random()
                quantum_factor = 1.0 + 0.2 * (generate_quantum_random_number() / 255.0 - 0.5)
                trust_matrix[i, j] = max(0.0, base_trust * quantum_factor)
    
    return trust_matrix

def distribute_nodes_to_networks(n_nodes, n_networks, network_type='random'):
    """
    Distribute nodes into initial networks
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes
    n_networks : int
        Number of networks
    network_type : str, default='random'
        Type of network distribution ('random', 'balanced', 'skewed')
        
    Returns:
    --------
    networks : list
        List of networks, where each network is a list of nodes
    """
    nodes = list(range(n_nodes))
    networks = [[] for _ in range(n_networks)]
    
    if network_type == 'balanced':
        # Distribute nodes evenly across networks
        for i, node in enumerate(nodes):
            networks[i % n_networks].append(node)
            
    elif network_type == 'skewed':
        # One large network, rest are small
        main_network_size = n_nodes // 2
        networks[0] = nodes[:main_network_size]
        
        # Distribute remaining nodes
        remaining = nodes[main_network_size:]
        for i, node in enumerate(remaining):
            networks[(i % (n_networks - 1)) + 1].append(node)
            
    else:  # random
        # Randomly distribute nodes
        random.shuffle(nodes)
        min_size = max(1, n_nodes // (n_networks * 2))  # Ensure each network has some nodes
        
        # First ensure each network has minimum nodes
        for i in range(n_networks):
            for _ in range(min_size):
                if nodes:
                    networks[i].append(nodes.pop())
        
        # Distribute remaining nodes randomly
        for node in nodes:
            networks[random.randint(0, n_networks - 1)].append(node)
    
    # Remove any empty networks that might have been created
    networks = [net for net in networks if net]
    
    return networks

def construct_distance_matrix(trust_matrix):
    """
    Convert trust matrix to distance matrix for topological analysis
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
        
    Returns:
    --------
    distance_matrix : numpy.ndarray
        Distance matrix for topological analysis
    """
    # Symmetrize the trust matrix by taking the average
    symmetric_trust = (trust_matrix + trust_matrix.T) / 2
    
    # Convert trust to distance (1 - trust)
    distance_matrix = 1 - symmetric_trust
    
    # Ensure diagonal is 0 (self-distance)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

#########################
# Topological Analysis Functions
#########################

def compute_quantum_betti_numbers(distance_matrix, max_dimension=2, max_edge_length=1.0):
    """
    Compute Betti numbers using quantum-enhanced persistence homology
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Distance matrix between nodes
    max_dimension : int, default=2
        Maximum homology dimension
    max_edge_length : float, default=1.0
        Maximum edge length for filtration
        
    Returns:
    --------
    betti_numbers : dict
        Dictionary with Betti numbers for dimensions 0, 1, and 2
    """
    # Create Vietoris-Rips complex
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # Compute persistent homology
    simplex_tree.compute_persistence()
    
    # Extract Betti numbers at filtration value (quantum-enhanced)
    quantum_factor = 1.0 + 0.05 * (generate_quantum_random_number() / 255.0 - 0.5)
    filtration_value = max_edge_length * quantum_factor
    
    try:
        betti_numbers = simplex_tree.persistent_betti_numbers(0, filtration_value)
        
        # Make sure we have values for all dimensions
        b0 = betti_numbers[0] if len(betti_numbers) > 0 else 0  # Connected components
        b1 = betti_numbers[1] if len(betti_numbers) > 1 else 0  # Cycles/loops
        b2 = betti_numbers[2] if len(betti_numbers) > 2 else 0  # Voids
    except Exception as e:
        print(f"Error computing Betti numbers: {e}")
        # Fallback to default values
        b0, b1, b2 = 1, 0, 0
    
    return {
        'betti0': b0,
        'betti1': b1,
        'betti2': b2
    }

def compute_persistent_homology(distance_matrix, max_dimension=2):
    """
    Compute persistent homology for a distance matrix
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Distance matrix between nodes
    max_dimension : int, default=2
        Maximum homology dimension
        
    Returns:
    --------
    persistence_pairs : list
        List of persistent homology pairs (dimension, birth, death)
    """
    # Create Vietoris-Rips complex
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # Compute persistence
    simplex_tree.compute_persistence()
    
    # Get persistence pairs
    persistence = simplex_tree.get_persistence()
    
    # Convert persistence diagram to list of (dim, birth, death) tuples
    persistence_pairs = []
    for pair in simplex_tree.persistence_pairs():
        if len(pair[0]) == 1:
            # Birth of 0-dimensional feature (connected component)
            birth = 0.0
            death = simplex_tree.filtration(pair[1][0]) if pair[1] else float('inf')
            persistence_pairs.append((0, birth, death))
        else:
            # Higher dimensional features
            dimension = len(pair[0][0]) - 1  # Dimension of feature
            birth = simplex_tree.filtration(pair[0][0])
            death = simplex_tree.filtration(pair[1][0]) if pair[1] else float('inf')
            persistence_pairs.append((dimension, birth, death))
    
    return persistence_pairs

def compute_all_networks_betti_numbers(networks, trust_matrix):
    """
    Compute Betti numbers for all networks
    
    Parameters:
    -----------
    networks : list
        List of networks, where each network is a list of node IDs
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
        
    Returns:
    --------
    all_betti_numbers : dict
        Dictionary with Betti numbers for each network
    """
    all_betti_numbers = {
        'networks': [],
        'betti0': 0,
        'betti1': 0,
        'betti2': 0
    }
    
    for network_idx, network in enumerate(networks):
        if len(network) <= 2:
            # Too small for meaningful topology
            network_betti = {'betti0': 1, 'betti1': 0, 'betti2': 0}
        else:
            # Extract submatrix for this network
            indices = np.array(network)
            submatrix = trust_matrix[np.ix_(indices, indices)]
            
            # Convert to distance matrix
            distance_matrix = construct_distance_matrix(submatrix)
            
            # Compute Betti numbers
            network_betti = compute_quantum_betti_numbers(distance_matrix)
        
        # Store network-specific Betti numbers
        all_betti_numbers['networks'].append(network_betti)
        
        # Update total Betti numbers
        all_betti_numbers['betti0'] += network_betti['betti0']
        all_betti_numbers['betti1'] += network_betti['betti1']
        all_betti_numbers['betti2'] += network_betti['betti2']
    
    return all_betti_numbers

def adjust_trust_with_topology(trust_matrix, betti_numbers, networks):
    """
    Adjust trust matrix based on topological features
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
    betti_numbers : dict
        Dictionary with Betti numbers for each network
    networks : list
        List of networks, where each network is a list of node IDs
        
    Returns:
    --------
    adjusted_trust_matrix : numpy.ndarray
        Adjusted trust matrix
    """
    # Make a copy of the trust matrix
    adjusted_trust_matrix = trust_matrix.copy()
    
    # Adjust trust based on topological features
    for network_idx, network in enumerate(networks):
        if len(network) <= 1:
            continue
            
        # Get Betti numbers for this network
        network_betti = betti_numbers['networks'][network_idx]
        
        # Calculate topological adjustment factor
        # Higher connectivity (β₀) => higher trust
        # Higher cycles (β₁) => lower trust (more complex relationships)
        # Higher voids (β₂) => higher trust (more robust structure)
        topo_factor = (1.0 + 0.1 * network_betti['betti0'] -
                      0.05 * network_betti['betti1'] +
                      0.05 * network_betti['betti2'])
        
        # Adjust trust within network
        for i in network:
            for j in network:
                if i != j:
                    # Apply quantum fluctuation to topological adjustment
                    quantum_factor = 1.0 + 0.1 * (generate_quantum_random_number() / 255.0 - 0.5)
                    final_factor = topo_factor * quantum_factor
                    
                    # Apply adjustment, ensuring trust stays in [0, 1]
                    adjusted_trust_matrix[i, j] = min(1.0, max(0.0, 
                                                            trust_matrix[i, j] * final_factor))
    
    return adjusted_trust_matrix

#########################
# Core Partitioning Algorithm
#########################

def compute_trust_forward(trust_matrix, network):
    """
    Compute r_i(B) for each node i in network
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix
    network : list
        List of nodes in the network
        
    Returns:
    --------
    forward_trust : dict
        Dictionary mapping node to its average trust in the network
    """
    forward_trust = {}
    
    for node in network:
        if len(network) > 1:
            # Calculate average trust from node to other nodes in network
            trust_sum = sum(trust_matrix[node, other] for other in network if other != node)
            forward_trust[node] = trust_sum / (len(network) - 1)
        else:
            forward_trust[node] = 0.0
            
    return forward_trust

def compute_trust_backward(trust_matrix, network):
    """
    Compute r_B(i) for each node i in network
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix
    network : list
        List of nodes in the network
        
    Returns:
    --------
    backward_trust : dict
        Dictionary mapping node to the average trust other nodes have in it
    """
    backward_trust = {}
    
    for node in network:
        if len(network) > 1:
            # Calculate average trust from other nodes in network to node
            trust_sum = sum(trust_matrix[other, node] for other in network if other != node)
            backward_trust[node] = trust_sum / (len(network) - 1)
        else:
            backward_trust[node] = 0.0
            
    return backward_trust

def perform_quantum_trust_partitioning(trust_matrix, networks=None, max_iterations=10, 
                                      threshold=0.5, stability_threshold=0.01,
                                      use_topology=True, save_history=False, verbose=False):
    """
    Perform quantum trust partitioning with topological features
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
    networks : list, optional
        Initial networks, each being a list of node indices
    max_iterations : int, default=10
        Maximum number of iterations for partitioning
    threshold : float, default=0.5
        Trust threshold for abandoning network
    stability_threshold : float, default=0.01
        Threshold for network stability (stopping criterion)
    use_topology : bool, default=True
        Whether to use topological features for partitioning
    save_history : bool, default=False
        Whether to save history of networks and trust matrices
    verbose : bool, default=False
        Whether to print progress information
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - final_networks: Final network partitioning
        - networks_history: History of networks (if save_history=True)
        - trust_history: History of trust matrices (if save_history=True)
        - betti_history: History of Betti numbers (if save_history=True)
        - n_iterations: Number of iterations performed
        - converged: Whether partitioning converged
    """
    n_nodes = trust_matrix.shape[0]
    
    # Initialize networks if not provided
    if networks is None:
        n_networks = max(2, n_nodes // 5)  # Default to 1 network per 5 nodes, at least 2
        networks = distribute_nodes_to_networks(n_nodes, n_networks)
    
    # Setup for history tracking
    networks_history = []
    trust_history = []
    betti_history = []
    
    if save_history:
        networks_history.append(networks.copy())
        trust_history.append(trust_matrix.copy())
    
    # Initial Betti numbers computation if using topology
    if use_topology:
        all_betti_numbers = compute_all_networks_betti_numbers(networks, trust_matrix)
        if save_history:
            betti_history.append(all_betti_numbers)
    
    # Main partitioning loop
    iteration = 0
    converged = False
    jump_events = []
    abandon_events = []
    
    while iteration < max_iterations and not converged:
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Keep copies for comparison
        old_networks = [network.copy() for network in networks]
        
        # Adjust trust matrix with topological features
        if use_topology:
            trust_matrix = adjust_trust_with_topology(trust_matrix, all_betti_numbers, networks)
        
        # Perform jump step
        networks, iteration_jump_events = jump_step(networks, trust_matrix)
        jump_events.extend(iteration_jump_events)
        
        # Perform abandon step
        networks, iteration_abandon_events = abandon_step(networks, trust_matrix, threshold)
        abandon_events.extend(iteration_abandon_events)
        
        # Clean up empty networks
        networks = [network for network in networks if len(network) > 0]
        
        # Update history
        if save_history:
            networks_history.append([network.copy() for network in networks])
            trust_history.append(trust_matrix.copy())
        
        # Recompute Betti numbers if using topology
        if use_topology:
            all_betti_numbers = compute_all_networks_betti_numbers(networks, trust_matrix)
            if save_history:
                betti_history.append(all_betti_numbers)
        
        # Check for convergence
        if networks_stable(old_networks, networks, stability_threshold):
            converged = True
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
        
        iteration += 1
    
    # Prepare result dictionary
    result = {
        'final_networks': networks,
        'n_iterations': iteration,
        'converged': converged,
        'jump_events': jump_events,
        'abandon_events': abandon_events
    }
    
    if save_history:
        result['networks_history'] = networks_history
        result['trust_history'] = trust_history
        if use_topology:
            result['betti_history'] = betti_history
    
    return result

def jump_step(networks, trust_matrix):
    """
    Perform the jump step of quantum trust partitioning
    
    Parameters:
    -----------
    networks : list
        List of networks, where each network is a list of node IDs
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
        
    Returns:
    --------
    new_networks : list
        Updated networks after jump step
    jump_events : list
        List of jump events (node, from_network, to_network)
    """
    n_networks = len(networks)
    new_networks = [network.copy() for network in networks]
    jump_events = []
    
    # Process each network
    for from_idx, network in enumerate(networks):
        for node in network:
            best_network = from_idx
            best_trust = compute_node_average_distrust(node, network, trust_matrix)
            
            # Consider all other networks
            for to_idx in range(n_networks):
                if to_idx == from_idx:
                    continue
                
                # Calculate average distrust if node joins this network
                target_network = networks[to_idx] + [node]
                node_distrust = compute_node_average_distrust(node, target_network, trust_matrix)
                
                # Apply quantum randomness in decision
                if node_distrust < best_trust or (
                    abs(node_distrust - best_trust) < 0.1 and 
                    generate_quantum_random_number() % 2 == 0):
                    
                    best_trust = node_distrust
                    best_network = to_idx
            
            # Jump to the best network if different from current
            if best_network != from_idx:
                # Remove from current network
                new_networks[from_idx].remove(node)
                
                # Add to new network
                new_networks[best_network].append(node)
                
                # Record jump event
                jump_events.append((node, from_idx, best_network))
    
    return new_networks, jump_events

def abandon_step(networks, trust_matrix, threshold=0.5):
    """
    Perform the abandon step of quantum trust partitioning
    
    Parameters:
    -----------
    networks : list
        List of networks, where each network is a list of node IDs
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
    threshold : float, default=0.5
        Trust threshold for abandoning network
        
    Returns:
    --------
    new_networks : list
        Updated networks after abandon step
    abandon_events : list
        List of abandon events (node, from_network)
    """
    new_networks = [network.copy() for network in networks]
    singleton_networks = []
    abandon_events = []
    
    # Process each network
    for network_idx, network in enumerate(networks):
        nodes_to_remove = []
        
        for node in network:
            # Calculate node's forward trust in its network
            forward_trust = compute_trust_forward(trust_matrix, network).get(node, 0)
            
            # Apply quantum randomness in threshold
            quantum_threshold = threshold * (1.0 + 0.1 * (generate_quantum_random_number() / 255.0 - 0.5))
            
            # Check if trust is below threshold
            if forward_trust < quantum_threshold:
                nodes_to_remove.append(node)
                
                # Create singleton network for this node
                singleton_networks.append([node])
                
                # Record abandon event
                abandon_events.append((node, network_idx))
        
        # Remove nodes that abandoned
        for node in nodes_to_remove:
            new_networks[network_idx].remove(node)
    
    # Add singleton networks
    new_networks.extend(singleton_networks)
    
    return new_networks, abandon_events

def compute_node_average_distrust(node, network, trust_matrix):
    """
    Compute the average distrust for a node in a network
    
    Parameters:
    -----------
    node : int
        Node ID
    network : list
        List of node IDs in the network
    trust_matrix : numpy.ndarray
        Trust matrix where [i,j] represents trust from node i to node j
        
    Returns:
    --------
    avg_distrust : float
        Average distrust for the node in the network
    """
    # Node's distrust is 1 - trust
    if len(network) <= 1:
        return 0.0  # No other nodes to trust/distrust
    
    distrust_sum = sum(1 - trust_matrix[node, other] for other in network if other != node)
    return distrust_sum / (len(network) - 1)

def networks_stable(old_networks, new_networks, threshold=0.01):
    """
    Check if networks have stabilized (converged)
    
    Parameters:
    -----------
    old_networks : list
        List of networks before iteration
    new_networks : list
        List of networks after iteration
    threshold : float, default=0.01
        Threshold for stability
        
    Returns:
    --------
    stable : bool
        Whether networks are stable
    """
    # If the number of networks changed, not stable
    if len(old_networks) != len(new_networks):
        return False
    
    # Count number of changed nodes
    total_nodes = sum(len(network) for network in old_networks)
    if total_nodes == 0:
        return True
    
    # Create sets for each network to compare membership
    old_sets = [set(network) for network in old_networks]
    new_sets = [set(network) for network in new_networks]
    
    # Compute best matching between old and new networks
    # This is a simple greedy matching for efficiency
    matched_new_sets = set()
    changed_nodes = 0
    
    for old_set in old_sets:
        best_match = None
        best_intersection = -1
        
        for i, new_set in enumerate(new_sets):
            if i in matched_new_sets:
                continue
                
            intersection = len(old_set.intersection(new_set))
            if intersection > best_intersection:
                best_intersection = intersection
                best_match = i
        
        if best_match is not None:
            matched_new_sets.add(best_match)
            # Count changed nodes (symmetric difference)
            changed_nodes += len(old_set.symmetric_difference(new_sets[best_match]))
        else:
            # No match found, all nodes changed
            changed_nodes += len(old_set)
    
    # Check if the ratio of changed nodes is below threshold
    return changed_nodes / total_nodes <= threshold

#########################
# Stability Analysis
#########################

def analyze_partitioning_stability(trust_matrix, perturbation_levels, max_iterations=5,
                                  use_topology=True, verbose=False):
    """
    Analyze the stability of quantum trust partitioning under perturbations
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Base trust matrix
    perturbation_levels : list
        List of perturbation levels to test
    max_iterations : int, default=5
        Maximum number of iterations for each partitioning
    use_topology : bool, default=True
        Whether to use topological features for partitioning
    verbose : bool, default=False
        Whether to print progress information
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - trust_matrices: List of perturbed trust matrices
        - networks_history: History of networks for each perturbation level
        - betti_histories: History of Betti numbers for each perturbation level
        - perturbation_levels: Levels of perturbation used
    """
    # Setup result storage
    trust_matrices = []
    networks_history = []
    betti_histories = []
    
    # Test each perturbation level
    for level in perturbation_levels:
        if verbose:
            print(f"Testing perturbation level: {level:.2f}")
        
        # Create perturbed trust matrix
        perturbed_matrix = perturb_trust_matrix(trust_matrix, level)
        trust_matrices.append(perturbed_matrix)
        
        # Run partitioning on perturbed matrix
        result = perform_quantum_trust_partitioning(
            perturbed_matrix,
            max_iterations=max_iterations,
            use_topology=use_topology,
            save_history=True,
            verbose=verbose
        )
        
        # Store results
        networks_history.append(result['networks_history'])
        if 'betti_history' in result:
            betti_histories.append(result['betti_history'])
        
    # Prepare result dictionary
    results = {
        'trust_matrices': trust_matrices,
        'networks_history': networks_history,
        'perturbation_levels': perturbation_levels
    }
    
    if betti_histories:
        results['betti_histories'] = betti_histories
    
    return results

def perturb_trust_matrix(trust_matrix, perturbation_level):
    """
    Perturb a trust matrix by adding random noise
    
    Parameters:
    -----------
    trust_matrix : numpy.ndarray
        Trust matrix to perturb
    perturbation_level : float
        Level of perturbation (0.0 to 1.0)
        
    Returns:
    --------
    perturbed_matrix : numpy.ndarray
        Perturbed trust matrix
    """
    n_nodes = trust_matrix.shape[0]
    
    # Create random perturbation
    quantum_perturbation = np.zeros_like(trust_matrix)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # Don't perturb self-trust
                # Mix classical and quantum randomness
                random_val = np.random.random() * 0.5 + generate_quantum_random_number() / 255.0 * 0.5
                quantum_perturbation[i, j] = (random_val * 2 - 1) * perturbation_level
    
    # Apply perturbation
    perturbed_matrix = trust_matrix + quantum_perturbation
    
    # Ensure values are in [0, 1]
    perturbed_matrix = np.clip(perturbed_matrix, 0, 1)
    
    # Ensure diagonal is 1
    np.fill_diagonal(perturbed_matrix, 1)
    
    return perturbed_matrix

#########################
# Utility Functions
#########################

def get_quantum_partition_statistics(result):
    """
    Calculate statistics for a quantum trust partitioning result
    
    Parameters:
    -----------
    result : dict
        Result from perform_quantum_trust_partitioning
        
    Returns:
    --------
    stats : dict
        Dictionary with statistics
    """
    networks = result['final_networks']
    trust_matrix = result['trust_history'][-1] if 'trust_history' in result else None
    
    # Basic statistics
    n_networks = len(networks)
    network_sizes = [len(network) for network in networks]
    
    stats = {
        'n_networks': n_networks,
        'network_sizes': network_sizes,
        'avg_network_size': sum(network_sizes) / n_networks if n_networks > 0 else 0,
        'max_network_size': max(network_sizes) if network_sizes else 0,
        'min_network_size': min(network_sizes) if network_sizes else 0
    }
    
    # Trust statistics if available
    if trust_matrix is not None:
        # Calculate average trust within and between networks
        trust_within = []
        trust_between = []
        
        for i, network1 in enumerate(networks):
            # Within network trust
            for a in network1:
                for b in network1:
                    if a != b:
                        trust_within.append(trust_matrix[a, b])
            
            # Between network trust
            for j, network2 in enumerate(networks):
                if i != j:
                    for a in network1:
                        for b in network2:
                            trust_between.append(trust_matrix[a, b])
        
        # Compute statistics
        stats['avg_trust_within'] = np.mean(trust_within) if trust_within else 0
        stats['avg_trust_between'] = np.mean(trust_between) if trust_between else 0
        stats['trust_ratio'] = (stats['avg_trust_within'] / stats['avg_trust_between'] 
                              if stats['avg_trust_between'] > 0 else float('inf'))
    
    return stats

def create_quantum_trust_partitioning_demo(n_nodes=20, n_networks=4, trust_type='small_world',
                                         max_iterations=10, with_topological_features=True,
                                         output_prefix="quantum_topological_partitioning",
                                         dark_mode=False):
    """
    Create a comprehensive demonstration of quantum trust partitioning
    
    Parameters:
    -----------
    n_nodes : int, default=20
        Number of nodes in the network
    n_networks : int, default=4
        Initial number of networks
    trust_type : str, default='small_world'
        Type of trust matrix to generate
    max_iterations : int, default=10
        Maximum number of iterations
    with_topological_features : bool, default=True
        Whether to use topological features
    output_prefix : str, default="quantum_topological_partitioning"
        Prefix for output files
    dark_mode : bool, default=False
        Whether to use dark mode for visualizations
        
    Returns:
    --------
    result : dict
        Result of the partitioning
    """
    # Import here to avoid circular imports
    from quantum_topological_visualizer import (
        visualize_quantum_network_with_topology,
        create_quantum_topology_animation
    )
    
    # Generate trust matrix
    trust_matrix = create_initial_quantum_trust_matrix(n_nodes, trust_type)
    
    # Generate initial networks
    initial_networks = distribute_nodes_to_networks(n_nodes, n_networks)
    
    # Run partitioning
    result = perform_quantum_trust_partitioning(
        trust_matrix,
        networks=initial_networks,
        max_iterations=max_iterations,
        use_topology=with_topological_features,
        save_history=True,
        verbose=True
    )
    
    # Visualize results
    # Create visualization of initial state
    G_initial = nx.Graph()
    for network_idx, network in enumerate(initial_networks):
        for node in network:
            G_initial.add_node(node, network=network_idx)
    
    initial_pos = nx.spring_layout(G_initial, seed=42)
    
    visualize_quantum_network_with_topology(
        G_initial, initial_pos, result['betti_history'][0], 
        title="Initial Quantum Network State",
        save_path=f"{output_prefix}_initial.png",
        dark_mode=dark_mode
    )
    
    # Create visualization of final state
    G_final = nx.Graph()
    for network_idx, network in enumerate(result['final_networks']):
        for node in network:
            G_final.add_node(node, network=network_idx)
    
    final_pos = nx.spring_layout(G_final, seed=42)
    
    visualize_quantum_network_with_topology(
        G_final, final_pos, result['betti_history'][-1], 
        title="Final Quantum Network State",
        save_path=f"{output_prefix}_final.png",
        dark_mode=dark_mode
    )
    
    # Create animation of network evolution
    create_quantum_topology_animation(
        result['networks_history'],
        result['trust_history'],
        {},  # Security thresholds are not used in this demo
        np.linspace(0, 1, 10),  # Filtration values
        result['betti_history'],
        title="Quantum Topological Network Evolution",
        output_path=f"{output_prefix}_evolution.gif",
        fps=2,
        dark_mode=dark_mode
    )
    
    return result

if __name__ == "__main__":
    # Simple demonstration if run directly
    print("Quantum Trust Partitioning with Topological Features")
    print("Running a simple demonstration...")
    
    # Create a small demonstration
    result = create_quantum_trust_partitioning_demo(
        n_nodes=15,
        n_networks=3,
        max_iterations=5,
        output_prefix="quantum_topological_demo"
    )
    
    # Print some statistics
    stats = get_quantum_partition_statistics(result)
    print("\nPartitioning complete!")
    print(f"Final number of networks: {stats['n_networks']}")
    for i, size in enumerate(stats['network_sizes']):
        print(f"  Network {i+1}: {size} nodes")
    print(f"Average trust within networks: {stats['avg_trust_within']:.4f}")
    print(f"Average trust between networks: {stats['avg_trust_between']:.4f}")
    print(f"Trust ratio (within/between): {stats['trust_ratio']:.4f}")
    
    print("\nVisualization files saved with prefix 'quantum_topological_demo'")

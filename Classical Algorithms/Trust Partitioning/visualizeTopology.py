#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizeTopology.py

This script demonstrates topological analysis and visualization using the topologicalPartitioner
module. It creates network examples, analyzes their topological properties using persistent
homology, and generates visualizations of the results.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from topologicalPartitioner import (
    compute_betti_numbers, 
    compute_persistent_homology_sequence,
    construct_distance_matrix,
    extract_topological_features,
    plot_persistence_diagram,
    plot_betti_curves,
    visualize_network_with_topology,
    create_topological_partitioning_animation,
    topological_partitioner
)

# Set output directory to the current folder (Trust Partitioning)
output_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure the directory exists (it should already)
os.makedirs(output_dir, exist_ok=True)

def generate_example_networks(n_types=3, n_nodes_per_network=15, noise_level=0.1):
    """
    Generate different types of networks with varying topological properties.
    More complex network structures with less spread and more interesting topology.
    
    Parameters:
    -----------
    n_types : int
        Number of different network types to generate
    n_nodes_per_network : int
        Approximate number of nodes per network
    noise_level : float
        Level of noise to add to trust values
    
    Returns:
    --------
    dict : Dictionary with networks, trust matrix, node security, and network security
    """
    # Initialize networks container
    networks = []
    node_counter = 0
    
    # Network type 1: Dense community with core-periphery structure
    n_nodes1 = n_nodes_per_network
    network1 = list(range(node_counter, node_counter + n_nodes1))
    networks.append(network1)
    node_counter += n_nodes1
    
    # Network type 2: Mesh/Grid structure with cycles
    n_nodes2 = n_nodes_per_network
    network2 = list(range(node_counter, node_counter + n_nodes2))
    networks.append(network2)
    node_counter += n_nodes2
    
    # Network type 3: Hierarchical structure with multiple connected components
    if n_types > 2:
        n_nodes3 = n_nodes_per_network
        network3 = list(range(node_counter, node_counter + n_nodes3))
        networks.append(network3)
        node_counter += n_nodes3
    
    # Total number of nodes
    n_total = node_counter
    
    # Initialize trust matrix with random noise
    trust_matrix = np.random.rand(n_total, n_total) * noise_level
    np.fill_diagonal(trust_matrix, 0)  # No trust to self
    
    # Create more complex network structures
    
    # Network 1: Dense community with core-periphery structure
    # First identify core nodes (about 1/3 of the network)
    core_size = max(3, n_nodes1 // 3)
    core_nodes = network1[:core_size]
    periphery_nodes = network1[core_size:]
    
    # Create dense connections among core nodes
    for i in core_nodes:
        for j in core_nodes:
            if i != j:
                trust_matrix[i, j] = 0.8 + 0.2 * np.random.rand()
    
    # Create connections from periphery to core (hub-and-spoke pattern)
    for idx, i in enumerate(periphery_nodes):
        # Connect to multiple core nodes with varying strength
        for core_idx, j in enumerate(core_nodes):
            # Varying strength to create more interesting patterns
            if core_idx < 2 or np.random.rand() < 0.4:  # Connect to at least 2 core nodes
                strength = 0.6 + 0.4 * np.random.rand()
                trust_matrix[i, j] = strength
                trust_matrix[j, i] = strength * (0.7 + 0.3 * np.random.rand())  # Asymmetric trust
    
    # Network 2: Mesh/Grid structure
    # Create a grid-like structure to generate interesting cycles (high β₁)
    grid_dim = int(np.sqrt(n_nodes2))
    for idx, i in enumerate(network2):
        row, col = idx // grid_dim, idx % grid_dim
        
        # Connect to right neighbor
        if col < grid_dim - 1 and idx + 1 < n_nodes2:
            next_node = network2[idx + 1]
            strength = 0.7 + 0.3 * np.random.rand()
            trust_matrix[i, next_node] = strength
            trust_matrix[next_node, i] = strength
        
        # Connect to bottom neighbor
        if row < grid_dim - 1 and idx + grid_dim < n_nodes2:
            below_node = network2[idx + grid_dim]
            strength = 0.7 + 0.3 * np.random.rand()
            trust_matrix[i, below_node] = strength
            trust_matrix[below_node, i] = strength
        
        # Add some diagonal connections to create more complex topology
        if (row < grid_dim - 1 and col < grid_dim - 1 and 
            idx + grid_dim + 1 < n_nodes2 and np.random.rand() < 0.4):
            diag_node = network2[idx + grid_dim + 1]
            strength = 0.5 + 0.5 * np.random.rand()
            trust_matrix[i, diag_node] = strength
            trust_matrix[diag_node, i] = strength
    
    # Network 3: Hierarchical structure with multiple connected components
    if n_types > 2:
        # Create several tightly connected communities within the network
        community_size = max(3, n_nodes3 // 4)
        for start_idx in range(0, n_nodes3, community_size):
            end_idx = min(start_idx + community_size, n_nodes3)
            community = network3[start_idx:end_idx]
            
            # Create dense connections within each community
            for i in community:
                for j in community:
                    if i != j:
                        trust_matrix[i, j] = 0.7 + 0.3 * np.random.rand()
            
            # If not the last community, create some connections to the next community
            if end_idx < n_nodes3:
                next_community_end = min(end_idx + community_size, n_nodes3)
                next_community = network3[end_idx:next_community_end]
                
                # Create a few bridge connections
                bridge_count = max(1, len(community) // 3)
                for _ in range(bridge_count):
                    i = np.random.choice(community)
                    j = np.random.choice(next_community)
                    strength = 0.4 + 0.4 * np.random.rand()  # Weaker connections between communities
                    trust_matrix[i, j] = strength
                    trust_matrix[j, i] = strength
    
    # Define security parameters
    node_security = {i: np.random.uniform(0.1, 0.4) for i in range(n_total)}
    
    # Define network security parameters
    network_security = {}
    for network_idx, network in enumerate(networks):
        # Get security values for nodes in this network
        security_values = [node_security[node] for node in network]
        # Use most lenient security among top n/2 nodes
        sorted_security = sorted(security_values)
        most_lenient = np.max(sorted_security[len(sorted_security) // 2:])
        network_security[network_idx] = most_lenient
    
    return {
        'networks': networks,
        'trust_matrix': trust_matrix,
        'node_security': node_security,
        'network_security': network_security
    }

def analyze_and_visualize_topology(example_data):
    """
    Perform topological analysis and generate visualizations
    
    Parameters:
    -----------
    example_data : dict
        Dictionary with networks, trust matrix, and security parameters
    """
    try:
        networks = example_data['networks']
        trust_matrix = example_data['trust_matrix']
        node_security = example_data['node_security']
        network_security = example_data['network_security']
        
        # Convert to distance matrix
        distance_matrix = construct_distance_matrix(trust_matrix)
        
        # Compute persistent homology with multiple filtration values
        print("Computing persistent homology...")
        ph_sequence = compute_persistent_homology_sequence(distance_matrix, max_dimension=2, filtration_steps=20)
        
        # Compute Betti numbers at a specific filtration value
        filtration_value = 0.5  # Adjust as needed
        betti_result = compute_betti_numbers(distance_matrix, max_dimension=2, max_edge_length=filtration_value)
        
        # Extract topological features
        features = extract_topological_features(distance_matrix, max_dimension=2)
        
        # Create a network graph for visualization
        G = nx.Graph()
        
        # Add all nodes
        all_nodes = [node for network in networks for node in network]
        G.add_nodes_from(all_nodes)
        
        # Add weighted edges based on trust matrix
        n = len(all_nodes)
        for i in range(n):
            for j in range(i+1, n):
                # Average the trust in both directions
                trust_val = (trust_matrix[i, j] + trust_matrix[j, i]) / 2
                if trust_val > 0.3:  # Only add edges with significant trust
                    G.add_edge(all_nodes[i], all_nodes[j], weight=trust_val)
          # Calculate position layout with tighter spacing (k=0.4 reduces spread)
        pos = nx.spring_layout(G, seed=42, k=0.4)
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # 1. Plot persistence diagram
        persistence_diagram = plot_persistence_diagram(betti_result['persistence'], max_dimension=2, 
                                                    title="Persistence Diagram of Network Trust")
        persistence_diagram.savefig(os.path.join(output_dir, "persistence_diagram.png"), dpi=300, bbox_inches='tight')
        plt.close(persistence_diagram)
        
        # 2. Plot Betti curves
        betti_curves = plot_betti_curves(ph_sequence['betti_sequences'], ph_sequence['filtration_values'], 
                                        title="Betti Curves of Network Trust")
        betti_curves.savefig(os.path.join(output_dir, "betti_curves.png"), dpi=300, bbox_inches='tight')
        plt.close(betti_curves)
        
        # 3. Visualize network with topological features
        if len(G.nodes()) > 0:
            try:
                network_viz = visualize_network_with_topology(G, pos, features, 
                                                           title="Network Trust with Topological Features")
                network_viz.savefig(os.path.join(output_dir, "network_topology.png"), dpi=300, bbox_inches='tight')
                plt.close(network_viz)
            except Exception as e:
                print(f"Warning: Could not create network topology visualization: {e}")
        else:
            print("Warning: Graph has no nodes, skipping network visualization")
        
        # 4. Generate a combined visualization showing subnetworks
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw each subnetwork with different colors
        for i, network in enumerate(networks):
            if network:  # Only process non-empty networks
                subG = G.subgraph(network)
                color = plt.cm.tab10(i % 10)
                nx.draw_networkx_nodes(subG, pos, node_color=[color]*len(subG), 
                                     node_size=100, label=f"Network {i+1}", alpha=0.8, ax=ax)
                
                # Draw edges within this subnetwork
                if subG.edges():
                    nx.draw_networkx_edges(subG, pos, edge_color=color, alpha=0.5, width=1.5, ax=ax)
        
        # Draw node labels
        if G.nodes():
            nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)
        
        ax.set_title(f"Network Communities with Topological Features\nBetti Numbers: β₀={features['betti0']}, " +
                  f"β₁={features['betti1']}, β₂={features['betti2']}")
        if networks:
            ax.legend()
        ax.set_axis_off()
        plt.savefig(os.path.join(output_dir, "network_communities.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization files saved to {output_dir}")
        
        return {
            'G': G, 
            'pos': pos, 
            'betti_result': betti_result, 
            'ph_sequence': ph_sequence,
            'features': features
        }
    except Exception as e:
        print(f"Error in topology analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_topological_partitioning(example_data, topo_weight=0.5):
    """
    Run the topological partitioning algorithm and visualize results
    
    Parameters:
    -----------
    example_data : dict
        Dictionary with networks, trust matrix, and security parameters
    topo_weight : float
        Weight given to topological features vs trust (0.0-1.0)
    """
    try:
        networks = example_data['networks']
        trust_matrix = example_data['trust_matrix']
        node_security = example_data['node_security']
        network_security = example_data['network_security']
        
        # Run the topological partitioner
        print(f"Running topological partitioning with topo_weight={topo_weight}...")
        new_networks, _, _, _, topo_results = topological_partitioner(
            networks, trust_matrix, node_security, network_security, 
            max_dimension=2, topo_weight=topo_weight
        )
          # Create a visualization of the partitioning with intermediate states
        # Generate intermediate network states
        intermediate_networks = generate_intermediate_network_states(networks, new_networks, 3)  # 3 intermediate steps
        network_evolution = [networks] + intermediate_networks + [new_networks]
        
        # Create animation
        print("Generating partitioning animation...")
        try:
            _ = create_topological_partitioning_animation(
                networks, trust_matrix, network_evolution, topo_results, 
                filename=f"topological_partitioning_w{int(topo_weight*10)}.gif"
            )
        except Exception as e:
            print(f"Warning: Could not create animation: {e}")
            
        # Create a variation with more complex Betti curves
        if topo_weight == 0.5:  # Only create variation for the default weight
            print("Generating variation with non-trivial Betti curves...")
            try:
                complex_networks, complex_trust = generate_complex_topology_networks(networks, trust_matrix)
                # Run topological partitioning on the complex networks
                complex_new_networks, _, _, _, complex_topo_results = topological_partitioner(
                    complex_networks, complex_trust, node_security, network_security, 
                    max_dimension=2, topo_weight=topo_weight
                )
                # Generate intermediate states
                complex_intermediates = generate_intermediate_network_states(complex_networks, complex_new_networks, 3)
                complex_evolution = [complex_networks] + complex_intermediates + [complex_new_networks]
                  # Create animation with complex topology
                _ = create_topological_partitioning_animation(
                    complex_networks, complex_trust, complex_evolution, complex_topo_results, 
                    filename=f"topological_partitioning_complex_w{int(topo_weight*10)}.gif",
                    colormap='viridis'  # Use viridis colormap for complex networks
                )
            except Exception as e:
                print(f"Warning: Could not create complex animation: {e}")
        
        
        # Print summary of changes in network structure
        print("\nNetwork Structure Changes:")
        print("-------------------------")
        print(f"Original networks: {[len(net) for net in networks]}")
        print(f"New networks: {[len(net) for net in new_networks]}")
        
        # Analyze topological changes
        print("\nTopological Analysis Summary:")
        print("----------------------------")
        global_features = topo_results["global_features"]
        print(f"Global Network: β₀={global_features['betti0']}, β₁={global_features['betti1']}, β₂={global_features['betti2']}")
        
        for network_idx, features in topo_results["original_network_features"].items():
            network_name = chr(65 + network_idx)  # Convert 0->A, 1->B, etc.
            print(f"Network {network_name}: β₀={features['betti0']}, β₁={features['betti1']}, β₂={features['betti2']}")
        
        return new_networks, topo_results
        
    except Exception as e:
        print(f"Error in topological partitioning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def run_comparison_study(example_data):
    """
    Run a comparison study with different weightings for topological features
    
    Parameters:
    -----------
    example_data : dict
        Dictionary with networks, trust matrix, and security parameters
    """
    # Different weightings to test
    topo_weights = [0.0, 0.3, 0.7, 1.0]
    
    results = {}
    
    for weight in topo_weights:
        print(f"\n=== Running with topology weight = {weight} ===")
        new_networks, topo_results = run_topological_partitioning(example_data, topo_weight=weight)
        results[weight] = {'networks': new_networks, 'topo_results': topo_results}
    
    # Compare results
    print("\nComparison of Results with Different Topology Weights:")
    print("----------------------------------------------------")
    
    for weight in topo_weights:
        print(f"\nWith topology weight = {weight}:")
        print(f"  Resulting networks: {[len(net) for net in results[weight]['networks']]}")
    
    return results

# Helper functions for creating intermediate network states and complex topology networks

def generate_intermediate_network_states(initial_networks, final_networks, num_intermediate_steps=2):
    """
    Generate intermediate network states between initial and final configurations.
    
    Parameters:
    -----------
    initial_networks : list
        List of initial networks
    final_networks : list
        List of final networks
    num_intermediate_steps : int
        Number of intermediate steps to generate
        
    Returns:
    --------
    list : List of intermediate network states
    """
    intermediate_states = []
    
    # Find all nodes that move between networks
    all_initial_nodes = set()
    initial_node_to_network = {}
    for i, network in enumerate(initial_networks):
        for node in network:
            all_initial_nodes.add(node)
            initial_node_to_network[node] = i
    
    # Track which network each node ends up in
    final_node_to_network = {}
    for i, network in enumerate(final_networks):
        for node in network:
            if node in all_initial_nodes:
                final_node_to_network[node] = i
    
    # Identify nodes that move between networks
    moving_nodes = {}
    for node in all_initial_nodes:
        initial_network = initial_node_to_network.get(node, -1)
        final_network = final_node_to_network.get(node, -1)
        if initial_network != final_network:
            moving_nodes[node] = (initial_network, final_network)
    
    # For each intermediate step, move a subset of the nodes
    for step in range(num_intermediate_steps):
        # Determine which nodes to move in this step
        step_progress = (step + 1) / (num_intermediate_steps + 1)
        nodes_to_move_count = int(len(moving_nodes) * step_progress)
        
        # Create a copy of the initial networks
        current_state = [list(network) for network in initial_networks]
        
        # Create any new networks that appear in the final state
        while len(current_state) < len(final_networks):
            current_state.append([])
        
        # Move a subset of nodes
        moved_count = 0
        for node, (from_network, to_network) in moving_nodes.items():
            if moved_count >= nodes_to_move_count:
                break
                
            # Skip if the node is not in the expected network
            if from_network < 0 or from_network >= len(current_state):
                continue
                
            if node in current_state[from_network]:
                # Move the node to its destination
                current_state[from_network].remove(node)
                
                # Ensure the destination network exists
                while len(current_state) <= to_network:
                    current_state.append([])
                    
                # Add the node to its new network
                if to_network >= 0:
                    current_state[to_network].append(node)
                
                moved_count += 1
        
        # Add the current state to our intermediate states
        intermediate_states.append(current_state)
    
    return intermediate_states

def generate_complex_topology_networks(base_networks, base_trust, add_cycles=True, add_voids=True):
    """
    Generate networks with more complex topological features (higher Betti numbers).
    This creates completely different network structures compared to the standard visualization.
    
    Parameters:
    -----------
    base_networks : list
        List of base networks
    base_trust : np.array
        Base trust matrix
    add_cycles : bool
        Whether to add cycles (Betti-1)
    add_voids : bool
        Whether to add voids (Betti-2)
        
    Returns:
    --------
    tuple : (complex_networks, complex_trust)
    """
    import numpy as np
    import copy
    
    # Create completely different networks instead of copying base networks
    # Get total number of nodes from base networks
    total_nodes = sum(len(network) for network in base_networks)
    
    # Create new network structure with different organization - fewer larger networks
    num_complex_networks = min(3, len(base_networks))  # Use at most 3 networks
    complex_networks = [[] for _ in range(num_complex_networks)]
    
    # Distribute nodes among networks in a different pattern
    node_counter = 0
    for i in range(total_nodes):
        # Assign node to a network based on a mathematical pattern different from base networks
        # This creates a completely different network structure
        network_idx = i % num_complex_networks
        complex_networks[network_idx].append(node_counter)
        node_counter += 1
    
    # Initialize new trust matrix
    complex_trust = np.zeros_like(base_trust)
    
    # Add base random noise
    noise_level = 0.05
    complex_trust = np.random.rand(*complex_trust.shape) * noise_level
    np.fill_diagonal(complex_trust, 0)  # No trust to self
    
    # Create complex structures in each network
    
    # First network: Create a densely connected core with branching periphery
    if len(complex_networks[0]) >= 6:
        # Create a dense core
        core_nodes = complex_networks[0][:4]
        peripheral_nodes = complex_networks[0][4:]
        
        # Connect core nodes strongly
        for i in range(len(core_nodes)):
            for j in range(i+1, len(core_nodes)):
                # High trust between core nodes
                complex_trust[core_nodes[i], core_nodes[j]] = 0.85 + 0.15 * np.random.rand()
                complex_trust[core_nodes[j], core_nodes[i]] = 0.85 + 0.15 * np.random.rand()
        
        # Connect peripheral nodes to core in a star-like pattern
        for i, node in enumerate(peripheral_nodes):
            # Connect to a single core node
            core_connection = core_nodes[i % len(core_nodes)]
            complex_trust[node, core_connection] = 0.7 + 0.3 * np.random.rand()
            complex_trust[core_connection, node] = 0.7 + 0.3 * np.random.rand()
    
    # Second network: Create a mesh/lattice structure with cycles
    if len(complex_networks) > 1 and len(complex_networks[1]) >= 6:
        lattice_size = int(np.sqrt(len(complex_networks[1])))
        lattice_size = max(2, lattice_size)  # Ensure at least 2x2
        
        for i in range(len(complex_networks[1])):
            row, col = i // lattice_size, i % lattice_size
            node = complex_networks[1][i]
            
            # Connect to right neighbor
            if col < lattice_size - 1 and i + 1 < len(complex_networks[1]):
                right_node = complex_networks[1][i + 1]
                complex_trust[node, right_node] = 0.8 + 0.2 * np.random.rand()
                complex_trust[right_node, node] = 0.8 + 0.2 * np.random.rand()
            
            # Connect to bottom neighbor
            if row < lattice_size - 1 and i + lattice_size < len(complex_networks[1]):
                bottom_node = complex_networks[1][i + lattice_size]
                complex_trust[node, bottom_node] = 0.8 + 0.2 * np.random.rand()
                complex_trust[bottom_node, node] = 0.8 + 0.2 * np.random.rand()
    
    # Third network: Create a high-dimensional structure (tetrahedra/voids)
    if len(complex_networks) > 2 and len(complex_networks[2]) >= 5:
        # Create cliques of 5 nodes (4-simplices) that generate higher Betti numbers
        for i in range(0, len(complex_networks[2]), 5):
            if i + 5 <= len(complex_networks[2]):
                simplex_nodes = complex_networks[2][i:i+5]
                # Connect all nodes in the simplex with high trust
                for j in range(len(simplex_nodes)):
                    for k in range(j+1, len(simplex_nodes)):
                        complex_trust[simplex_nodes[j], simplex_nodes[k]] = 0.9
                        complex_trust[simplex_nodes[k], simplex_nodes[j]] = 0.9
    
    return complex_networks, complex_trust

if __name__ == "__main__":
    print("Topological Network Visualization")
    print("-------------------------------")
    
    try:
        # Generate example networks
        print("Generating example networks...")
        example_data = generate_example_networks(n_types=4, n_nodes_per_network=8)
        
        # Analyze and visualize topology
        print("Analyzing and visualizing topology...")
        viz_results = analyze_and_visualize_topology(example_data)
        
        print("Visualization complete, now running partitioning...")
        
        # Run the topological partitioning algorithm
        new_networks, topo_results = run_topological_partitioning(example_data, topo_weight=0.5)
        
        print("\nProcess complete. Check the topo_network_animations folder for results.")
    except Exception as e:
        import traceback
        print(f"Error encountered: {str(e)}")
        traceback.print_exc()
        print("\nProcess failed with errors.")

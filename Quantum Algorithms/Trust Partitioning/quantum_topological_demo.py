#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_topological_demo.py

This script demonstrates the complete quantum topological network partitioning system.
It combines quantum network partitioning with persistent homology and Betti numbers
to create a comprehensive, visual demonstration of the topological approach.

The script:
1. Initializes quantum networks with random trust values
2. Computes Betti numbers and persistent homology
3. Uses topological features to guide quantum network partitioning
4. Creates visualizations and animations showing the process and results
5. Evaluates the effectiveness of topology-guided partitioning

Author: Jeffrey Morais, BTQ
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
from tqdm import tqdm
import warnings

# Import quantum network partitioning functions
from quantumTrustPartitioner import (
    simulate_quantum_random_bit,
    generate_quantum_random_number,
    create_trust_matrix,
    visualize_network,
    initialize_nodes_and_networks,
    evaluate_network_integrity,
    distribute_nodes_across_networks
)

# Import quantum Betti number partitioning functions
from quantumBettiPartitioner import (
    compute_quantum_betti_numbers,
    compute_network_topological_features,
    partition_networks_with_topology,
    run_quantum_topology_partitioning
)

# Import visualization functions
from visualizeQuantumTopology import (
    visualize_simplicial_complex_3d,
    visualize_filtration_evolution,
    create_filtration_animation,
    visualize_betti_number_influence,
    create_betti_guided_partitioning_animation
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

#########################
# Demonstration Functions
#########################

def run_basic_demo():
    """Run a basic demonstration of quantum topological partitioning"""
    print("=" * 80)
    print("QUANTUM TOPOLOGICAL NETWORK PARTITIONING - BASIC DEMO")
    print("=" * 80)
    print("\nInitializing quantum networks...")
    
    # Initialize networks with random trust values
    n_nodes = 30
    n_networks = 3
    nodes, networks, node_positions, trust_matrices = initialize_nodes_and_networks(
        n_nodes=n_nodes, 
        n_networks=n_networks
    )
    
    print(f"Initialized {n_networks} networks with {n_nodes} total nodes")
    for i, network in enumerate(networks):
        print(f"  Network {i}: {len(network)} nodes")
    
    # Visualize initial networks
    print("\nVisualizing initial networks...")
    fig = visualize_network(networks, node_positions, trust_matrices)
    plt.savefig(os.path.join(output_dir, "quantum_topo_demo_initial.png"), dpi=300)
    plt.close(fig)
    
    # Compute topological features
    print("\nComputing topological features...")
    topo_features = compute_network_topological_features(
        networks, node_positions, trust_matrices
    )
    
    # Display Betti numbers for each network
    print("\nBetti numbers for initial networks:")
    for net_idx, network in enumerate(networks):
        if net_idx in topo_features:
            features = topo_features[net_idx]
            if 'betti_numbers' in features:
                betti_nums = features['betti_numbers']
                print(f"  Network {net_idx} ({len(network)} nodes):")
                print(f"    β₀={betti_nums.get('betti0', 0)}, β₁={betti_nums.get('betti1', 0)}, β₂={betti_nums.get('betti2', 0)}")
    
    # Run topology-guided partitioning
    print("\nRunning topology-guided partitioning...")
    new_networks, new_trust_matrices = partition_networks_with_topology(
        networks, 
        node_positions, 
        trust_matrices,
        betti_weight=0.3,
        visualize=True
    )
    
    print(f"\nPartitioning complete. {len(new_networks)} networks after partitioning:")
    for i, network in enumerate(new_networks):
        print(f"  Network {i}: {len(network)} nodes")
    
    # Compute new topological features
    new_topo_features = compute_network_topological_features(
        new_networks, node_positions, new_trust_matrices
    )
    
    # Display new Betti numbers
    print("\nBetti numbers after partitioning:")
    for net_idx, network in enumerate(new_networks):
        if net_idx in new_topo_features:
            features = new_topo_features[net_idx]
            if 'betti_numbers' in features:
                betti_nums = features['betti_numbers']
                print(f"  Network {net_idx} ({len(network)} nodes):")
                print(f"    β₀={betti_nums.get('betti0', 0)}, β₁={betti_nums.get('betti1', 0)}, β₂={betti_nums.get('betti2', 0)}")
    
    # Create animation of the partitioning process
    print("\nCreating animation of the partitioning process...")
    ani = create_betti_guided_partitioning_animation(
        networks,
        node_positions,
        trust_matrices,
        steps=5
    )
    ani.save(os.path.join(output_dir, "quantum_topo_demo_partitioning.gif"), 
            writer='pillow', fps=1, dpi=100)
    
    print("\nBasic demonstration complete! Visualizations saved to the output directory.")

def compare_partitioning_methods():
    """Compare standard quantum partitioning vs. topology-guided partitioning"""
    print("=" * 80)
    print("COMPARING QUANTUM PARTITIONING METHODS")
    print("=" * 80)
    
    # Initialize networks
    n_nodes = 40
    n_networks = 2
    nodes, networks, node_positions, trust_matrices = initialize_nodes_and_networks(
        n_nodes=n_nodes, 
        n_networks=n_networks
    )
    
    print(f"Initialized {n_networks} networks with {n_nodes} total nodes")
    
    # Create copies for different methods
    standard_networks = networks.copy()
    standard_trust_matrices = dict(trust_matrices)
    
    topo_networks = networks.copy()
    topo_trust_matrices = dict(trust_matrices)
    
    # Run standard quantum partitioning
    print("\nRunning standard quantum partitioning...")
    for _ in range(3):  # Run 3 rounds
        standard_networks, standard_trust_matrices = distribute_nodes_across_networks(
            standard_networks,
            node_positions,
            standard_trust_matrices
        )
    
    # Run topology-guided partitioning
    print("\nRunning topology-guided partitioning...")
    for _ in range(3):  # Run 3 rounds
        topo_networks, topo_trust_matrices = partition_networks_with_topology(
            topo_networks, 
            node_positions, 
            topo_trust_matrices,
            betti_weight=0.3,
            visualize=False
        )
    
    # Compare results
    print("\nComparison of results:")
    print(f"  Standard method: {len(standard_networks)} networks")
    for i, network in enumerate(standard_networks):
        print(f"    Network {i}: {len(network)} nodes")
    
    print(f"  Topology-guided method: {len(topo_networks)} networks")
    for i, network in enumerate(topo_networks):
        print(f"    Network {i}: {len(network)} nodes")
    
    # Compute topological features for both results
    standard_topo_features = compute_network_topological_features(
        standard_networks, node_positions, standard_trust_matrices
    )
    
    topo_method_features = compute_network_topological_features(
        topo_networks, node_positions, topo_trust_matrices
    )
    
    # Create visualization comparing the methods
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Standard method visualization
    ax1 = axes[0]
    for net_idx, network in enumerate(standard_networks):
        if not network:
            continue
        
        # Get positions
        net_pos = {node: node_positions[node] for node in network}
        
        # Draw nodes
        color = plt.cm.tab10(net_idx % 10)
        ax1.scatter(
            [net_pos[node][0] for node in network],
            [net_pos[node][1] for node in network],
            s=100, color=color, alpha=0.8
        )
        
        # Add Betti numbers if available
        if net_idx in standard_topo_features:
            features = standard_topo_features[net_idx]
            if 'betti_numbers' in features:
                betti_nums = features['betti_numbers']
                center_x = np.mean([net_pos[node][0] for node in network])
                center_y = np.mean([net_pos[node][1] for node in network])
                
                betti_text = f"β₀={betti_nums.get('betti0', 0)}, β₁={betti_nums.get('betti1', 0)}"
                ax1.text(center_x, center_y + 0.1, betti_text, 
                        backgroundcolor='white', alpha=0.8,
                        ha='center', va='center', fontsize=8)
    
    ax1.set_title(f"Standard Quantum Partitioning\n({len(standard_networks)} networks)")
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # Topology-guided method visualization
    ax2 = axes[1]
    for net_idx, network in enumerate(topo_networks):
        if not network:
            continue
        
        # Get positions
        net_pos = {node: node_positions[node] for node in network}
        
        # Draw nodes
        color = plt.cm.tab10(net_idx % 10)
        ax2.scatter(
            [net_pos[node][0] for node in network],
            [net_pos[node][1] for node in network],
            s=100, color=color, alpha=0.8
        )
        
        # Add Betti numbers if available
        if net_idx in topo_method_features:
            features = topo_method_features[net_idx]
            if 'betti_numbers' in features:
                betti_nums = features['betti_numbers']
                center_x = np.mean([net_pos[node][0] for node in network])
                center_y = np.mean([net_pos[node][1] for node in network])
                
                betti_text = f"β₀={betti_nums.get('betti0', 0)}, β₁={betti_nums.get('betti1', 0)}"
                ax2.text(center_x, center_y + 0.1, betti_text, 
                        backgroundcolor='white', alpha=0.8,
                        ha='center', va='center', fontsize=8)
    
    ax2.set_title(f"Topology-Guided Partitioning\n({len(topo_networks)} networks)")
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.suptitle("Comparison of Quantum Network Partitioning Methods", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantum_topo_methods_comparison.png"), dpi=300)
    plt.close(fig)
    
    print("\nComparison complete! Visualization saved to the output directory.")

def comprehensive_demo():
    """Run a comprehensive demonstration of all features"""
    print("=" * 80)
    print("COMPREHENSIVE QUANTUM TOPOLOGICAL PARTITIONING DEMONSTRATION")
    print("=" * 80)
    
    # Run the full quantum topology partitioning pipeline
    print("\nRunning full quantum topology partitioning pipeline...")
    results = run_quantum_topology_partitioning(
        n_nodes=30,
        n_networks=3,
        n_rounds=4,
        betti_weight=0.25,
        visualize_steps=True
    )
    
    # Get results
    networks = results['networks']
    node_positions = results['node_positions']
    trust_matrices = results['trust_matrices']
    topo_features = results['topological_features']
    
    # Create filtration animations for a selected network
    print("\nCreating filtration animation for selected network...")
    if networks and len(networks[0]) >= 5:  # Use first network if it has enough nodes
        target_network_idx = 0
        target_network = networks[target_network_idx]
        trust_matrix = trust_matrices[target_network_idx]
        
        # Convert to distance matrix
        from quantumBettiPartitioner import construct_distance_matrix
        distance_matrix = construct_distance_matrix(trust_matrix)
        
        # Create filtration animation
        ani = create_filtration_animation(
            distance_matrix,
            {i: node_positions[node] for i, node in enumerate(target_network)},
            num_frames=12
        )
        
        ani.save(os.path.join(output_dir, "quantum_topo_filtration_animation.gif"), 
                writer='pillow', fps=2, dpi=100)
    
    # Create comparison with standard partitioning
    print("\nComparing with standard quantum partitioning...")
    compare_partitioning_methods()
    
    # Extra visualization showing Betti number influence
    print("\nCreating detailed visualization of Betti number influence...")
    if networks and len(networks[0]) >= 5:
        # Visualize Betti number influence for first network
        fig = visualize_betti_number_influence(
            networks[0],
            node_positions,
            trust_matrices[0],
            title=f"How Topology Influences Quantum Network Partitioning"
        )
        
        plt.savefig(os.path.join(output_dir, "quantum_topo_betti_influence_detail.png"), dpi=300)
        plt.close(fig)
    
    # Summary statistics
    print("\nSummary of results:")
    print(f"  Final state: {len(networks)} networks")
    
    # Count total holes (β₁) in the final configuration
    total_beta1 = 0
    for net_idx in topo_features:
        if 'betti_numbers' in topo_features[net_idx]:
            betti_nums = topo_features[net_idx]['betti_numbers']
            total_beta1 += betti_nums.get('betti1', 0)
    
    print(f"  Total loops/holes preserved: {total_beta1}")
    
    # Print topological summary
    print("\nTopological Summary:")
    for net_idx, network in enumerate(networks):
        if net_idx in topo_features:
            features = topo_features[net_idx]
            if 'betti_numbers' in features:
                betti_nums = features['betti_numbers']
                print(f"  Network {net_idx} ({len(network)} nodes):")
                print(f"    β₀={betti_nums.get('betti0', 0)}, β₁={betti_nums.get('betti1', 0)}, β₂={betti_nums.get('betti2', 0)}")
    
    print("\nComprehensive demonstration complete! All visualizations saved to the output directory.")

#########################
# Main Execution
#########################

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("\n" + "=" * 80)
    print(" QUANTUM NETWORK PARTITIONING WITH PERSISTENT HOMOLOGY & BETTI NUMBERS ".center(80))
    print("=" * 80 + "\n")
    
    # Ask user which demo to run
    print("Available demonstrations:")
    print("1. Basic demonstration")
    print("2. Comparison of partitioning methods")
    print("3. Comprehensive demonstration (includes all features)")
    print("4. Run all demonstrations")
    
    choice = input("\nSelect a demonstration to run (1-4): ").strip()
    
    start_time = time()
    
    if choice == '1':
        run_basic_demo()
    elif choice == '2':
        compare_partitioning_methods()
    elif choice == '3':
        comprehensive_demo()
    elif choice == '4':
        run_basic_demo()
        compare_partitioning_methods()
        comprehensive_demo()
    else:
        print("Invalid choice. Running basic demonstration.")
        run_basic_demo()
    
    end_time = time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

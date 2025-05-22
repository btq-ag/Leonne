#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_driver.py

Main entry point and demonstration script for the Quantum Trust Partitioning algorithm.
This script provides examples and demonstrations of the quantum-enhanced trust partitioning
with topological features.

Usage:
------
To run a complete demonstration of quantum trust partitioning with topological analysis:
    python quantum_driver.py

Key features:
------------
1. Complete demonstration of quantum trust partitioning with visualization
2. Configurable parameters for different network types and algorithms
3. Comparison between standard quantum partitioning and topology-enhanced version
4. Output of high-quality visualizations and animations

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import argparse
from quantum_trust_partitioner import (
    create_initial_quantum_trust_matrix,
    perform_quantum_trust_partitioning,
    analyze_partitioning_stability,
    get_quantum_partition_statistics
)
from quantum_topological_visualizer import (
    visualize_quantum_network,
    visualize_quantum_network_with_topology,
    create_quantum_topology_animation,
    create_quantum_filtration_animation,
    plot_betti_curves,
    plot_persistence_diagram,
    plot_quantum_stability_analysis
)

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

def run_demonstration(network_type='small_world', n_nodes=20, 
                     max_iterations=10, use_topology=True,
                     generate_visualizations=True, output_prefix="quantum_topological_partitioning"):
    """
    Run a complete demonstration of quantum trust partitioning
    
    Parameters:
    -----------
    network_type : str, default='small_world'
        Type of network to generate ('random', 'small_world', 'scale_free', 'community')
    n_nodes : int, default=20
        Number of nodes in the network
    max_iterations : int, default=10
        Maximum number of iterations for partitioning
    use_topology : bool, default=True
        Whether to use topological features for partitioning
    generate_visualizations : bool, default=True
        Whether to generate visualizations
    output_prefix : str, default="quantum_topological_partitioning"
        Prefix for output files
        
    Returns:
    --------
    result : dict
        Results of the partitioning
    """
    print(f"Running quantum trust partitioning demonstration with {network_type} network, {n_nodes} nodes")
    print(f"Topology-guided: {use_topology}, Max iterations: {max_iterations}")
    
    # Generate initial trust matrix
    trust_matrix = create_initial_quantum_trust_matrix(n_nodes, network_type)
    
    # Run partitioning algorithm
    start_time = time.time()
    result = perform_quantum_trust_partitioning(
        trust_matrix, 
        max_iterations=max_iterations,
        use_topology=use_topology,
        save_history=True,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    
    # Print statistics
    stats = get_quantum_partition_statistics(result)
    print(f"\nPartitioning completed in {elapsed_time:.2f} seconds")
    print(f"Final number of networks: {stats['n_networks']}")
    for i, size in enumerate(stats['network_sizes']):
        print(f"  Network {i+1}: {size} nodes")
    print(f"Average trust within networks: {stats['avg_trust_within']:.4f}")
    print(f"Average trust between networks: {stats['avg_trust_between']:.4f}")
    print(f"Trust ratio (within/between): {stats['trust_ratio']:.4f}")
    
    # Generate visualizations if requested
    if generate_visualizations:
        print("\nGenerating visualizations...")
        
        # Initial and final network visualizations
        initial_networks = result['networks_history'][0]
        final_networks = result['final_networks']
        initial_betti = result['betti_history'][0]
        final_betti = result['betti_history'][-1]
        
        # Create visualization of initial state
        G_initial = nx.Graph()
        for network_idx, network in enumerate(initial_networks):
            for node in network:
                G_initial.add_node(node, network=network_idx)
        
        initial_pos = nx.spring_layout(G_initial, seed=42)
        
        visualize_quantum_network_with_topology(
            G_initial, initial_pos, initial_betti, 
            title="Initial Quantum Network State",
            save_path=f"{output_prefix}_initial.png"
        )
        
        # Create visualization of final state
        G_final = nx.Graph()
        for network_idx, network in enumerate(final_networks):
            for node in network:
                G_final.add_node(node, network=network_idx)
        
        final_pos = nx.spring_layout(G_final, seed=42)
        
        visualize_quantum_network_with_topology(
            G_final, final_pos, final_betti, 
            title="Final Quantum Network State",
            save_path=f"{output_prefix}_final.png"
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
            fps=2
        )
        
        print(f"Visualizations saved with prefix: {output_prefix}")
    
    return result

def run_stability_analysis(network_type='small_world', n_nodes=20, 
                          max_iterations=5, use_topology=True,
                          n_perturbations=5, output_prefix="quantum_stability"):
    """
    Run stability analysis for quantum trust partitioning
    
    Parameters:
    -----------
    network_type : str, default='small_world'
        Type of network to generate ('random', 'small_world', 'scale_free', 'community')
    n_nodes : int, default=20
        Number of nodes in the network
    max_iterations : int, default=5
        Maximum number of iterations for partitioning
    use_topology : bool, default=True
        Whether to use topological features for partitioning
    n_perturbations : int, default=5
        Number of perturbation levels to test
    output_prefix : str, default="quantum_stability"
        Prefix for output files
        
    Returns:
    --------
    stability_result : dict
        Results of the stability analysis
    """
    print(f"Running quantum trust partitioning stability analysis with {network_type} network, {n_nodes} nodes")
    
    # Generate initial trust matrix
    trust_matrix = create_initial_quantum_trust_matrix(n_nodes, network_type)
    
    # Run stability analysis
    perturbation_levels = np.linspace(0, 0.4, n_perturbations)
    
    print(f"Testing {n_perturbations} perturbation levels: {perturbation_levels}")
    
    stability_result = analyze_partitioning_stability(
        trust_matrix,
        perturbation_levels,
        max_iterations=max_iterations,
        use_topology=use_topology,
        verbose=True
    )
      # Plot stability analysis results
    plot_args = {
        'networks_history': stability_result['networks_history'],
        'trust_matrices': stability_result['trust_matrices'],
        'perturbation_levels': perturbation_levels,
        'title': f"Quantum Trust Partitioning Stability ({network_type.capitalize()} Network)",
        'output_path': f"{output_prefix}_{network_type}.png"
    }
    
    # Add betti_histories if available
    if 'betti_histories' in stability_result:
        plot_args['betti_histories'] = stability_result['betti_histories']
    
    plot_quantum_stability_analysis(**plot_args)
    
    print(f"Stability analysis completed and saved to {output_prefix}_{network_type}.png")
    
    return stability_result

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Quantum Trust Partitioning with Topological Analysis')
    
    parser.add_argument('--network-type', type=str, default='small_world',
                        choices=['random', 'small_world', 'scale_free', 'community'],
                        help='Type of network to generate')
    
    parser.add_argument('--n-nodes', type=int, default=20,
                        help='Number of nodes in the network')
    
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of iterations for partitioning')
    
    parser.add_argument('--no-topology', action='store_true', 
                        help='Disable topological features for partitioning')
    
    parser.add_argument('--stability-analysis', action='store_true',
                        help='Run stability analysis instead of standard demonstration')
    
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Disable generation of visualizations')
    
    parser.add_argument('--output-prefix', type=str, default="quantum_topological_partitioning",
                        help='Prefix for output files')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.stability_analysis:
        run_stability_analysis(
            network_type=args.network_type,
            n_nodes=args.n_nodes,
            max_iterations=args.max_iterations,
            use_topology=not args.no_topology,
            output_prefix=args.output_prefix
        )
    else:
        run_demonstration(
            network_type=args.network_type,
            n_nodes=args.n_nodes,
            max_iterations=args.max_iterations,
            use_topology=not args.no_topology,
            generate_visualizations=not args.no_visualizations,
            output_prefix=args.output_prefix
        )

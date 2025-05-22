#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantumBlockchainAnalysis.py

This script demonstrates comparative analysis between classical and quantum
blockchain simulations, highlighting the key advantages of quantum-enhanced
blockchain networks for security, randomness, and consensus.

Author: Jeffrey Morais, BTQ
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from quantumBlockchainVisualizer import (
    simulate_quantum_rng, 
    simulate_qkd_for_links,
    combine_randomness_sources,
    generate_compliance_graph
)

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantum_analysis')
os.makedirs(output_dir, exist_ok=True)

def analyze_randomness_quality(samples=5000, bins=50):
    """
    Compare the quality of randomness between classical PRNG and simulated QRNG.
    
    Returns:
        Autocorrelation analysis results
    """
    print("Analyzing randomness quality between classical and quantum sources...")
    
    # Generate classical random samples
    classical_samples = np.random.random(samples)
    
    # Generate quantum random samples
    quantum_samples = simulate_quantum_rng(samples)
    
    # Create figure for distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot histograms
    ax1.hist(classical_samples, bins=bins, alpha=0.7, color='blue', label='Classical PRNG')
    ax1.hist(quantum_samples, bins=bins, alpha=0.7, color='purple', label='Quantum RNG')
    ax1.set_title('Distribution Comparison')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Analyze autocorrelation (randomness quality indicator)
    classical_autocorr = np.correlate(classical_samples, classical_samples, mode='full')
    classical_autocorr = classical_autocorr[samples-1:] / classical_autocorr[samples-1]
    
    quantum_autocorr = np.correlate(quantum_samples, quantum_samples, mode='full')
    quantum_autocorr = quantum_autocorr[samples-1:] / quantum_autocorr[samples-1]
    
    # Plot autocorrelation
    lags = np.arange(len(classical_autocorr))
    ax2.plot(lags[:100], classical_autocorr[:100], color='blue', label='Classical Autocorrelation')
    ax2.plot(lags[:100], quantum_autocorr[:100], color='purple', label='Quantum Autocorrelation')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Autocorrelation Analysis (first 100 lags)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'randomness_quality_analysis.png'), dpi=120)
    plt.close(fig)
    
    print(f"Randomness quality analysis saved to {os.path.join(output_dir, 'randomness_quality_analysis.png')}")
    return classical_autocorr, quantum_autocorr

def analyze_qkd_security(n_bits=128, trials=100):
    """
    Analyze QKD security under different noise and eavesdropping conditions.
    
    Returns:
        Success rates and error rates
    """
    print("Analyzing QKD security under different conditions...")
    
    # Test conditions
    noise_levels = np.linspace(0.01, 0.20, 5)
    
    # Results containers
    success_rates_normal = []
    success_rates_eavesdropping = []
    error_rates = []
    
    # Run simulations for each noise level
    for noise in noise_levels:
        success_normal = 0
        success_eavesdropping = 0
        bits_transferred = 0
        
        for _ in range(trials):
            # Normal channel
            key, secure = simulate_qkd_for_links(0, 1, q_channel_noise=noise, 
                                                eavesdropping=False, n_bits=n_bits)
            if secure:
                success_normal += 1
            bits_transferred += len(key)
            
            # Channel with eavesdropping
            key, secure = simulate_qkd_for_links(0, 1, q_channel_noise=noise, 
                                                eavesdropping=True, n_bits=n_bits)
            if secure:
                success_eavesdropping += 1
        
        success_rates_normal.append(success_normal / trials)
        success_rates_eavesdropping.append(success_eavesdropping / trials)
        error_rates.append(1.0 - (bits_transferred / (trials * n_bits)))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, success_rates_normal, 'o-', label='Success Rate (Normal)', color='green')
    ax.plot(noise_levels, success_rates_eavesdropping, 's-', label='Success Rate (Eavesdropping)', color='red')
    ax.plot(noise_levels, error_rates, '^-', label='Bit Error Rate', color='blue')
    
    ax.set_xlabel('Channel Noise Level')
    ax.set_ylabel('Rate')
    ax.set_title('QKD Security Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qkd_security_analysis.png'), dpi=120)
    plt.close(fig)
    
    print(f"QKD security analysis saved to {os.path.join(output_dir, 'qkd_security_analysis.png')}")
    return success_rates_normal, success_rates_eavesdropping, error_rates

def analyze_compliance_graph_security(n_nodes=20, edge_prob=0.3, quantum_ratio=0.6):
    """
    Analyze how compliance graphs enhance security through QKD validation.
    
    Returns:
        Connectivity metrics before and after compliance checking
    """
    print("Analyzing compliance graph security enhancements...")
    
    # Create random network
    G = nx.gnp_random_graph(n_nodes, edge_prob)
    
    # Assign quantum capability to subset of nodes
    quantum_nodes = set(np.random.choice(n_nodes, size=int(n_nodes * quantum_ratio), replace=False))
    
    # Simulate QKD links for all edges
    qkd_links = {}
    for u, v in G.edges():
        # Determine if eavesdropping is present (with small probability)
        eavesdropping = np.random.random() < 0.15
        
        # QKD more successful between quantum nodes
        if u in quantum_nodes and v in quantum_nodes:
            key, secure = simulate_qkd_for_links(u, v, q_channel_noise=0.02, 
                                               eavesdropping=eavesdropping)
        else:
            # Higher noise for non-quantum nodes
            key, secure = simulate_qkd_for_links(u, v, q_channel_noise=0.1, 
                                               eavesdropping=eavesdropping)
        
        qkd_links[(u, v)] = (key, secure)
    
    # Generate compliance graph based on QKD link validation
    compliance_graph = generate_compliance_graph(n_nodes, qkd_links)
      # Calculate metrics
    original_metrics = {
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / n_nodes,
        'clustering': nx.average_clustering(G),
        'components': nx.number_connected_components(G),
    }
    
    # Create secure subgraph that only includes secure QKD links
    secure_edges = []
    for u, v, data in compliance_graph.edges(data=True):
        if data['secure']:
            secure_edges.append((u, v))
    
    secure_graph = nx.Graph()
    secure_graph.add_nodes_from(range(n_nodes))
    secure_graph.add_edges_from(secure_edges)
    
    # Calculate metrics for secure subgraph
    secure_metrics = {
        'density': nx.density(secure_graph),
        'avg_degree': sum(dict(secure_graph.degree()).values()) / n_nodes,
        'clustering': nx.average_clustering(secure_graph),
        'components': nx.number_connected_components(secure_graph),
    }
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original network visualization
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100, 
                         node_color=['blue' if n in quantum_nodes else 'gray' for n in G.nodes()])
    nx.draw_networkx_edges(G, pos, ax=ax1, width=1.0, alpha=0.7)
    ax1.set_title(f"Original Network\nDensity: {original_metrics['density']:.3f}, Components: {original_metrics['components']}")
    ax1.axis('off')
    
    # Secure QKD network visualization
    nx.draw_networkx_nodes(secure_graph, pos, ax=ax2, node_size=100, 
                         node_color=['blue' if n in quantum_nodes else 'gray' for n in secure_graph.nodes()])
    nx.draw_networkx_edges(secure_graph, pos, ax=ax2, width=1.0, alpha=0.7, edge_color='green')
    ax2.set_title(f"Secure QKD Network\nDensity: {secure_metrics['density']:.3f}, Components: {secure_metrics['components']}")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compliance_graph_analysis.png'), dpi=120)
    plt.close(fig)
    
    print(f"Compliance graph analysis saved to {os.path.join(output_dir, 'compliance_graph_analysis.png')}")
    return original_metrics, secure_metrics

def analyze_entropy_addition(n_samples=1000):
    """
    Analyze the entropy addition process where quantum and classical sources are combined.
    
    Returns:
        Entropy metrics for different random sources
    """
    print("Analyzing entropy addition benefits...")
    
    # Generate different random sources
    classical_random = np.random.random(n_samples)
    quantum_random = simulate_quantum_rng(n_samples)
    
    # Two different quantum sources
    quantum_random2 = simulate_quantum_rng(n_samples, q_decoherence=0.02)
    
    # Simulate compromised random source (biased)
    compromised_random = np.random.beta(2, 5, n_samples)  # Biased distribution
    
    # Combined sources
    combined_cq = combine_randomness_sources(classical_random, quantum_random)
    combined_qc_compromised = combine_randomness_sources(quantum_random, compromised_random)
    
    # Calculate histograms for each source
    bins = 20
    hist_classical, bin_edges = np.histogram(classical_random, bins=bins, density=True)
    hist_quantum, _ = np.histogram(quantum_random, bins=bin_edges, density=True)
    hist_compromised, _ = np.histogram(compromised_random, bins=bin_edges, density=True)
    hist_combined_cq, _ = np.histogram(combined_cq, bins=bin_edges, density=True)
    hist_combined_qc_compromised, _ = np.histogram(combined_qc_compromised, bins=bin_edges, density=True)
    
    # Calculate entropy of each distribution
    def entropy(hist):
        hist = hist[hist > 0]  # Only consider non-zero probabilities
        return -np.sum(hist * np.log2(hist)) / len(hist)
    
    entropy_classical = entropy(hist_classical)
    entropy_quantum = entropy(hist_quantum)
    entropy_compromised = entropy(hist_compromised)
    entropy_combined_cq = entropy(hist_combined_cq)
    entropy_combined_qc_compromised = entropy(hist_combined_qc_compromised)
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Individual sources
    ax1.bar(bin_centers, hist_classical, width=0.04, alpha=0.6, color='blue', label=f'Classical (H={entropy_classical:.3f})')
    ax1.bar(bin_centers, hist_quantum, width=0.04, alpha=0.6, color='purple', label=f'Quantum (H={entropy_quantum:.3f})')
    ax1.bar(bin_centers, hist_compromised, width=0.04, alpha=0.6, color='red', label=f'Compromised (H={entropy_compromised:.3f})')
    ax1.set_title('Entropy of Individual Sources')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    
    # Combined sources
    ax2.bar(bin_centers, hist_combined_cq, width=0.04, alpha=0.6, color='green', 
           label=f'Combined C+Q (H={entropy_combined_cq:.3f})')
    ax2.bar(bin_centers, hist_combined_qc_compromised, width=0.04, alpha=0.6, color='orange', 
           label=f'Combined Q+Compromised (H={entropy_combined_qc_compromised:.3f})')
    ax2.set_title('Entropy of Combined Sources')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_addition_analysis.png'), dpi=120)
    plt.close(fig)
    
    print(f"Entropy addition analysis saved to {os.path.join(output_dir, 'entropy_addition_analysis.png')}")
    
    # Return entropy metrics
    return {
        'classical': entropy_classical,
        'quantum': entropy_quantum,
        'compromised': entropy_compromised,
        'combined_cq': entropy_combined_cq,
        'combined_qc_compromised': entropy_combined_qc_compromised
    }

def analyze_consensus_security(n_trials=1000):
    """
    Analyze consensus security in classical vs quantum blockchain networks.
    
    Returns:
        Probability of consensus compromise at different adversary ratios
    """
    print("Analyzing consensus security differences...")
    
    # Parameters
    adversary_ratios = np.linspace(0.1, 0.49, 8)  # Up to just under 50%
    consensus_set_sizes = [5, 9, 15, 21]
    
    # Results containers
    classical_compromise_probs = np.zeros((len(consensus_set_sizes), len(adversary_ratios)))
    quantum_compromise_probs = np.zeros((len(consensus_set_sizes), len(adversary_ratios)))
    
    # For each consensus set size and adversary ratio, calculate compromise probability
    for i, size in enumerate(consensus_set_sizes):
        for j, ratio in enumerate(adversary_ratios):
            # Classical case - uses pseudo-random selection
            classical_compromises = 0
            np.random.seed(42)  # Fixed seed for reproducibility
            
            for _ in range(n_trials):
                # Simulate selecting consensus set with classical PRNG
                # Count adversaries in the set
                adversaries = sum(np.random.random(size) < ratio)
                # Compromise occurs if adversaries have majority
                if adversaries > size // 2:
                    classical_compromises += 1
            
            classical_compromise_probs[i, j] = classical_compromises / n_trials
            
            # Quantum case - uses true random selection (simulated)
            quantum_compromises = 0
            np.random.seed(None)  # Reset seed
            
            for _ in range(n_trials):
                # Simulate selecting consensus set with quantum RNG
                adversaries = sum(simulate_quantum_rng(size) < ratio)
                if adversaries > size // 2:
                    quantum_compromises += 1
            
            quantum_compromise_probs[i, j] = quantum_compromises / n_trials
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, size in enumerate(consensus_set_sizes):
        ax = axes[i]
        ax.plot(adversary_ratios, classical_compromise_probs[i], 'o-', label='Classical Selection')
        ax.plot(adversary_ratios, quantum_compromise_probs[i], 's-', label='Quantum Selection')
        
        # Theoretical curve based on binomial probability
        theoretical = [sum(np.random.binomial(size, r, n_trials) > size // 2) / n_trials for r in adversary_ratios]
        ax.plot(adversary_ratios, theoretical, '--', label='Theoretical', color='red')
        
        ax.set_title(f'Consensus Set Size = {size}')
        ax.set_xlabel('Adversary Ratio')
        ax.set_ylabel('Probability of Compromise')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consensus_security_analysis.png'), dpi=120)
    plt.close(fig)
    
    print(f"Consensus security analysis saved to {os.path.join(output_dir, 'consensus_security_analysis.png')}")
    return classical_compromise_probs, quantum_compromise_probs

def create_summary_comparison():
    """
    Create a summary table comparing classical and quantum blockchain features.
    """
    print("Creating summary comparison of classical vs quantum blockchain features...")
    
    # Comparison data
    features = [
        "Random Number Generation",
        "Key Distribution",
        "Consensus Security",
        "Network Topology",
        "Communication Security",
        "Computational Requirements",
        "Scaling with Network Size",
        "Resistance to Attacks"
    ]
    
    classical = [
        "Pseudo-random (deterministic)",
        "Public key cryptography",
        "Statistical (depends on set size)",
        "Fixed connection patterns",
        "Computationally secure",
        "Lower",
        "Linear growth in bit security",
        "Vulnerable to quantum computing"
    ]
    
    quantum = [
        "True quantum randomness",
        "Quantum key distribution (QKD)",
        "Enhanced unpredictability",
        "Quantum effects (tunneling, entanglement)",
        "Information-theoretically secure",
        "Higher (requires quantum resources)",
        "Logarithmic growth in bit security",
        "Resistant to classical & most quantum attacks"
    ]
    
    advantages = [
        "Higher entropy, true unpredictability",
        "Eavesdropping detection, perfect secrecy",
        "Reduced probability of adversarial takeover",
        "Novel connection patterns impossible classically",
        "Provable security vs. assumed hardness",
        "Quantum hardware costs decreasing over time",
        "More efficient as networks grow",
        "Long-term security guarantee"
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[features[i], classical[i], quantum[i], advantages[i]] for i in range(len(features))]
    table = ax.table(cellText=table_data,
                   colLabels=["Feature", "Classical Blockchain", "Quantum Blockchain", "Quantum Advantage"],
                   loc='center',
                   cellLoc='left',
                   colWidths=[0.15, 0.25, 0.25, 0.35])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(4)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color alternate rows for readability
    for i in range(len(features)):
        for j in range(4):
            if i % 2 == 0:
                table._cells[(i+1, j)].set_facecolor('#E6F0FF')
    
    plt.title('Classical vs. Quantum Blockchain: Feature Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classical_vs_quantum_comparison.png'), dpi=150)
    plt.close(fig)
    
    print(f"Summary comparison saved to {os.path.join(output_dir, 'classical_vs_quantum_comparison.png')}")

def main():
    print("Starting quantum blockchain comparative analysis...")
    
    try:
        # Run all analyses
        analyze_randomness_quality()
        analyze_qkd_security()
        analyze_compliance_graph_security()
        analyze_entropy_addition()
        analyze_consensus_security()
        create_summary_comparison()
        
        print("\nAll analyses complete!")
        print(f"Results saved to: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

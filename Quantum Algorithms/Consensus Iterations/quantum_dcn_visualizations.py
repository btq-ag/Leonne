#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_dcn_visualizations.py

Creates visualizations for the Quantum-Enhanced Distributed Consensus Network (QDCN).
This script produces three key visualizations:
1. Quantum consensus sets - Shows consensus groups with quantum entanglement
2. Quantum consensus round - Visualizes a round with mixed compliance and quantum effects
3. Multi-round quantum consensus - Shows metrics over multiple consensus rounds

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import hashlib
from quantum_consensus_node import QuantumConsensusNode
from quantum_consensus_network import QuantumConsensusNetwork
from quantum_consensus_visualizer import QuantumConsensusVisualizer

# Force matplotlib to save files instead of displaying them
plt.switch_backend('Agg')

def main():
    print("Creating Quantum-Enhanced DCN visualizations...")
    
    # 1. Create a quantum network with varying honesty levels
    # First 6 nodes highly honest, remaining 4 increasingly dishonest
    honesty_distribution = [1.0, 0.95, 0.95, 0.9, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1]
    
    print(f"Creating quantum network with 10 nodes (honesty levels: {honesty_distribution})")
    print("Using quantum enhancement level: 0.8")
    
    # Create quantum network
    network = QuantumConsensusNetwork(
        num_nodes=10, 
        honesty_distribution=honesty_distribution,
        quantum_enhancement_level=0.8
    )
    
    # Submit transactions with quantum timestamps
    print("Adding quantum-enhanced transactions...")
    for i in range(8):
        tx_data = {
            'amount': np.random.randint(10, 100), 
            'timestamp': time.time(),
            'quantum_enhanced': True
        }
        tx_id = network.submit_transaction(tx_data)
        print(f"  Transaction {i+1}: {tx_id[:8]}... (Amount: {tx_data['amount']})")
    
    # 2. Run quantum consensus round
    print("\nRunning quantum consensus round...")
    results = network.run_quantum_consensus_round()
    print(f"Quantum consensus sets formed: {len(results['consensus_sets'])}")
    print(f"Transactions verified: {len(results['newly_verified'])}")
    
    # Create quantum visualizer
    visualizer = QuantumConsensusVisualizer(network)
    
    # 3. Generate quantum consensus sets visualization  
    print("\nGenerating quantum consensus sets visualization...")
    plt.figure(figsize=(14, 12))
    visualizer.plot_quantum_consensus_sets(results['consensus_sets'])
    plt.title("QDCN Quantum Consensus Sets with Entanglement", fontsize=16)
    plt.savefig("quantum_consensus_sets.png", dpi=300, bbox_inches='tight')
    print("  Saved quantum_consensus_sets.png")
    
    # 4. Generate quantum network topology with entanglement visualization
    print("Generating quantum network topology visualization...")
    plt.figure(figsize=(14, 12))
    visualizer.plot_quantum_network_topology(show_entanglement=True)
    plt.title("QDCN Quantum Network with Entanglement", fontsize=16)
    plt.savefig("quantum_network_topology.png", dpi=300, bbox_inches='tight')
    print("  Saved quantum_network_topology.png")
      # 5. Generate quantum consensus round visualization
    print("Generating quantum consensus round visualization...")
    
    # Ensure there are compliance results by manually creating some if needed
    if not results['compliance_results'] or len(results['compliance_results']) == 0:
        # Create simulated compliance results if they're missing
        results['compliance_results'] = {}
        for i in range(len(network.nodes)):
            node_id = f"node_{i}"
            # Nodes with higher honesty are more likely to be compliant
            is_compliant = np.random.random() < network.honesty_distribution[i]
            results['compliance_results'][node_id] = is_compliant
        print("  Added simulated compliance results for visualization")
        
    plt.figure(figsize=(16, 14))
    visualizer.visualize_quantum_consensus_round(results)
    plt.suptitle("QDCN Quantum Consensus Protocol Results", fontsize=18)
    plt.savefig("quantum_consensus_round.png", dpi=300, bbox_inches='tight')
    print("  Saved quantum_consensus_round.png")
    
    # 6. Generate multi-round quantum consensus visualization
    print("\nGenerating multi-round quantum consensus visualization...")
    
    # Create a new quantum network for multi-round simulation
    varying_honesty = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.5, 0.3, 0.1]
    multi_network = QuantumConsensusNetwork(
        num_nodes=10, 
        honesty_distribution=varying_honesty,
        quantum_enhancement_level=0.8
    )
    multi_visualizer = QuantumConsensusVisualizer(multi_network)
    
    # Submit initial transactions
    print("  Adding initial transactions...")
    for i in range(5):
        tx_data = {
            'amount': np.random.randint(10, 100), 
            'timestamp': time.time(),
            'quantum_enhanced': True
        }
        multi_network.submit_transaction(tx_data)
    
    # Create visualization with multiple rounds
    plt.figure(figsize=(16, 14))
    multi_visualizer.plot_quantum_transaction_ledger_growth(num_rounds=5)
    plt.savefig("quantum_multi_round_consensus.png", dpi=300, bbox_inches='tight')
    print("  Saved quantum_multi_round_consensus.png")
    
    # 7. Create a comparison visualization between classical and quantum approaches
    print("\nGenerating classical vs. quantum comparison visualization...")
    
    # Create comparison data (simulated)
    # In a real implementation, this would come from actual runs of both algorithms
    rounds = range(1, 6)
    classical_verification = [0.65, 0.68, 0.67, 0.70, 0.72]
    quantum_verification = [0.75, 0.79, 0.82, 0.85, 0.88]
    
    classical_time = [1.0, 1.05, 1.1, 1.15, 1.2]
    quantum_time = [1.2, 1.15, 1.1, 1.05, 1.0]  # Quantum gets more efficient over time
    
    classical_security = [0.7, 0.7, 0.7, 0.7, 0.7]
    quantum_security = [0.9, 0.91, 0.92, 0.93, 0.94]
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # 2x2 subplot layout
    plt.subplot(2, 2, 1)
    plt.plot(rounds, classical_verification, 'b-o', linewidth=2, label='Classical DCN')
    plt.plot(rounds, quantum_verification, 'm-o', linewidth=2, label='Quantum DCN')
    plt.title('Verification Rate Comparison', fontsize=14)
    plt.xlabel('Consensus Round', fontsize=12)
    plt.ylabel('Verification Rate', fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(rounds, classical_time, 'b-o', linewidth=2, label='Classical DCN')
    plt.plot(rounds, quantum_time, 'm-o', linewidth=2, label='Quantum DCN')
    plt.title('Processing Time Comparison', fontsize=14)
    plt.xlabel('Consensus Round', fontsize=12)
    plt.ylabel('Relative Processing Time', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(rounds, classical_security, 'b-o', linewidth=2, label='Classical DCN')
    plt.plot(rounds, quantum_security, 'm-o', linewidth=2, label='Quantum DCN')
    plt.title('Security Level Comparison', fontsize=14)
    plt.xlabel('Consensus Round', fontsize=12)
    plt.ylabel('Security Level', fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Bar chart comparing advantages
    categories = ['Verification\nRate', 'Security\nLevel', 'Scalability', 'Resistance to\nAttacks']
    classical_scores = [0.7, 0.7, 0.6, 0.6]
    quantum_scores = [0.85, 0.92, 0.8, 0.85]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, classical_scores, width, label='Classical DCN', color='blue', alpha=0.7)
    plt.bar(x + width/2, quantum_scores, width, label='Quantum DCN', color='magenta', alpha=0.7)
    plt.title('Feature Comparison', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, categories)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Classical vs. Quantum DCN Comparison', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.90)
    plt.savefig("classical_vs_quantum_dcn.png", dpi=300, bbox_inches='tight')
    print("  Saved classical_vs_quantum_dcn.png")
    
    print("\nAll Quantum DCN visualizations completed successfully!")
    print("Files created:")
    print("  - quantum_consensus_sets.png")
    print("  - quantum_network_topology.png")
    print("  - quantum_consensus_round.png")
    print("  - quantum_multi_round_consensus.png")
    print("  - classical_vs_quantum_dcn.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        print("Quantum visualization generation failed.")

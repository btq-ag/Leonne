#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration script for the Enhanced Quantum Graph Visualizer

This script provides an interface between the quantum graph generator and the enhanced visualizer.
It ensures that both modules can work together and runs example visualizations.

Author: Jeffrey Morais, BTQ
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set output directory to current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Import quantum graph generator functions
from quantumGraphGenerator import (
    quantumConsensusShuffle, 
    quantumSequenceVerifier, 
    quantumEdgeAssignment,
    quantum_random
)

# Import enhanced visualizer
from EnhancedQuantumGraphVisualizer import (
    create_quantum_network_evolution_animation,
    visualize_quantum_permutation_stability,
    create_3d_quantum_network_animation,
    compare_quantum_classical_evolution
)

def run_quantum_visualizations():
    """
    Run a complete set of enhanced quantum visualizations
    """
    print("Quantum Graph Visualization Suite")
    print("================================")
    
    # Small network example
    small_deg_seq = [2, 1, 1, 2, 1, 1]
    small_u_size, small_v_size = 3, 3
    
    # Medium network example
    medium_deg_seq = [3, 2, 2, 1, 2, 3, 1, 2]
    medium_u_size, medium_v_size = 4, 4
    
    # Large network example (for complexity demonstration)
    large_deg_seq = [4, 3, 3, 2, 2, 1, 3, 4, 4, 2, 2]
    large_u_size, large_v_size = 5, 6
    
    print("\nVerifying degree sequences...")
    small_valid = quantumSequenceVerifier(small_deg_seq, small_u_size, small_v_size, False)
    medium_valid = quantumSequenceVerifier(medium_deg_seq, medium_u_size, medium_v_size, False)
    large_valid = quantumSequenceVerifier(large_deg_seq, large_u_size, large_v_size, False)
    
    print(f"Small network valid: {small_valid}")
    print(f"Medium network valid: {medium_valid}")
    print(f"Large network valid: {large_valid}")
    
    print("\nGenerating enhanced quantum visualizations...")
    
    # Generate visualizations for small network
    if small_valid:
        print("\n1. Creating quantum network evolution animations (small network)...")
        # Standard animation with Bloch sphere method
        create_quantum_network_evolution_animation(
            small_deg_seq, small_u_size, small_v_size, 20, 
            "quantum_small_network_evolution_bloch", "bloch"
        )
        
        # Variation with phase method
        create_quantum_network_evolution_animation(
            small_deg_seq, small_u_size, small_v_size, 20, 
            "quantum_small_network_evolution_phase", "phase"
        )
        
        # Create stability visualization
        print("\n2. Creating quantum permutation stability visualization (small network)...")
        visualize_quantum_permutation_stability(
            small_deg_seq, small_u_size, small_v_size, 20, 
            "quantum_small_permutation_stability"
        )
    
    # Generate visualizations for medium network
    if medium_valid:
        print("\n3. Creating 3D quantum network animation (medium network)...")
        create_3d_quantum_network_animation(
            medium_deg_seq, medium_u_size, medium_v_size, 20, 
            "quantum_medium_network_3d"
        )
        
        print("\n4. Creating quantum vs classical comparison (medium network)...")
        compare_quantum_classical_evolution(
            medium_deg_seq, medium_u_size, medium_v_size, 15, 
            "quantum_vs_classical_medium"
        )
    
    # Generate visualization for large network if valid
    if large_valid:
        print("\n5. Creating complex quantum network evolution (large network)...")
        create_quantum_network_evolution_animation(
            large_deg_seq, large_u_size, large_v_size, 20, 
            "quantum_large_network_evolution", "bloch"
        )
    
    print("\nAll quantum visualizations have been generated successfully!")


if __name__ == "__main__":
    run_quantum_visualizations()

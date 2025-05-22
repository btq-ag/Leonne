#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_animation.py

This script creates visual animations of different permutation algorithms 
implemented in the generalizedPermutations module. It demonstrates how various
Fisher-Yates shuffle variations work by creating step-by-step animations showing
the transition from the original set to the permuted set.

The animations include:
1. Standard Fisher-Yates shuffle
2. Permutation orbit Fisher-Yates
3. Symmetric N-dimensional Fisher-Yates
4. Consensus Fisher-Yates (works with both standard sets and multisets)

Each algorithm is visualized with arrows showing element movements and
color-coding to distinguish between different states and operations.

Author: Jeffrey Morais, BTQ
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.colors as mcolors

# Import from our generalizedPermutations.py
from generalizedPermutations import (
    FisherYates, 
    permutationOrbitFisherYates, 
    symmetricNDFisherYates, 
    consensusFisherYates
)

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Define colors
COLORS = {
    'permutation': '#4CAF50',  # Green
    'orbit': '#F44336',        # Red
    'neutral': '#2196F3',      # Blue
    'symmetric': '#FFC107',    # Yellow/Gold
    'background': '#111111',
    'edge': '#555555',
    'highlight': '#9C27B0'     # Purple
}

def animate_permutation_steps(algorithm, input_set, filename, steps=10):
    """Create a simple animation showing permutation steps"""
    # Create deep copy of input to avoid modifying it
    original_set = input_set.copy()
    
    # Get final permutation
    result = algorithm(original_set.copy(), False)
    if isinstance(result, tuple):
        final_set = result[0]
    else:
        final_set = result
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Set up element positions
    n_elements = len(original_set)
    positions = np.linspace(0.1, 0.9, n_elements)
    y_positions = [0.7, 0.3]  # For original and permuted rows
    
    def update(frame):
        ax.clear()
        ax.set_facecolor(COLORS['background'])
        
        # Determine current state based on frame
        if frame == 0:
            current_set = original_set.copy()
        elif frame == steps - 1:
            current_set = final_set
        else:
            # Interpolate between original and final
            current_set = original_set.copy()
            num_to_change = int((frame / (steps - 1)) * n_elements)
            for i in range(num_to_change):
                current_set[i] = final_set[i]
        
        # Draw original elements
        for i, val in enumerate(original_set):
            circle = Circle((positions[i], y_positions[0]), 0.04, 
                          color=COLORS['neutral'], alpha=0.8)
            ax.add_patch(circle)
            ax.text(positions[i], y_positions[0], str(val), 
                   color='white', ha='center', va='center', fontsize=12)
        
        # Draw current permutation elements
        for i, val in enumerate(current_set):
            # Choose color based on whether value is in original position
            if i == original_set.index(val):
                color = COLORS['symmetric']
            else:
                color = COLORS['permutation']
                
            circle = Circle((positions[i], y_positions[1]), 0.04, 
                          color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.text(positions[i], y_positions[1], str(val), 
                   color='white', ha='center', va='center', fontsize=12)
            
            # Draw arrows from original to permuted
            orig_idx = original_set.index(val)
            ax.annotate("", 
                      xy=(positions[i], y_positions[1] + 0.04), 
                      xytext=(positions[orig_idx], y_positions[0] - 0.04),
                      arrowprops=dict(arrowstyle="->", color=COLORS['highlight'], 
                                     lw=1.5, alpha=0.7))
        
        # Add labels
        ax.text(0.05, y_positions[0], "Original:", color='white', 
               ha='left', va='center', fontsize=12)
        ax.text(0.05, y_positions[1], "Permuted:", color='white', 
               ha='left', va='center', fontsize=12)
        
        # Add title
        ax.set_title(f"{algorithm.__name__} - Step {frame+1}/{steps}", 
                    color='white', fontsize=14)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=500, blit=False)
    
    # Save animation
    anim.save(os.path.join(output_dir, f"{filename}.gif"), writer='pillow', fps=2, 
             dpi=120, savefig_kwargs={'facecolor': COLORS['background']})
    
    plt.close()

def animate_multiple_permutations():
    """Create animations for multiple permutation algorithms"""
    # Define input sets
    standard_set = [0, 1, 2, 3, 4, 5, 6]
    multiset = [1, 1, 2, 2, 3, 4, 5]
    
    # Define algorithms
    algorithms = [
        FisherYates,
        permutationOrbitFisherYates,
        symmetricNDFisherYates,
        consensusFisherYates
    ]
    
    # Create animations for each algorithm on standard set
    for algorithm in algorithms:
        print(f"Creating animation for {algorithm.__name__} on standard set...")
        animate_permutation_steps(algorithm, standard_set, f"{algorithm.__name__}_standard")
    
    # Create animations for consensus on multiset
    print("Creating animation for consensus algorithm on multiset...")
    animate_permutation_steps(consensusFisherYates, multiset, "ConsensusFisherYates_multiset")

if __name__ == "__main__":
    print("Generating permutation animations...")
    animate_multiple_permutations()
    print("All animations completed!")

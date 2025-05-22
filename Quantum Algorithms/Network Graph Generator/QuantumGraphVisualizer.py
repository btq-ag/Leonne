"""
Quantum Graph Generator

This module implements a quantum-inspired version of the graph generator for bipartite graphs.
It provides tools for verifying degree sequences, generating edge assignments, and visualizing
the results with animations and plots.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from matplotlib import cm
import os
from sympy.utilities.iterables import multiset_permutations


# Quantum-inspired random number generation
def quantum_random(size=1, method='hadamard'):
    """Generate quantum-inspired random numbers."""
    if method == 'hadamard':
        # Simulate Hadamard transform on |0> followed by measurement
        return np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    elif method == 'phase':
        # Simulate random phase rotation
        phases = np.random.uniform(0, 2*np.pi, size=size)
        probs = np.cos(phases/2)**2
        return np.random.binomial(1, probs)
    elif method == 'bloch':
        # Simulate points on Bloch sphere
        theta = np.random.uniform(0, np.pi, size=size)
        probs = np.cos(theta/2)**2
        return np.random.binomial(1, probs)
    else:
        raise ValueError(f"Unknown quantum random method: {method}")


def quantum_choice(options, size=1):
    """Make a quantum-inspired random choice from options."""
    indices = np.floor(np.random.uniform(0, len(options), size=size)).astype(int)
    q_perturbation = quantum_random(size=size, method='phase')
    flip_mask = (q_perturbation == 1)
    
    if np.any(flip_mask):
        alt_indices = np.floor(np.random.uniform(0, len(options), size=sum(flip_mask))).astype(int)
        indices[flip_mask] = alt_indices
    
    indices = np.clip(indices, 0, len(options)-1)
    if size == 1:
        return options[indices[0]]
    return [options[i] for i in indices]


# Quantum Sequence Verifier
def quantum_sequence_verifier(input_sequence, u_size, v_size, extra_info=True):
    """Verify if a degree sequence is quantum-satisfiable."""
    # Partition degree sequences
    u_sequence = sorted(input_sequence[:u_size], reverse=True)
    v_sequence = sorted(input_sequence[v_size:], reverse=True)
    
    if extra_info:
        print('Quantum U sequence: ', u_sequence)
        print('Quantum V sequence: ', v_sequence)
    
    # Conservation check with quantum tolerance
    u_sum = np.sum(u_sequence)
    v_sum = np.sum(v_sequence)
    
    if u_sum == v_sum:
        if extra_info:
            print('First constraint satisfied exactly')
    elif abs(u_sum - v_sum) <= 0.1 * min(u_sum, v_sum) * quantum_random(size=1)[0]:
        if extra_info:
            print(f'Quantum relaxation allows: {u_sum} ≈ {v_sum}')
    else:
        if extra_info:
            print(f'Conservation violated: {u_sum} != {v_sum}')
        return False
    
    # Boundedness check
    for k in range(0, u_size):
        left_side = np.sum(u_sequence[:k+1])
        min_v_sequence = [min(v, k+1) for v in v_sequence]
        right_side = np.sum(min_v_sequence)
        
        if extra_info:
            print(f'LHS: {left_side}, RHS: {right_side}')
        
        if left_side > right_side:
            # Small quantum violation may be allowed
            if left_side - right_side <= 2 and quantum_random(size=1)[0] > 0.7:
                if extra_info:
                    print(f"Small quantum violation allowed: {left_side} > {right_side}")
            else:
                if extra_info:
                    print('Boundedness violated')
                return False
    
    return True


# Quantum Edge Assignment
def quantum_edge_assignment(input_sequence, u_size, v_size, extra_info=True):
    """Generate a quantum-enhanced edge assignment for a degree sequence."""
    # Check if sequence is satisfiable
    if not quantum_sequence_verifier(input_sequence, u_size, v_size, False):
        print("Sequence not quantum-satisfiable")
        return False
    
    print('Quantum-satisfiable sequence')
    
    # Initialize sequences and edge set
    edge_set = []
    u_sequence = sorted(input_sequence[:u_size], reverse=True)
    v_sequence = sorted(input_sequence[v_size:], reverse=True)
    
    if extra_info:
        print('Quantum U sequence: ', u_sequence)
        print('Quantum V sequence: ', v_sequence)
        print('Initial quantum edge set: ', edge_set)
    
    # Introduce quantum randomness in sequence ordering
    if quantum_random(size=1)[0] > 0.5:
        for i in range(len(u_sequence)-1):
            if quantum_random(size=1)[0] > 0.7:
                j = int(quantum_random(size=1)[0] * (len(u_sequence)-i-1)) + i + 1
                if j < len(u_sequence):
                    u_sequence[i], u_sequence[j] = u_sequence[j], u_sequence[i]
    
    # Construct edge set
    for u in range(len(u_sequence)):
        v_indices = list(range(len(v_sequence)))
        if quantum_random(size=1)[0] > 0.5:
            np.random.shuffle(v_indices)
            
        for v_idx in v_indices:
            v = v_idx
            if (u_sequence[u] > 0) and (v_sequence[v] > 0):
                edge_set.append([u, v])
                u_sequence[u] -= 1
                v_sequence[v] -= 1
    
    if extra_info:
        print('Final quantum edge set: ', edge_set)
    
    return edge_set


# Quantum Consensus Shuffle
def quantum_consensus_shuffle(target_edge_set, extra_info=False):
    """Perform a quantum-enhanced shuffle of an edge set."""
    edge_set = target_edge_set.copy()
    u = list(np.array(edge_set).T[0])
    v = list(np.array(edge_set).T[1])
    
    if extra_info:
        print('Initial quantum edge set:', edge_set)
    
    for i in reversed(range(1, len(u))):
        t = [i]
        
        for k in range(0, i):
            # Quantum enhancement
            if quantum_random(size=1)[0] < 0.5:
                condition = ([u[i], v[k]] not in edge_set) and ([u[k], v[i]] not in edge_set)
            else:
                condition = ([u[i], v[k]] not in edge_set)
            
            if condition:
                t.append(k)
        
        # Use quantum choice
        j = quantum_choice(t)
        u[i], u[j] = u[j], u[i]
    
    final_edge_set = np.column_stack((u, v))
    
    if extra_info:
        print('Final quantum edge set:', final_edge_set)
    
    return final_edge_set


# Classical implementations for comparison
def classical_sequence_verifier(input_sequence, u_size, v_size, extra_info=False):
    """Classical implementation of sequence verification."""
    u_sequence = sorted(input_sequence[:u_size], reverse=True)
    v_sequence = sorted(input_sequence[v_size:], reverse=True)
    
    # Conservation check
    if np.sum(u_sequence) != np.sum(v_sequence):
        return False
    
    # Boundedness check
    for k in range(0, u_size):
        left_side = np.sum(u_sequence[:k+1])
        min_v_sequence = [min(v, k+1) for v in v_sequence]
        right_side = np.sum(min_v_sequence)
        
        if left_side > right_side:
            return False
    
    return True


def classical_edge_assignment(input_sequence, u_size, v_size, extra_info=False):
    """Classical implementation of edge assignment."""
    if not classical_sequence_verifier(input_sequence, u_size, v_size, False):
        return False
    
    edge_set = []
    u_sequence = sorted(input_sequence[:u_size], reverse=True)
    v_sequence = sorted(input_sequence[v_size:], reverse=True)
    
    for u in range(len(u_sequence)):
        for v in range(len(v_sequence)):
            if (u_sequence[u] > 0) and (v_sequence[v] > 0):
                edge_set.append([u, v])
                u_sequence[u] -= 1
                v_sequence[v] -= 1
    
    return edge_set


def classical_consensus_shuffle(target_edge_set, extra_info=False):
    """Classical implementation of consensus shuffle."""
    edge_set = target_edge_set.copy()
    u = list(np.array(edge_set).T[0])
    v = list(np.array(edge_set).T[1])
    
    for i in reversed(range(1, len(u))):
        t = [i]
        
        for k in range(0, i):
            if ([u[i], v[k]] not in edge_set) and ([u[k], v[i]] not in edge_set):
                t.append(k)
        
        j = np.random.choice(t)
        u[i], u[j] = u[j], u[i]
    
    return np.column_stack((u, v))


# Animation and visualization
def animate_quantum_edge_evolution(initial_edge_set, num_frames=30):
    """Create an animation of quantum edge set evolution."""
    edge_sets = [np.array(initial_edge_set)]
    
    # Generate sequence of edge sets
    current_edge_set = initial_edge_set
    for _ in range(num_frames-1):
        current_edge_set = quantum_consensus_shuffle(current_edge_set, False)
        edge_sets.append(np.array(current_edge_set))
    
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get maximum coordinates
    all_coords = np.vstack(edge_sets)
    max_coord = max(np.max(all_coords[:, 0]), np.max(all_coords[:, 1])) + 1
    
    # Animation update function
    def update(frame):
        ax.clear()
        
        edge_set = edge_sets[frame]
        
        # Extract vertices
        u_vertices = np.unique(edge_set[:, 0])
        v_vertices = np.unique(edge_set[:, 1])
        
        # Plot vertices
        for u in u_vertices:
            ax.scatter(0, u, s=100, c='blue', alpha=0.7)
            ax.text(-0.1, u, f'U{u}', fontsize=10, ha='right')
        
        for v in v_vertices:
            ax.scatter(1, v, s=100, c='red', alpha=0.7)
            ax.text(1.1, v, f'V{v}', fontsize=10, ha='left')
        
        # Plot edges with quantum coloring
        for edge in edge_set:
            edge_color = cm.viridis(quantum_random(size=1)[0])
            ax.plot([0, 1], [edge[0], edge[1]], c=edge_color, alpha=0.5, linewidth=2)
        
        # Annotations
        ax.set_title(f'Quantum Edge Set Evolution (Frame {frame+1}/{len(edge_sets)})')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, max_coord + 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['U', 'V'])
        ax.grid(alpha=0.3)
        
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(edge_sets), interval=200, blit=False)
    ani.save('quantum_edge_evolution.gif', writer='pillow', fps=5, dpi=100)
    
    return ani


def compare_classical_quantum(deg_seq, u_size, v_size, num_samples=50):
    """Create a comparative analysis of classical vs quantum approaches."""
    # Verify sequences
    if not classical_sequence_verifier(deg_seq, u_size, v_size):
        print("Sequence not classically satisfiable")
        return
    
    if not quantum_sequence_verifier(deg_seq, u_size, v_size, False):
        print("Sequence not quantum satisfiable")
        return
    
    # Create edge sets
    classical_edges = classical_edge_assignment(deg_seq, u_size, v_size)
    quantum_edges = quantum_edge_assignment(deg_seq, u_size, v_size, False)
    
    # Generate samples
    classical_samples = []
    quantum_samples = []
    
    for _ in range(num_samples):
        classical_samples.append(classical_consensus_shuffle(classical_edges))
        quantum_samples.append(quantum_consensus_shuffle(quantum_edges))
    
    # Create comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Flatten for histograms
    classical_flat = np.vstack(classical_samples).flatten()
    quantum_flat = np.vstack(quantum_samples).flatten()
    
    # Plot histograms
    axs[0].hist(classical_flat, bins=20, alpha=0.7, color='blue')
    axs[0].set_title('Classical Edge Distribution')
    axs[0].set_xlabel('Edge Values')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(alpha=0.3)
    
    axs[1].hist(quantum_flat, bins=20, alpha=0.7, color='magenta')
    axs[1].set_title('Quantum Edge Distribution')
    axs[1].set_xlabel('Edge Values')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classical_vs_quantum_distribution.png', dpi=300)
    
    # Create animation
    animate_quantum_edge_evolution(quantum_edges)


# Main execution
if __name__ == "__main__":
    print("Quantum Graph Generator")
    print("======================")
    
    # Example degree sequence
    deg_seq = [2, 2, 2, 2, 3, 1]  # length 6
    u_size, v_size = 3, 3
    
    print("\nVerifying sequence:")
    classical_result = classical_sequence_verifier(deg_seq, u_size, v_size)
    quantum_result = quantum_sequence_verifier(deg_seq, u_size, v_size, True)
    print("Classical verification:", classical_result)
    print("Quantum verification:", quantum_result)
    
    print("\nGenerating edge assignments:")
    classical_edges = classical_edge_assignment(deg_seq, u_size, v_size)
    quantum_edges = quantum_edge_assignment(deg_seq, u_size, v_size)
    
    print("Classical edges:", classical_edges)
    print("Quantum edges:", quantum_edges)
    
    print("\nCreating visualizations...")
    try:
        compare_classical_quantum(deg_seq, u_size, v_size)
        print("\nVisualizations created:")
        print("1. classical_vs_quantum_distribution.png")
        print("2. quantum_edge_evolution.gif")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        
        # Try just creating the animation
        try:
            print("Attempting to create just the animation...")
            animate_quantum_edge_evolution(quantum_edges)
            print("Animation created: quantum_edge_evolution.gif")
        except Exception as e:
            print(f"Error creating animation: {e}")

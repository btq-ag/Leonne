# Quantum Permutation Visualizer
# Creates animations and plots for quantum permutation algorithms

"""
quantumPermutationVisualizer.py

Visualization module for quantum permutation algorithms that creates
animations and plots to demonstrate quantum permutation behavior on networks.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
from quantumGeneralizedPermutations import (
    quantumFisherYates,
    quantumPermutationOrbitFisherYates,
    quantumSymmetricNDFisherYates,
    quantumConsensusFisherYates,
    quantumEdgeShuffle,
    quantum_random_int
)


def create_bloch_sphere_animation(steps=60, filename="quantum_superposition_animation.gif"):
    """
    Creates an animation of quantum states evolving on a Bloch sphere.
    Represents the quantum superpositions used during shuffling.
    
    Parameters:
        steps: Number of animation frames
        filename: Output file name
    """
    # Set up figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Draw sphere with slight transparency
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2)
    
    # Draw axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5, linewidth=1)
    
    # Add axis labels
    ax.text(1.1, 0, 0, r'$|0\rangle$', fontsize=14)
    ax.text(-1.1, 0, 0, r'$|1\rangle$', fontsize=14)
    ax.text(0, 1.1, 0, r'$|+\rangle$', fontsize=14)
    ax.text(0, -1.1, 0, r'$|-\rangle$', fontsize=14)
    ax.text(0, 0, 1.1, r'$|+i\rangle$', fontsize=14)
    ax.text(0, 0, -1.1, r'$|-i\rangle$', fontsize=14)
    
    # Remove grid and axis ticks for cleaner look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set plot limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    # Title
    ax.set_title("Quantum State Evolution During Permutation", fontsize=16)
    
    # Initialize point and path
    point = ax.plot([0], [0], [1], 'ro', markersize=10)[0]
    path_x, path_y, path_z = [], [], []
    path_line, = ax.plot([], [], [], 'r-', alpha=0.5)
    
    # Text annotation for quantum state information
    state_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)
    step_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, fontsize=12)
    
    # Animation update function
    def update(frame):
        # Calculate new point position
        # Using a spiral path to illustrate quantum state evolution
        theta = 4 * np.pi * frame / steps
        phi = np.pi * (0.5 - frame / steps)
        
        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Update point position
        point.set_data([x], [y])
        point.set_3d_properties([z])
        
        # Update path
        path_x.append(x)
        path_y.append(y)
        path_z.append(z)
        path_line.set_data(path_x, path_y)
        path_line.set_3d_properties(path_z)
        
        # Create a quantum state representation
        # |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        psi0 = np.cos(phi/2)
        psi1 = np.exp(1j * theta) * np.sin(phi/2)
        
        # Update text
        state_text.set_text(f"|ψ⟩ = {psi0:.2f}|0⟩ + {psi1:.2f}|1⟩")
        step_text.set_text(f"Step: {frame+1}/{steps}")
        
        # Rotate view for more dynamic visualization
        ax.view_init(elev=30, azim=frame)
        
        return point, path_line, state_text, step_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
    
    # Save animation
    anim.save(filename, writer='pillow', fps=10)
    plt.close()
    
    return filename


def create_quantum_permutation_visualization(input_data=[0, 1, 2, 3, 4], 
                                           filename="quantum_permutation_visualization.gif"):
    """
    Creates an animation showing quantum permutation process with 
    Bloch sphere representation of quantum states.
    
    Parameters:
        input_data: List to be permuted
        filename: Output animation filename
    """
    # Get quantum permutation steps
    _, quantum_states = quantumFisherYates(input_data, extraInfo=False)
    steps = len(quantum_states) + 2  # Add extra frames for start/end
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
    
    # Bloch sphere on left
    ax1 = fig.add_subplot(gs[0], projection='3d')
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, color='lightblue', alpha=0.2)
    
    # Draw axes
    ax1.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax1.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax1.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5, linewidth=1)
    
    # Add axis labels
    ax1.text(1.1, 0, 0, r'$|0\rangle$', fontsize=12)
    ax1.text(-1.1, 0, 0, r'$|1\rangle$', fontsize=12)
    ax1.text(0, 0, 1.1, r'$|+\rangle$', fontsize=12)
    
    # Clean up appearance
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_zlim(-1.2, 1.2)
    ax1.set_title("Quantum State", fontsize=14)
    
    # Permutation visualization on right
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.5, len(input_data) - 0.5)
    ax2.set_ylim(-1, 5)
    ax2.axis('off')
    ax2.set_title("Quantum Permutation Process", fontsize=14)
    
    # Initialize array visualization
    array_elements = []
    array_copy = input_data.copy()
    
    for i, val in enumerate(array_copy):
        circle = plt.Circle((i, 2), 0.4, color='lightblue', alpha=0.7)
        ax2.add_patch(circle)
        text = ax2.text(i, 2, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        array_elements.append((circle, text))
    
    # Initialize quantum state visualization
    quantum_point = ax1.plot([0], [0], [1], 'ro', markersize=10)[0]
    quantum_path_x, quantum_path_y, quantum_path_z = [], [], []
    quantum_path = ax1.plot([], [], [], 'r-', alpha=0.3)[0]
    
    # Text for state information
    state_text = ax2.text(len(input_data)/2, 4, "", ha='center', fontsize=12)
    swap_text = ax2.text(len(input_data)/2, 0, "", ha='center', fontsize=12)
    
    # Swap indicator
    swap_arrows = None
    
    # Animation update function
    def update(frame):
        nonlocal swap_arrows, array_copy
        
        # Clear previous swap arrows if they exist
        if swap_arrows:
            for arrow in swap_arrows:
                arrow.remove()
            swap_arrows = None
        
        # First frame: show initial state
        if frame == 0:
            # Reset to initial state
            array_copy = input_data.copy()
            for i, val in enumerate(array_copy):
                array_elements[i][0].set_color('lightblue')
                array_elements[i][1].set_text(str(val))
            
            # Initial quantum state
            quantum_point.set_data([0], [0])
            quantum_point.set_3d_properties([1])
            quantum_path_x.clear()
            quantum_path_y.clear()
            quantum_path_z.clear()
            quantum_path.set_data([], [])
            quantum_path.set_3d_properties([])
            
            # Update text
            state_text.set_text("Initial State")
            swap_text.set_text("")
            
            return [quantum_point, quantum_path, state_text, swap_text] + [elem for pair in array_elements for elem in pair]
        
        # Last frame: show final state
        elif frame == steps - 1:
            # Show final state
            for i, val in enumerate(array_copy):
                array_elements[i][0].set_color('lightgreen')
            
            # Update text
            state_text.set_text("Quantum Permutation Complete")
            swap_text.set_text(f"Result: {array_copy}")
            
            return [quantum_point, quantum_path, state_text, swap_text] + [elem for pair in array_elements for elem in pair]
        
        # Intermediate frames: show permutation steps
        else:
            step_idx = frame - 1
            if step_idx < len(quantum_states):
                # Get quantum state and indices from stored steps
                state, (idx1, idx2) = quantum_states[step_idx]
                
                # Perform the swap in our array copy
                array_copy[idx1], array_copy[idx2] = array_copy[idx2], array_copy[idx1]
                
                # Update array visualization
                for i, val in enumerate(array_copy):
                    array_elements[i][1].set_text(str(val))
                    if i == idx1 or i == idx2:
                        array_elements[i][0].set_color('magenta')
                    else:
                        array_elements[i][0].set_color('lightblue')
                
                # Draw swap arrows
                arrow1 = ax2.arrow(idx1, 1.4, idx2-idx1, 0, head_width=0.1, head_length=0.1, 
                                  fc='red', ec='red', alpha=0.7)
                arrow2 = ax2.arrow(idx2, 1.4, idx1-idx2, 0, head_width=0.1, head_length=0.1, 
                                  fc='red', ec='red', alpha=0.7)
                swap_arrows = [arrow1, arrow2]
                
                # Update quantum state visualization
                # Convert quantum state to Bloch sphere coordinates
                if len(state) == 2:
                    # For a simple 2D state
                    theta = 2 * np.arccos(abs(state[0]))
                    phi = np.angle(state[1]) - np.angle(state[0])
                    
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                else:
                    # For higher dimensional states, project to 3D
                    # This is a simplification for visualization
                    norms = [abs(amp)**2 for amp in state]
                    total = sum(norms)
                    if total > 0:
                        weights = [n/total for n in norms]
                        # Project using weighted spherical coordinates
                        thetas = [i * np.pi / len(state) for i in range(len(state))]
                        phis = [i * 2 * np.pi / len(state) for i in range(len(state))]
                        
                        x = sum(w * np.sin(t) * np.cos(p) for w, t, p in zip(weights, thetas, phis))
                        y = sum(w * np.sin(t) * np.sin(p) for w, t, p in zip(weights, thetas, phis))
                        z = sum(w * np.cos(t) for w, t in zip(weights, thetas))
                    else:
                        x, y, z = 0, 0, 1
                
                # Update quantum point
                quantum_point.set_data([x], [y])
                quantum_point.set_3d_properties([z])
                
                # Update quantum path
                quantum_path_x.append(x)
                quantum_path_y.append(y)
                quantum_path_z.append(z)
                quantum_path.set_data(quantum_path_x, quantum_path_y)
                quantum_path.set_3d_properties(quantum_path_z)
                
                # Update text
                state_text.set_text(f"Step {step_idx+1}: Quantum Swap")
                swap_text.set_text(f"Swapping indices {idx1} and {idx2}")
                
                # Rotate Bloch sphere for better visualization
                ax1.view_init(elev=30, azim=step_idx * 10 % 360)
                
                return [quantum_point, quantum_path, state_text, swap_text, arrow1, arrow2] + [elem for pair in array_elements for elem in pair]
            
            return [quantum_point, quantum_path, state_text, swap_text] + [elem for pair in array_elements for elem in pair]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=1000, blit=True)
    
    # Save animation
    anim.save(filename, writer='pillow', fps=1)
    plt.close()
    
    return filename


def create_quantum_vs_classical_plot(input_data=[0, 1, 2, 3, 4], iterations=50000, 
                                   filename="quantum_vs_classical_distribution.png"):
    """
    Creates a plot comparing quantum and classical permutation distributions.
    
    Parameters:
        input_data: Input array to permute
        iterations: Number of iterations for distribution analysis
        filename: Output plot filename
    """
    from sympy.utilities.iterables import multiset_permutations
    from quantumGeneralizedPermutations import FisherYates
    
    # Track distributions
    permutations = list(multiset_permutations(input_data))
    quantum_counts = {tuple(p): 0 for p in permutations}
    classical_counts = {tuple(p): 0 for p in permutations}
    
    # Generate permutations and count occurrences
    for _ in range(iterations):
        # Quantum permutation
        q_result, _ = quantumFisherYates(input_data.copy(), extraInfo=False)
        quantum_counts[tuple(q_result)] += 1
        
        # Classical permutation
        c_result = FisherYates(input_data.copy(), extraInfo=False)
        classical_counts[tuple(c_result)] += 1
    
    # Convert to probabilities
    quantum_probs = [count/iterations for count in quantum_counts.values()]
    classical_probs = [count/iterations for count in classical_counts.values()]
    
    # Ideal probability (uniform distribution)
    ideal_prob = 1/len(permutations)
    
    # Calculate entropy
    def entropy(probs):
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
    
    quantum_entropy = entropy(quantum_probs)
    classical_entropy = entropy(classical_probs)
    max_entropy = np.log2(len(permutations))
    
    # Calculate statistical distance from uniform
    quantum_distance = sum(abs(p - ideal_prob) for p in quantum_probs) / 2
    classical_distance = sum(abs(p - ideal_prob) for p in classical_probs) / 2
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Distribution plot
    ax1 = fig.add_subplot(221)
    x = np.arange(len(permutations))
    width = 0.35
    ax1.bar(x - width/2, quantum_probs, width, label='Quantum', color='magenta', alpha=0.7)
    ax1.bar(x + width/2, classical_probs, width, label='Classical', color='blue', alpha=0.7)
    ax1.axhline(y=ideal_prob, color='black', linestyle='--', alpha=0.7, label='Ideal')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Permutation Index')
    ax1.set_title('Permutation Distribution Comparison')
    ax1.legend()
    
    # Log-scale deviation plot
    ax2 = fig.add_subplot(222)
    quantum_dev = [abs(p - ideal_prob) for p in quantum_probs]
    classical_dev = [abs(p - ideal_prob) for p in classical_probs]
    ax2.bar(x - width/2, quantum_dev, width, label='Quantum', color='magenta', alpha=0.7)
    ax2.bar(x + width/2, classical_dev, width, label='Classical', color='blue', alpha=0.7)
    ax2.set_ylabel('|P - Ideal|')
    ax2.set_xlabel('Permutation Index')
    ax2.set_title('Deviation from Ideal Distribution')
    ax2.set_yscale('log')
    ax2.legend()
    
    # Entropy comparison
    ax3 = fig.add_subplot(223)
    entropy_vals = [classical_entropy, quantum_entropy, max_entropy]
    entropy_labels = ['Classical', 'Quantum', 'Maximum']
    colors = ['blue', 'magenta', 'gray']
    ax3.bar(entropy_labels, entropy_vals, color=colors, alpha=0.7)
    ax3.set_ylabel('Shannon Entropy (bits)')
    ax3.set_title('Entropy Comparison')
    ax3.set_ylim(0, max_entropy * 1.1)
    
    for i, v in enumerate(entropy_vals):
        ax3.text(i, v + 0.05, f'{v:.4f}', ha='center')
    
    # Statistical distance comparison
    ax4 = fig.add_subplot(224)
    distance_vals = [classical_distance, quantum_distance]
    distance_labels = ['Classical', 'Quantum']
    ax4.bar(distance_labels, distance_vals, color=colors[:2], alpha=0.7)
    ax4.set_ylabel('Statistical Distance from Uniform')
    ax4.set_title('Uniformity Comparison (lower is better)')
    ax4.set_ylim(0, max(distance_vals) * 1.5)
    
    for i, v in enumerate(distance_vals):
        ax4.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Annotation with statistics
    quantum_advantage = (quantum_entropy / classical_entropy - 1) * 100
    uniformity_improvement = (classical_distance / quantum_distance - 1) * 100
    
    plt.figtext(0.5, 0.01, 
               f"Quantum Advantage: {quantum_advantage:.2f}% higher entropy\n"
               f"Uniformity Improvement: {uniformity_improvement:.2f}% better approximation to uniform distribution\n"
               f"Input: {input_data}, {iterations:,} iterations", 
               ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return filename


def create_quantum_edge_shuffle_animation(edge_set=None, filename="quantum_edge_shuffle_animation.gif"):
    """
    Creates an animation demonstrating quantum edge shuffling for bipartite graphs.
    
    Parameters:
        edge_set: Input edge set to shuffle
        filename: Output animation filename
    """
    # Default edge set if none provided
    if edge_set is None:
        edge_set = np.array([[1, 2], [2, 1], [2, 3], [3, 2], [4, 4]])
    
    # Get quantum edge shuffle steps
    final_edges, final_left, quantum_states = quantumEdgeShuffle(edge_set, extraInfo=False)
    
    # Set up figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3])
    
    # Top left: quantum state
    ax_state = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax_state.plot_surface(x, y, z, color='lightblue', alpha=0.2)
    ax_state.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax_state.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax_state.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5, linewidth=1)
    
    ax_state.set_xticks([])
    ax_state.set_yticks([])
    ax_state.set_zticks([])
    ax_state.set_xlim(-1.2, 1.2)
    ax_state.set_ylim(-1.2, 1.2)
    ax_state.set_zlim(-1.2, 1.2)
    ax_state.set_title("Quantum State", fontsize=12)
    
    # Top right: step info
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    step_text = ax_info.text(0.5, 0.5, "", ha='center', va='center', fontsize=14)
    
    # Bottom: bipartite graph visualization
    ax_graph = fig.add_subplot(gs[1, :])
    ax_graph.set_xlim(-1, 10)
    ax_graph.set_ylim(-1, 5)
    ax_graph.axis('off')
    ax_graph.set_title("Quantum Edge Shuffle Evolution", fontsize=14)
    
    # Number of nodes on each side
    left_nodes = np.unique(edge_set[:, 0])
    right_nodes = np.unique(edge_set[:, 1])
    
    # Initialize visualization elements
    edge_lines = []
    left_circles = []
    right_circles = []
    
    # Draw initial bipartite graph
    edges_copy = edge_set.copy()
    
    # Position nodes
    left_pos = {node: (1, 4-i) for i, node in enumerate(left_nodes)}
    right_pos = {node: (8, 4-i) for i, node in enumerate(right_nodes)}
    
    # Draw left nodes
    for node in left_nodes:
        x, y = left_pos[node]
        circle = plt.Circle((x, y), 0.4, color='lightblue', alpha=0.7)
        ax_graph.add_patch(circle)
        ax_graph.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold')
        left_circles.append((node, circle))
    
    # Draw right nodes
    for node in right_nodes:
        x, y = right_pos[node]
        circle = plt.Circle((x, y), 0.4, color='lightgreen', alpha=0.7)
        ax_graph.add_patch(circle)
        ax_graph.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold')
        right_circles.append((node, circle))
    
    # Draw edges
    for edge in edges_copy:
        left_node, right_node = edge
        x1, y1 = left_pos[left_node]
        x2, y2 = right_pos[right_node]
        line, = ax_graph.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
        edge_lines.append((edge, line))
    
    # Initialize quantum state point
    quantum_point = ax_state.plot([0], [0], [1], 'ro', markersize=10)[0]
    quantum_path_x, quantum_path_y, quantum_path_z = [], [], []
    quantum_path = ax_state.plot([], [], [], 'r-', alpha=0.3)[0]
    
    # Number of frames in animation
    steps = len(quantum_states) + 2  # Add frames for start/end
    
    # Animation update function
    def update(frame):
        nonlocal edges_copy
        
        # First frame: show initial state
        if frame == 0:
            # Reset to initial state
            edges_copy = edge_set.copy()
            
            # Update edge lines
            for edge_data, line in edge_lines:
                left_node, right_node = edge_data
                x1, y1 = left_pos[left_node]
                x2, y2 = right_pos[right_node]
                line.set_data([x1, x2], [y1, y2])
                line.set_color('black')
                line.set_alpha(0.5)
            
            # Reset node colors
            for _, circle in left_circles:
                circle.set_color('lightblue')
            for _, circle in right_circles:
                circle.set_color('lightgreen')
            
            # Reset quantum state
            quantum_point.set_data([0], [0])
            quantum_point.set_3d_properties([1])
            quantum_path_x.clear()
            quantum_path_y.clear()
            quantum_path_z.clear()
            quantum_path.set_data([], [])
            quantum_path.set_3d_properties([])
            
            # Update text
            step_text.set_text("Initial Edge Configuration")
            
            return [quantum_point, quantum_path, step_text] + [line for _, line in edge_lines] + [circle for _, circle in left_circles] + [circle for _, circle in right_circles]
        
        # Last frame: show final state
        elif frame == steps - 1:
            # Update edges with final configuration
            edges_copy = final_edges
            
            # Update edge lines
            for i, (edge_data, line) in enumerate(edge_lines):
                if i < len(final_edges):
                    left_node, right_node = final_edges[i]
                    x1, y1 = left_pos[left_node]
                    x2, y2 = right_pos[right_node]
                    line.set_data([x1, x2], [y1, y2])
                    line.set_color('blue')
                    line.set_alpha(0.8)
            
            # Highlight left column nodes in final state
            for node, circle in left_circles:
                if node in final_left:
                    circle.set_color('magenta')
                else:
                    circle.set_color('lightblue')
            
            # Update text
            step_text.set_text("Final Edge Configuration\n(Quantum Shuffled)")
            
            return [quantum_point, quantum_path, step_text] + [line for _, line in edge_lines] + [circle for _, circle in left_circles] + [circle for _, circle in right_circles]
        
        # Intermediate frames: show shuffle steps
        else:
            step_idx = frame - 1
            if step_idx < len(quantum_states):
                # Get quantum state and indices from stored steps
                state, (idx1, idx2) = quantum_states[step_idx]
                
                # Update edges with intermediate configuration
                # Swap the left column values for edges at idx1 and idx2
                left_col = edges_copy[:, 0].tolist()
                right_col = edges_copy[:, 1].tolist()
                left_col[idx1], left_col[idx2] = left_col[idx2], left_col[idx1]
                edges_copy = np.column_stack((left_col, right_col))
                
                # Update edge lines
                for i, (_, line) in enumerate(edge_lines):
                    if i < len(edges_copy):
                        left_node, right_node = edges_copy[i]
                        x1, y1 = left_pos[left_node]
                        x2, y2 = right_pos[right_node]
                        line.set_data([x1, x2], [y1, y2])
                        if i == idx1 or i == idx2:
                            line.set_color('red')
                            line.set_alpha(0.8)
                        else:
                            line.set_color('black')
                            line.set_alpha(0.5)
                
                # Highlight swapped nodes
                for node, circle in left_circles:
                    if node in [edges_copy[idx1, 0], edges_copy[idx2, 0]]:
                        circle.set_color('magenta')
                    else:
                        circle.set_color('lightblue')
                
                # Update quantum state visualization
                if len(state) == 2:
                    # For a simple 2D state
                    theta = 2 * np.arccos(abs(state[0]) if abs(state[0]) <= 1 else 1)
                    phi = np.angle(state[1]) - np.angle(state[0])
                    
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                else:
                    # For higher dimensional states, project to 3D
                    norms = [abs(amp)**2 for amp in state]
                    total = sum(norms)
                    if total > 0:
                        weights = [n/total for n in norms]
                        thetas = [i * np.pi / len(state) for i in range(len(state))]
                        phis = [i * 2 * np.pi / len(state) for i in range(len(state))]
                        
                        x = sum(w * np.sin(t) * np.cos(p) for w, t, p in zip(weights, thetas, phis))
                        y = sum(w * np.sin(t) * np.sin(p) for w, t, p in zip(weights, thetas, phis))
                        z = sum(w * np.cos(t) for w, t in zip(weights, thetas))
                    else:
                        x, y, z = 0, 0, 1
                
                # Update quantum point
                quantum_point.set_data([x], [y])
                quantum_point.set_3d_properties([z])
                
                # Update quantum path
                quantum_path_x.append(x)
                quantum_path_y.append(y)
                quantum_path_z.append(z)
                quantum_path.set_data(quantum_path_x, quantum_path_y)
                quantum_path.set_3d_properties(quantum_path_z)
                
                # Update text
                step_text.set_text(f"Step {step_idx+1}: Edge Swap\nSwapping edges at indices {idx1} and {idx2}")
                
                # Rotate Bloch sphere for better visualization
                ax_state.view_init(elev=30, azim=step_idx * 15 % 360)
                
                return [quantum_point, quantum_path, step_text] + [line for _, line in edge_lines] + [circle for _, circle in left_circles] + [circle for _, circle in right_circles]
            
            return [quantum_point, quantum_path, step_text] + [line for _, line in edge_lines] + [circle for _, circle in left_circles] + [circle for _, circle in right_circles]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=1000, blit=True)
    
    # Save animation
    anim.save(filename, writer='pillow', fps=1)
    plt.close()
    
    return filename


def main():
    """Main function to generate all visualizations."""
    print("Generating Quantum Permutation Visualizations...")
    
    # 1. Create Bloch sphere animation
    print("1. Creating quantum superposition animation...")
    bloch_animation = create_bloch_sphere_animation(steps=60, 
                                                  filename="quantum_superposition_animation.gif")
    
    # 2. Create quantum permutation visualization
    print("2. Creating quantum permutation animation...")
    perm_animation = create_quantum_permutation_visualization(
        input_data=[0, 1, 2, 3, 4],
        filename="quantum_permutation_visualization.gif"
    )
    
    # 3. Create quantum vs classical distribution plot
    print("3. Creating quantum vs classical distribution plot...")
    dist_plot = create_quantum_vs_classical_plot(
        input_data=[0, 1, 2, 3, 4],
        iterations=10000,  # Lower for demo purposes
        filename="quantum_vs_classical_distribution.png"
    )
    
    # 4. Create quantum edge shuffle animation
    print("4. Creating quantum edge shuffle animation...")
    edge_animation = create_quantum_edge_shuffle_animation(
        edge_set=np.array([[1, 2], [2, 1], [2, 3], [3, 2], [4, 4]]),
        filename="quantum_edge_shuffle_animation.gif"
    )
    
    print("\nAll visualizations complete.")
    print(f"Files generated:\n- {bloch_animation}\n- {perm_animation}\n- {dist_plot}\n- {edge_animation}")
    
    return {
        'superposition_animation': bloch_animation,
        'permutation_animation': perm_animation,
        'distribution_plot': dist_plot,
        'edge_shuffle_animation': edge_animation
    }


if __name__ == "__main__":
    main()

"""
Enhanced Quantum Graph Visualizer

This module implements advanced visualizations for quantum-inspired graph generators for bipartite graphs.
It provides tools for visualizing quantum network evolution, stability analysis, and comparative visualizations.
Inspired by classical visualizations but enhanced to showcase quantum-specific properties.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import networkx as nx
import os
from tqdm import tqdm
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# Set output directory to current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define colors for different node types and quantum states
COLORS = {
    'u_nodes': '#2196F3',           # Blue for U nodes
    'v_nodes': '#FFC107',           # Yellow for V nodes
    'entangled': '#9C27B0',         # Purple for entangled
    'superposition': '#4CAF50',     # Green for superposition
    'decoherence': '#F44336',       # Red for decoherence
    'edge': '#555555',              # Gray for edges
    'quantum_edge': '#00BCD4',      # Cyan for quantum edges
    'highlight': '#FF5722'          # Orange for highlighting
}

# Import functions from quantumGraphGenerator - ensure these functions are available
from quantumGraphGenerator import (
    quantum_random, quantum_choice, 
    quantumConsensusShuffle, 
    quantumSequenceVerifier as quantum_sequence_verifier, 
    quantumEdgeAssignment as quantum_edge_assignment
)


def create_quantum_network_evolution_animation(degree_sequence, u_size, v_size, n_frames=30, 
                                              filename="quantum_network_evolution_animation",
                                              quantum_method='phase'):
    """
    Create an animation showing quantum network evolution under repeated consensus shuffles.
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_frames : int
        Number of animation frames
    filename : str
        Base filename for the output animation
    quantum_method : str
        Method for quantum randomness ('hadamard', 'phase', 'bloch')
    """
    # Verify the sequence and get initial edge assignment
    if not quantum_sequence_verifier(degree_sequence, u_size, v_size, False):
        print("Invalid quantum degree sequence. Cannot create animation.")
        return
    
    # Get initial edge assignment
    initial_edges = quantum_edge_assignment(degree_sequence, u_size, v_size, False)
    
    if initial_edges is False:
        print("Failed to generate initial edges. Cannot create animation.")
        return
    
    # Determine figure size based on network size
    base_size = 10
    size_factor = max(1, np.log2(max(u_size, v_size)) / 2)
    fig_size = (base_size * size_factor, base_size * size_factor)
    
    # Set up the figure with a quantum-inspired dark background
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#0a0a14')
    ax.set_facecolor('#0a0a14')
    
    # Generate a series of network states through quantum consensus shuffling
    edge_sets = [initial_edges]
    current_edges = initial_edges.copy()
    quantum_states = [np.random.random(len(initial_edges))]  # Track quantum state for coloring
    
    print("Generating quantum network evolution frames...")
    for _ in tqdm(range(n_frames - 1)):
        current_edges = quantumConsensusShuffle(current_edges, False, quantum_method).tolist()
        edge_sets.append(current_edges)
        
        # Generate quantum state values for edge coloring
        # This simulates quantum effects on edges (entanglement, superposition)
        new_states = quantum_states[-1] * 0.7 + np.random.random(len(current_edges)) * 0.3
        quantum_states.append(new_states)
    
    # Create a function to convert edge sets to networkx graphs
    def create_graph_from_edges(edges):
        G = nx.DiGraph()  # Using directed graph to show edge directionality
        
        # Add all nodes
        for u in range(u_size):
            G.add_node(f'u{u}', type='u')
        for v in range(v_size):
            G.add_node(f'v{v}', type='v')
        
        # Add edges
        for u, v in edges:
            G.add_edge(f'u{u}', f'v{v}')
        
        return G
    
    # Create initial graph and set positions
    G_initial = create_graph_from_edges(edge_sets[0])
    
    # Use a circular layout with U nodes on left, V nodes on right
    pos = {}
    
    # Calculate node size based on network size
    node_size = max(100, min(500, 1000 / np.sqrt(u_size + v_size)))
    font_size = max(6, min(12, 18 / np.sqrt(u_size + v_size)))
    
    # Position U nodes in a semicircle on the left
    radius_factor = 1.0 + (max(u_size, v_size) / 10)
    for i in range(u_size):
        angle = np.pi/2 + (i * np.pi) / (u_size - 1) if u_size > 1 else np.pi
        pos[f'u{i}'] = np.array([-np.cos(angle) * radius_factor, np.sin(angle) * radius_factor])
    
    # Position V nodes in a semicircle on the right
    for i in range(v_size):
        angle = np.pi/2 + (i * np.pi) / (v_size - 1) if v_size > 1 else np.pi
        pos[f'v{i}'] = np.array([np.cos(angle) * radius_factor, np.sin(angle) * radius_factor])
    
    # Create a custom colormap for quantum effects
    quantum_cmap = LinearSegmentedColormap.from_list(
        'quantum_cmap', 
        [COLORS['decoherence'], COLORS['superposition'], COLORS['entangled']], 
        N=256
    )
    
    # Animation update function
    def update(frame):
        ax.clear()
        ax.set_facecolor('#0a0a14')
        
        # Create graph for current frame
        G = create_graph_from_edges(edge_sets[frame])
        
        # Draw background glow for quantum effects
        quantum_phase = (frame / n_frames) * 2 * np.pi
        background_glow = 0.05 + 0.03 * np.sin(quantum_phase)
        ax.set_facecolor((background_glow, background_glow, background_glow*1.5, 1.0))
        
        # Draw nodes with quantum-inspired glow
        u_node_color = to_rgba(COLORS['u_nodes'])
        v_node_color = to_rgba(COLORS['v_nodes'])
        
        # Add quantum effect to node colors - subtle pulsing
        u_alpha = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(quantum_phase + np.pi/3))
        v_alpha = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(quantum_phase + 2*np.pi/3))
        
        u_color = (u_node_color[0], u_node_color[1], u_node_color[2], u_alpha)
        v_color = (v_node_color[0], v_node_color[1], v_node_color[2], v_alpha)
        
        # Draw U nodes
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes if n.startswith('u')],
                              node_color=[u_color]*u_size,
                              node_size=node_size,
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax)
        
        # Draw V nodes
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes if n.startswith('v')],
                              node_color=[v_color]*v_size,
                              node_size=node_size,
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax)
        
        # Draw edges with quantum-inspired effects
        edge_colors = []
        for i, edge in enumerate(G.edges()):
            if i < len(quantum_states[frame]):
                # Color edge based on its quantum state
                edge_colors.append(quantum_cmap(quantum_states[frame][i]))
            else:
                edge_colors.append(COLORS['edge'])
        
        # Draw edges with arrows and varying widths based on quantum states
        edge_widths = []
        for i in range(len(G.edges())):
            if i < len(quantum_states[frame]):
                # Width based on quantum state - thicker for more "quantum" edges
                edge_widths.append(0.5 + 1.5 * quantum_states[frame][i])
            else:
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(G, pos,
                              arrowstyle='-|>',
                              arrowsize=10,
                              width=edge_widths,
                              edge_color=edge_colors,
                              alpha=0.7,
                              connectionstyle='arc3,rad=0.1',  # Curved edges
                              ax=ax)
        
        # Draw labels with quantum-inspired glow
        nx.draw_networkx_labels(G, pos, font_size=font_size, 
                              font_color='white', font_weight='bold', 
                              font_family='monospace', ax=ax)
        
        # Add quantum method and frame info in corner
        quantum_info = f"Quantum Method: {quantum_method}"
        ax.text(0.02, 0.02, quantum_info, transform=ax.transAxes, 
               color='white', fontsize=10, alpha=0.7,
               bbox=dict(facecolor='black', alpha=0.2, boxstyle='round'))
        
        frame_info = f"Frame {frame+1}/{n_frames}"
        ax.text(0.98, 0.02, frame_info, transform=ax.transAxes, 
               color='white', fontsize=10, alpha=0.7, ha='right',
               bbox=dict(facecolor='black', alpha=0.2, boxstyle='round'))
        
        # Add a descriptive title
        ax.set_title(f"Quantum Network Evolution - {quantum_method.capitalize()} Method", 
                    color='white', fontsize=14, pad=20)
        
        # Remove axis ticks and labels for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        
        return ax
    
    # Create animation with progress bar
    print(f"Creating animation with {n_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    
    # Save animation
    print(f"Saving animation to {filename}.gif...")
    ani.save(f"{filename}.gif", writer=PillowWriter(fps=5), dpi=120)
    plt.close()
    
    print(f"Animation saved successfully to {filename}.gif")
    return ani


def visualize_quantum_permutation_stability(degree_sequence, u_size, v_size, n_samples=50, 
                                          filename="quantum_permutation_stability",
                                          quantum_methods=['hadamard', 'phase', 'bloch']):
    """
    Visualize the stability of quantum permutations compared to classical ones
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_samples : int
        Number of permutation samples to generate
    filename : str
        Base filename for the output visualization
    quantum_methods : list
        List of quantum methods to compare
    """
    if not quantum_sequence_verifier(degree_sequence, u_size, v_size, False):
        print("Invalid quantum degree sequence. Cannot create stability visualization.")
        return

    # Get initial edge assignment
    initial_edges = quantum_edge_assignment(degree_sequence, u_size, v_size, False)
    
    if initial_edges is False:
        print("Failed to generate initial edges. Cannot create stability visualization.")
        return
    
    # Setup the figure for stability plot - 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f5f5f5')
    fig.suptitle('Quantum vs Classical Permutation Stability Analysis', fontsize=18, y=0.98)
    
    # Flatten for easier indexing
    axs = axs.flatten()
    
    # Generate samples for each method
    print("Generating permutation samples...")
    
    # Classical method (for comparison)
    classical_samples = []
    for _ in tqdm(range(n_samples), desc="Classical"):
        # Simulate classical shuffle without quantum effects
        edges = initial_edges.copy()
        for _ in range(5):  # Multiple shuffles to explore the space
            u = np.array(edges)[:, 0].tolist()
            v = np.array(edges)[:, 1].tolist()
            
            for i in reversed(range(1, len(u))):
                t = [i]
                for k in range(0, i):
                    if ([u[i], v[k]] not in edges) and ([u[k], v[i]] not in edges):
                        t.append(k)
                j = np.random.choice(t)
                u[i], u[j] = u[j], u[i]
            
            edges = np.column_stack((u, v)).tolist()
        
        classical_samples.append(np.array(edges).flatten())
    
    # Classical stability plot
    axs[0].set_title('Classical Permutation Distribution', fontsize=14)
    classical_matrix = np.vstack(classical_samples)
    
    # Calculate stability metrics
    classical_mean = np.mean(classical_matrix, axis=0)
    classical_std = np.std(classical_matrix, axis=0)
    classical_stability = 1 - (classical_std / (np.max(classical_matrix) - np.min(classical_matrix) + 1e-10))
    
    # Plot classical stability
    positions = np.arange(len(classical_mean))
    axs[0].bar(positions, classical_stability, color='#3f51b5', alpha=0.7)
    axs[0].set_xlabel('Edge Position Index', fontsize=12)
    axs[0].set_ylabel('Stability (1 - normalized std dev)', fontsize=12)
    axs[0].grid(alpha=0.3)
    axs[0].text(0.02, 0.95, f'Avg Stability: {np.mean(classical_stability):.4f}', 
               transform=axs[0].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    # For each quantum method
    for i, method in enumerate(quantum_methods):
        quantum_samples = []
        for _ in tqdm(range(n_samples), desc=f"Quantum ({method})"):
            # Use the quantum shuffle function
            edges = initial_edges.copy()
            for _ in range(5):  # Multiple shuffles to explore the space
                edges = quantumConsensusShuffle(edges, False, method).tolist()
            quantum_samples.append(np.array(edges).flatten())
        
        # Quantum stability plot
        axs[i+1].set_title(f'Quantum Permutation Distribution ({method})', fontsize=14)
        quantum_matrix = np.vstack(quantum_samples)
        
        # Calculate stability metrics
        quantum_mean = np.mean(quantum_matrix, axis=0)
        quantum_std = np.std(quantum_matrix, axis=0)
        quantum_stability = 1 - (quantum_std / (np.max(quantum_matrix) - np.min(quantum_matrix) + 1e-10))
        
        # Plot quantum stability
        quantum_color = COLORS['entangled'] if method == 'bloch' else (
            COLORS['superposition'] if method == 'phase' else COLORS['decoherence'])
        
        axs[i+1].bar(positions, quantum_stability, color=quantum_color, alpha=0.7)
        axs[i+1].set_xlabel('Edge Position Index', fontsize=12)
        axs[i+1].set_ylabel('Stability (1 - normalized std dev)', fontsize=12)
        axs[i+1].grid(alpha=0.3)
        axs[i+1].text(0.02, 0.95, f'Avg Stability: {np.mean(quantum_stability):.4f}', 
                     transform=axs[i+1].transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save high-resolution figure
    print(f"Saving stability visualization to {filename}.png...")
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stability visualization saved successfully to {filename}.png")


def create_3d_quantum_network_animation(degree_sequence, u_size, v_size, n_frames=30, 
                                       filename="quantum_network_3d_animation",
                                       quantum_method='bloch'):
    """
    Create a 3D animation of quantum network evolution showing quantum effects
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_frames : int
        Number of animation frames
    filename : str
        Base filename for the output animation
    quantum_method : str
        Method for quantum randomness ('hadamard', 'phase', 'bloch')
    """
    # Verify the sequence and get initial edge assignment
    if not quantum_sequence_verifier(degree_sequence, u_size, v_size, False):
        print("Invalid quantum degree sequence. Cannot create 3D animation.")
        return
    
    # Get initial edge assignment
    initial_edges = quantum_edge_assignment(degree_sequence, u_size, v_size, False)
    
    if initial_edges is False:
        print("Failed to generate initial edges. Cannot create 3D animation.")
        return
    
    # Setup figure for 3D visualization
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Generate a series of network states through quantum consensus shuffling
    edge_sets = [initial_edges]
    current_edges = initial_edges.copy()
    
    # Generate quantum state data for visualization
    quantum_phases = np.zeros((n_frames, len(initial_edges)))
    quantum_amplitudes = np.zeros((n_frames, len(initial_edges)))
    
    print("Generating 3D quantum network states...")
    for i in tqdm(range(n_frames - 1)):
        current_edges = quantumConsensusShuffle(current_edges, False, quantum_method).tolist()
        edge_sets.append(current_edges)
        
        # Generate quantum data for this frame
        # Phase values (0 to 2π)
        quantum_phases[i+1] = quantum_phases[i] + np.random.uniform(0, np.pi/8, len(current_edges))
        # Amplitude values (0 to 1)
        quantum_amplitudes[i+1] = 0.8 * quantum_amplitudes[i] + 0.2 * np.random.random(len(current_edges))
    
    # Set node positions in 3D space
    pos_3d = {}
    
    # Position U nodes in a partial sphere on the left
    for i in range(u_size):
        theta = np.pi * (0.25 + 0.5 * (i / max(1, u_size - 1)))
        phi = np.pi * (0.25 + 0.5 * (i / max(1, u_size - 1)))
        x = -1 + 0.5 * np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pos_3d[f'u{i}'] = (x, y, z)
    
    # Position V nodes in a partial sphere on the right
    for i in range(v_size):
        theta = np.pi * (0.25 + 0.5 * (i / max(1, v_size - 1)))
        phi = np.pi * (0.25 + 0.5 * (i / max(1, v_size - 1)))
        x = 1 - 0.5 * np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pos_3d[f'v{i}'] = (x, y, z)
    
    # Animation update function
    def update_3d(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Set axis limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        
        # Remove grid and axis labels for cleaner look
        ax.grid(False)
        ax.set_axis_off()
        
        # Add title with quantum method
        ax.set_title(f"3D Quantum Network - {quantum_method.capitalize()} Method (Frame {frame+1}/{n_frames})", 
                    color='white', pad=20, fontsize=14)
        
        # Draw nodes with quantum glow effect
        for i in range(u_size):
            # U nodes in blue with quantum fluctuation
            node_color = to_rgba(COLORS['u_nodes'])
            glow = 0.5 + 0.5 * np.sin(frame/n_frames * 2*np.pi + i*0.5)
            size = 80 + 40 * glow
            
            x, y, z = pos_3d[f'u{i}']
            ax.scatter(x, y, z, s=size, color=node_color, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add node label
            ax.text(x*1.1, y*1.1, z*1.1, f'U{i}', color='white', fontsize=10)
        
        for i in range(v_size):
            # V nodes in yellow with quantum fluctuation
            node_color = to_rgba(COLORS['v_nodes'])
            glow = 0.5 + 0.5 * np.sin(frame/n_frames * 2*np.pi + i*0.5 + np.pi)
            size = 80 + 40 * glow
            
            x, y, z = pos_3d[f'v{i}']
            ax.scatter(x, y, z, s=size, color=node_color, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add node label
            ax.text(x*1.1, y*1.1, z*1.1, f'V{i}', color='white', fontsize=10)
        
        # Draw edges with quantum-inspired effects
        edges = edge_sets[frame]
        
        for i, (u, v) in enumerate(edges):
            u_pos = pos_3d[f'u{u}']
            v_pos = pos_3d[f'v{v}']
            
            # Create points along the edge for quantum wave effect
            num_points = 20
            x = np.linspace(u_pos[0], v_pos[0], num_points)
            y = np.linspace(u_pos[1], v_pos[1], num_points)
            z = np.linspace(u_pos[2], v_pos[2], num_points)
            
            # Add quantum wave modulation based on phase
            if i < len(quantum_phases[frame]):
                phase = quantum_phases[frame, i]
                amplitude = quantum_amplitudes[frame, i] * 0.2
                
                # Add quantum wave effect to edge
                z_mod = z + amplitude * np.sin(phase + np.linspace(0, 2*np.pi, num_points))
                
                # Color based on quantum method
                if quantum_method == 'bloch':
                    edge_color = cm.cool(amplitude * 5)
                elif quantum_method == 'phase':
                    edge_color = cm.plasma(phase / (2*np.pi))
                else:
                    edge_color = cm.viridis(amplitude * 5)
                
                # Plot edge with variable width
                ax.plot(x, y, z_mod, color=edge_color, linewidth=1.5 + 2*amplitude, alpha=0.7)
                
                # Add probability cloud around edge (if high quantum amplitude)
                if amplitude > 0.1:
                    for _ in range(3):
                        # Small random offset for cloud effect
                        cloud_offset = np.random.normal(0, 0.05, (num_points, 3))
                        cloud_x = x + cloud_offset[:, 0]
                        cloud_y = y + cloud_offset[:, 1]
                        cloud_z = z_mod + cloud_offset[:, 2]
                        
                        # Plot probability cloud with low opacity
                        ax.plot(cloud_x, cloud_y, cloud_z, color=edge_color, 
                               linewidth=0.5, alpha=0.15)
            else:
                # Fallback for any extra edges
                ax.plot(x, y, z, color=COLORS['edge'], linewidth=1, alpha=0.5)
        
        # Rotate view for dynamic perspective
        ax.view_init(elev=20 + 10*np.sin(frame/n_frames * 2*np.pi), 
                    azim=frame * (360/n_frames) % 360)
        
        return ax
    
    # Create animation
    print(f"Creating 3D animation with {n_frames} frames...")
    ani = animation.FuncAnimation(fig, update_3d, frames=n_frames, blit=False)
    
    # Save animation
    print(f"Saving 3D animation to {filename}.gif...")
    ani.save(f"{filename}.gif", writer=PillowWriter(fps=5), dpi=120)
    plt.close()
    
    print(f"3D animation saved successfully to {filename}.gif")
    return ani


def compare_quantum_classical_evolution(degree_sequence, u_size, v_size, n_frames=15, 
                                       filename="quantum_vs_classical_evolution"):
    """
    Create a side-by-side animation comparing classical and quantum evolution
    
    Parameters:
    -----------
    degree_sequence : list
        The degree sequence [u1, u2, ..., v1, v2, ...]
    u_size : int
        Number of nodes in the U set
    v_size : int
        Number of nodes in the V set
    n_frames : int
        Number of animation frames
    filename : str
        Base filename for the output animation
    """
    # Verify sequences and get initial edge assignments
    if not quantum_sequence_verifier(degree_sequence, u_size, v_size, False):
        print("Invalid degree sequence. Cannot create comparison animation.")
        return
    
    # Get initial edge assignments - same starting point for fair comparison
    initial_edges = quantum_edge_assignment(degree_sequence, u_size, v_size, False)
    
    if initial_edges is False:
        print("Failed to generate initial edges. Cannot create comparison animation.")
        return
    
    # Set up the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#f8f8f8')
    fig.subplots_adjust(wspace=0.3)
    
    # Set titles for each subplot
    ax1.set_title("Classical Network Evolution", fontsize=16, pad=20)
    ax2.set_title("Quantum Network Evolution", fontsize=16, pad=20)
    
    # Generate edge set sequences
    classical_edge_sets = [initial_edges]
    quantum_edge_sets = [initial_edges]
    
    # Track quantum states for coloring
    quantum_states = [np.random.random(len(initial_edges))]
    
    current_classical = initial_edges.copy()
    current_quantum = initial_edges.copy()
    
    print("Generating comparison frames...")
    for _ in tqdm(range(n_frames - 1)):
        # Classical evolution
        u = np.array(current_classical)[:, 0].tolist()
        v = np.array(current_classical)[:, 1].tolist()
        
        for i in reversed(range(1, len(u))):
            t = [i]
            for k in range(0, i):
                if ([u[i], v[k]] not in current_classical) and ([u[k], v[i]] not in current_classical):
                    t.append(k)
            j = np.random.choice(t)
            u[i], u[j] = u[j], u[i]
        
        current_classical = np.column_stack((u, v)).tolist()
        classical_edge_sets.append(current_classical)
        
        # Quantum evolution
        current_quantum = quantumConsensusShuffle(current_quantum, False, 'bloch').tolist()
        quantum_edge_sets.append(current_quantum)
        
        # Update quantum states
        new_states = quantum_states[-1] * 0.7 + np.random.random(len(current_quantum)) * 0.3
        quantum_states.append(new_states)
    
    # Create networkx graphs and positions
    def create_graph_from_edges(edges, u_size, v_size):
        G = nx.DiGraph()
        
        for u in range(u_size):
            G.add_node(f'u{u}', type='u')
        for v in range(v_size):
            G.add_node(f'v{v}', type='v')
        
        for u, v in edges:
            G.add_edge(f'u{u}', f'v{v}')
        
        return G
    
    # Create positions for nodes (same for both graphs)
    pos = {}
    
    # Position U nodes in a semicircle on the left
    for i in range(u_size):
        angle = np.pi/2 + (i * np.pi) / (u_size - 1) if u_size > 1 else np.pi
        pos[f'u{i}'] = np.array([-np.cos(angle), np.sin(angle)])
    
    # Position V nodes in a semicircle on the right
    for i in range(v_size):
        angle = np.pi/2 + (i * np.pi) / (v_size - 1) if v_size > 1 else np.pi
        pos[f'v{i}'] = np.array([np.cos(angle), np.sin(angle)])
    
    # Node size based on network size
    node_size = max(100, min(500, 1000 / np.sqrt(u_size + v_size)))
    font_size = max(8, min(12, 16 / np.sqrt(u_size + v_size)))
    
    # Create a custom colormap for quantum effects
    quantum_cmap = LinearSegmentedColormap.from_list(
        'quantum_cmap', 
        [COLORS['decoherence'], COLORS['superposition'], COLORS['entangled']], 
        N=256
    )
    
    # Animation update function
    def update_comparison(frame):
        ax1.clear()
        ax2.clear()
        
        # Set consistent style for both plots
        for ax in [ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        
        # Update frame counter in figure
        frame_info = f"Frame {frame+1}/{n_frames}"
        fig.suptitle(frame_info, fontsize=14, y=0.98)
        
        # Draw classical network (left)
        G_classical = create_graph_from_edges(classical_edge_sets[frame], u_size, v_size)
        
        nx.draw_networkx_nodes(G_classical, pos,
                              nodelist=[n for n in G_classical.nodes if n.startswith('u')],
                              node_color=COLORS['u_nodes'],
                              node_size=node_size,
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax1)
        
        nx.draw_networkx_nodes(G_classical, pos,
                              nodelist=[n for n in G_classical.nodes if n.startswith('v')],
                              node_color=COLORS['v_nodes'],
                              node_size=node_size,
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax1)
        
        nx.draw_networkx_edges(G_classical, pos,
                              arrowstyle='-|>',
                              arrowsize=10,
                              width=1.0,
                              edge_color=COLORS['edge'],
                              alpha=0.7,
                              ax=ax1)
        
        nx.draw_networkx_labels(G_classical, pos, 
                              font_size=font_size, 
                              font_weight='bold',
                              ax=ax1)
        
        # Draw quantum network (right)
        G_quantum = create_graph_from_edges(quantum_edge_sets[frame], u_size, v_size)
        
        # Draw nodes with quantum glow effect
        glow_factor = 0.5 + 0.5 * np.sin(frame/n_frames * 2*np.pi)
        
        nx.draw_networkx_nodes(G_quantum, pos,
                              nodelist=[n for n in G_quantum.nodes if n.startswith('u')],
                              node_color=COLORS['u_nodes'],
                              node_size=node_size * (1 + 0.2 * glow_factor),
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax2)
        
        nx.draw_networkx_nodes(G_quantum, pos,
                              nodelist=[n for n in G_quantum.nodes if n.startswith('v')],
                              node_color=COLORS['v_nodes'],
                              node_size=node_size * (1 + 0.2 * glow_factor),
                              edgecolors='white',
                              linewidths=0.5,
                              ax=ax2)
        
        # Quantum edges with color and width based on quantum state
        edge_colors = []
        edge_widths = []
        
        for i in range(len(G_quantum.edges())):
            if i < len(quantum_states[frame]):
                edge_colors.append(quantum_cmap(quantum_states[frame][i]))
                edge_widths.append(0.5 + 1.5 * quantum_states[frame][i])
            else:
                edge_colors.append(COLORS['edge'])
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(G_quantum, pos,
                              arrowstyle='-|>',
                              arrowsize=10,
                              width=edge_widths,
                              edge_color=edge_colors,
                              alpha=0.7,
                              connectionstyle='arc3,rad=0.1',
                              ax=ax2)
        
        nx.draw_networkx_labels(G_quantum, pos, 
                              font_size=font_size, 
                              font_weight='bold',
                              ax=ax2)
        
        # Add quantum indicators
        ax2.text(0.02, 0.02, "Quantum Effects Active", transform=ax2.transAxes,
               color=COLORS['entangled'], fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        return ax1, ax2
    
    # Create animation
    print(f"Creating comparison animation with {n_frames} frames...")
    ani = animation.FuncAnimation(fig, update_comparison, frames=n_frames, blit=False)
    
    # Save animation
    print(f"Saving comparison animation to {filename}.gif...")
    ani.save(f"{filename}.gif", writer=PillowWriter(fps=5), dpi=120)
    plt.close()
    
    print(f"Comparison animation saved successfully to {filename}.gif")
    return ani


# Main execution when run directly
if __name__ == "__main__":
    print("Enhanced Quantum Graph Visualizer")
    print("================================")
    
    # Example degree sequence
    example_deg_seq = [2, 2, 2, 2, 3, 1]  # length 6
    example_u_size, example_v_size = 3, 3
    
    print("\nCreating quantum visualizations...")
    
    # Create quantum network evolution animation
    create_quantum_network_evolution_animation(
        example_deg_seq, example_u_size, example_v_size, 
        n_frames=20, quantum_method='bloch'
    )
    
    # Create variation with different quantum method
    create_quantum_network_evolution_animation(
        example_deg_seq, example_u_size, example_v_size, 
        n_frames=20, filename="quantum_network_evolution_phase", 
        quantum_method='phase'
    )
    
    # Create stability visualization
    visualize_quantum_permutation_stability(
        example_deg_seq, example_u_size, example_v_size, 
        n_samples=30
    )
    
    # Create 3D quantum visualization
    create_3d_quantum_network_animation(
        example_deg_seq, example_u_size, example_v_size, 
        n_frames=20
    )
    
    # Create comparison animation
    compare_quantum_classical_evolution(
        example_deg_seq, example_u_size, example_v_size, 
        n_frames=15
    )
    
    print("\nAll visualizations completed successfully!")

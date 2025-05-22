#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topological_concepts.py

This module explains and demonstrates the key topological concepts used in the topological
partitioning framework, including the Čech complex, Vietoris-Rips complex, persistent homology,
and Betti numbers. It provides both theoretical background and practical examples.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import os
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import gudhi as gd
import string

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topo_network_animations')
os.makedirs(output_dir, exist_ok=True)

#########################
# Theoretical Examples
#########################

def explain_simplicial_complex():
    """
    Explain and visualize simplicial complexes
    """
    print("\n=== SIMPLICIAL COMPLEXES ===")
    print("A simplicial complex is a collection of simplices (points, edges, triangles, tetrahedra, etc.)")
    print("that fit together in a specific way.")
    print("\nExamples of simplices:")
    print("- 0-simplex: a vertex (point)")
    print("- 1-simplex: an edge (line segment)")
    print("- 2-simplex: a triangle (including its interior)")
    print("- 3-simplex: a tetrahedron (including its interior)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    # 0-simplex (vertex)
    ax = axes[0]
    ax.scatter([0.5], [0.5], s=100, c='red')
    ax.set_title("0-simplex (vertex)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 1-simplex (edge)
    ax = axes[1]
    ax.plot([0.2, 0.8], [0.5, 0.5], 'blue', linewidth=2)
    ax.scatter([0.2, 0.8], [0.5, 0.5], s=100, c='red')
    ax.set_title("1-simplex (edge)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 2-simplex (triangle)
    ax = axes[2]
    vertices = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
    triangle = plt.Polygon(vertices, color='lightblue', alpha=0.7)
    ax.add_patch(triangle)
    for i in range(3):
        ax.plot([vertices[i, 0], vertices[(i+1)%3, 0]], 
               [vertices[i, 1], vertices[(i+1)%3, 1]], 'blue', linewidth=2)
    ax.scatter(vertices[:, 0], vertices[:, 1], s=100, c='red')
    ax.set_title("2-simplex (triangle)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 3-simplex (tetrahedron)
    ax = axes[3]
    ax.remove()  # Remove the original axis
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    
    # Define the vertices of the tetrahedron
    v0 = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    v2 = np.array([0.5, np.sqrt(3)/2, 0])
    v3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])
    
    # Draw the edges
    edges = [(v0, v1), (v0, v2), (v0, v3), (v1, v2), (v1, v3), (v2, v3)]
    for edge in edges:
        ax.plot3D([edge[0][0], edge[1][0]], 
                 [edge[0][1], edge[1][1]], 
                 [edge[0][2], edge[1][2]], 'blue', linewidth=2)
    
    # Draw the faces (semi-transparent)
    faces = [(v0, v1, v2), (v0, v1, v3), (v0, v2, v3), (v1, v2, v3)]
    for face in faces:
        vtx = np.array([face[0], face[1], face[2]])
        tri = plt.art3d.Poly3DCollection([vtx])
        tri.set_color('lightblue')
        tri.set_alpha(0.3)
        ax.add_collection3d(tri)
    
    # Draw the vertices
    vertices = np.array([v0, v1, v2, v3])
    ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=100, c='red')
    
    ax.set_title("3-simplex (tetrahedron)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "simplicial_complex_basics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBasic simplices visualization saved to: {output_path}")

def explain_cech_vietoris_rips():
    """
    Explain and visualize the difference between Čech and Vietoris-Rips complexes
    """
    print("\n=== ČECH AND VIETORIS-RIPS COMPLEXES ===")
    print("These are methods for building simplicial complexes from point clouds:")
    print("\nČech Complex:")
    print("- Create a ball of radius ε around each point")
    print("- Create a simplex for each set of points whose balls have a common intersection")
    print("- Computationally expensive but captures the true topology")
    
    print("\nVietoris-Rips Complex:")
    print("- Create an edge between points that are at most 2ε apart")
    print("- Create a simplex for every clique in the resulting graph")
    print("- More computationally efficient, but may miss some topological features")
    
    # Create a point cloud
    np.random.seed(42)
    n_points = 10
    points = np.random.rand(n_points, 2)
    epsilon = 0.35
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Čech complex
    ax1.set_title(f"Čech Complex (ε = {epsilon})")
    
    # Draw points
    ax1.scatter(points[:, 0], points[:, 1], s=80, c='blue', zorder=10)
    
    # Draw balls around points
    for i, point in enumerate(points):
        circle = Circle(point, epsilon, color='lightblue', alpha=0.4)
        ax1.add_patch(circle)
        ax1.text(point[0], point[1], string.ascii_uppercase[i], 
                fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Find intersections for Čech complex (simplified)
    for i, j in combinations(range(n_points), 2):
        dist = np.linalg.norm(points[i] - points[j])
        if dist <= 2 * epsilon:  # Balls intersect
            ax1.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                     'k-', alpha=0.7, linewidth=1.5)
    
    # Plot triangles for Čech (simplified)
    for i, j, k in combinations(range(n_points), 3):
        dij = np.linalg.norm(points[i] - points[j])
        djk = np.linalg.norm(points[j] - points[k])
        dik = np.linalg.norm(points[i] - points[k])
        
        # Check if the circumradius is less than epsilon (simplified)
        if dij <= 2*epsilon and djk <= 2*epsilon and dik <= 2*epsilon:
            triangle = plt.Polygon([points[i], points[j], points[k]], 
                                  color='lightgreen', alpha=0.3)
            ax1.add_patch(triangle)
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    
    # Plot Vietoris-Rips complex
    ax2.set_title(f"Vietoris-Rips Complex (ε = {epsilon})")
    
    # Draw points
    ax2.scatter(points[:, 0], points[:, 1], s=80, c='blue', zorder=10)
    
    # Draw edges for Vietoris-Rips
    for i, j in combinations(range(n_points), 2):
        dist = np.linalg.norm(points[i] - points[j])
        if dist <= 2 * epsilon:  # Points are within 2*epsilon
            ax2.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                     'k-', alpha=0.7, linewidth=1.5)
    
    # Plot triangles for Vietoris-Rips
    for i, j, k in combinations(range(n_points), 3):
        dij = np.linalg.norm(points[i] - points[j])
        djk = np.linalg.norm(points[j] - points[k])
        dik = np.linalg.norm(points[i] - points[k])
        
        if dij <= 2*epsilon and djk <= 2*epsilon and dik <= 2*epsilon:
            triangle = plt.Polygon([points[i], points[j], points[k]], 
                                  color='salmon', alpha=0.3)
            ax2.add_patch(triangle)
            
    # Add labels to points
    for i, point in enumerate(points):
        ax2.text(point[0], point[1], string.ascii_uppercase[i], 
                fontsize=12, ha='center', va='center', fontweight='bold')
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "cech_vs_vietoris_rips.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nČech vs Vietoris-Rips visualization saved to: {output_path}")

def explain_filtration():
    """
    Explain and visualize the filtration process in persistent homology
    """
    print("\n=== FILTRATION IN PERSISTENT HOMOLOGY ===")
    print("Filtration is the process of growing simplicial complexes as the scale parameter (ε) increases.")
    print("This allows us to track when topological features appear and disappear:")
    print("- Birth: When a feature (component, cycle, void) appears")
    print("- Death: When a feature gets filled in or merged with another")
    print("- Persistence: How long a feature survives (death - birth)")
    
    # Create points in a circle with a hole
    n_circle = 20
    theta = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
    circle_points = np.column_stack([np.cos(theta), np.sin(theta)])
    circle_points += np.random.normal(0, 0.1, circle_points.shape)
    
    # Create filtration steps
    epsilons = [0.2, 0.4, 0.7, 1.0]
    
    # Create visualization
    fig, axes = plt.subplots(1, len(epsilons), figsize=(15, 4))
    
    for i, epsilon in enumerate(epsilons):
        ax = axes[i]
        
        # Draw points
        ax.scatter(circle_points[:, 0], circle_points[:, 1], s=50, c='blue', zorder=10)
        
        # Draw edges
        for p1, p2 in combinations(circle_points, 2):
            dist = np.linalg.norm(p1 - p2)
            if dist <= 2 * epsilon:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.7, linewidth=1)
        
        # Draw triangles (simplified)
        for p1, p2, p3 in combinations(circle_points, 3):
            d12 = np.linalg.norm(p1 - p2)
            d23 = np.linalg.norm(p2 - p3)
            d13 = np.linalg.norm(p1 - p3)
            
            if d12 <= 2*epsilon and d23 <= 2*epsilon and d13 <= 2*epsilon:
                triangle = plt.Polygon([p1, p2, p3], color='salmon', alpha=0.3)
                ax.add_patch(triangle)
        
        ax.set_title(f"ε = {epsilon}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "filtration_process.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFiltration process visualization saved to: {output_path}")

def explain_betti_numbers():
    """
    Explain and visualize Betti numbers
    """
    print("\n=== BETTI NUMBERS ===")
    print("Betti numbers count the number of topological features at each dimension:")
    print("- β₀: Number of connected components")
    print("- β₁: Number of 1-dimensional holes (cycles)")
    print("- β₂: Number of 2-dimensional voids (cavities)")
    print("- βₙ: Number of n-dimensional voids")
    
    # Create example shapes with different Betti numbers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # β₀ example: Three disconnected components
    ax = axes[0]
    ax.set_title("β₀ = 3 (Three Connected Components)")
    
    # Create three separate components
    component_positions = [
        np.array([[0.2, 0.2], [0.3, 0.5], [0.5, 0.3]]),
        np.array([[0.7, 0.7], [0.8, 0.8], [0.6, 0.8], [0.7, 0.6]]),
        np.array([[0.3, 0.8], [0.2, 0.7]])
    ]
    
    colors = ['blue', 'green', 'red']
    
    for i, positions in enumerate(component_positions):
        # Draw component
        ax.scatter(positions[:, 0], positions[:, 1], s=80, c=colors[i])
        
        # Draw edges within component
        G = nx.complete_graph(len(positions))
        pos = {i: pos for i, pos in enumerate(positions)}
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.7, width=1.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # β₁ example: A shape with two holes
    ax = axes[1]
    ax.set_title("β₁ = 2 (Two Cycles/Holes)")
    
    # Create a figure "8" shape
    theta1 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    circle1 = 0.2 * np.column_stack([np.cos(theta1), np.sin(theta1)]) + np.array([0.35, 0.5])
    
    theta2 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    circle2 = 0.2 * np.column_stack([np.cos(theta2), np.sin(theta2)]) + np.array([0.65, 0.5])
    
    # Draw the two loops
    ax.scatter(circle1[:, 0], circle1[:, 1], s=80, c='purple')
    ax.scatter(circle2[:, 0], circle2[:, 1], s=80, c='purple')
    
    # Draw edges for first circle
    for i in range(len(circle1)):
        ax.plot([circle1[i, 0], circle1[(i+1)%len(circle1), 0]], 
               [circle1[i, 1], circle1[(i+1)%len(circle1), 1]], 
               'k-', alpha=0.7, linewidth=1.5)
    
    # Draw edges for second circle
    for i in range(len(circle2)):
        ax.plot([circle2[i, 0], circle2[(i+1)%len(circle2), 0]], 
               [circle2[i, 1], circle2[(i+1)%len(circle2), 1]], 
               'k-', alpha=0.7, linewidth=1.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # β₂ example: A hollow sphere (3D)
    ax = axes[2]
    ax.remove()  # Remove the original axis
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("β₂ = 1 (One Void/Cavity)")
    
    # Create a discretized sphere
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = 0.3 * np.outer(np.cos(u), np.sin(v)) + 0.5
    y = 0.3 * np.outer(np.sin(u), np.sin(v)) + 0.5
    z = 0.3 * np.outer(np.ones(np.size(u)), np.cos(v)) + 0.5
    
    # Draw the surface of the sphere
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0)
    
    # Draw some points on the sphere
    theta = np.linspace(0, 2*np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    
    points = []
    for t in theta:
        for p in phi:
            points.append([
                0.3 * np.cos(t) * np.sin(p) + 0.5,
                0.3 * np.sin(t) * np.sin(p) + 0.5,
                0.3 * np.cos(p) + 0.5
            ])
    
    points = np.array(points)
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], s=30, c='blue')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "betti_numbers_examples.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBetti numbers visualization saved to: {output_path}")

def explain_persistence_diagram():
    """
    Explain and visualize persistence diagrams and barcodes
    """
    print("\n=== PERSISTENCE DIAGRAMS AND BARCODES ===")
    print("Persistence diagrams plot the birth vs death times of topological features:")
    print("- Each point (b, d) represents a feature that appears at b and disappears at d")
    print("- Points near the diagonal have low persistence (noise)")
    print("- Points far from the diagonal have high persistence (significant features)")
    print("\nBarcodes represent the same information as horizontal bars:")
    print("- Each bar spans from birth to death time of a feature")
    print("- Longer bars represent more persistent features")
    
    # Create synthetic persistence data
    np.random.seed(42)
    
    # H0 (connected components)
    h0_births = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    h0_deaths = np.array([0.8, 1.0, 0.3, 0.4, 0.2])
    
    # H1 (cycles)
    h1_births = np.array([0.2, 0.3, 0.4, 0.5])
    h1_deaths = np.array([0.9, 0.8, 0.6, 0.7])
    
    # H2 (voids)
    h2_births = np.array([0.4, 0.5])
    h2_deaths = np.array([0.7, 0.8])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot persistence diagram
    ax1.set_title("Persistence Diagram")
    
    # Plot diagonal
    lim = [0, 1.1]
    ax1.plot(lim, lim, 'k--', alpha=0.3)
    
    # Plot H0 features
    ax1.scatter(h0_births, h0_deaths, s=100, c='blue', label="H₀", alpha=0.8)
    
    # Plot H1 features
    ax1.scatter(h1_births, h1_deaths, s=100, c='red', label="H₁", alpha=0.8)
    
    # Plot H2 features
    ax1.scatter(h2_births, h2_deaths, s=100, c='green', label="H₂", alpha=0.8)
    
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot barcode
    ax2.set_title("Persistence Barcode")
    
    y_pos = 0
    bar_height = 0.6
    y_gap = 1.5
    
    # Plot H0 bars
    for i, (birth, death) in enumerate(zip(h0_births, h0_deaths)):
        y = y_pos + i * y_gap
        ax2.plot([birth, death], [y, y], linewidth=5, color='blue', solid_capstyle='butt')
        ax2.text(1.02, y, f"H₀-{i}", va='center')
    
    y_pos = len(h0_births) * y_gap + 2
    
    # Plot H1 bars
    for i, (birth, death) in enumerate(zip(h1_births, h1_deaths)):
        y = y_pos + i * y_gap
        ax2.plot([birth, death], [y, y], linewidth=5, color='red', solid_capstyle='butt')
        ax2.text(1.02, y, f"H₁-{i}", va='center')
    
    y_pos = y_pos + len(h1_births) * y_gap + 2
    
    # Plot H2 bars
    for i, (birth, death) in enumerate(zip(h2_births, h2_deaths)):
        y = y_pos + i * y_gap
        ax2.plot([birth, death], [y, y], linewidth=5, color='green', solid_capstyle='butt')
        ax2.text(1.02, y, f"H₂-{i}", va='center')
    
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(-1, y_pos + len(h2_births) * y_gap + 1)
    ax2.set_xlabel("Filtration Value (ε)")
    ax2.yaxis.set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add legends for dimensions
    ax2.plot([], [], linewidth=5, color='blue', label="H₀ (Components)")
    ax2.plot([], [], linewidth=5, color='red', label="H₁ (Cycles)")
    ax2.plot([], [], linewidth=5, color='green', label="H₂ (Voids)")
    ax2.legend(loc='upper center')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "persistence_diagram_barcode.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPersistence diagram and barcode visualization saved to: {output_path}")

#########################
# Network Applications
#########################

def explain_network_topology_applications():
    """
    Explain and visualize applications of topology to networks
    """
    print("\n=== NETWORK TOPOLOGY APPLICATIONS ===")
    print("Topological data analysis offers several insights into network structure:")
    print("1. Identifying clusters and communities (H₀)")
    print("2. Detecting loops and cycles in information flow (H₁)")
    print("3. Finding higher-dimensional structures like 3D cavities (H₂)")
    print("4. Comparing networks based on their topological signatures")
    print("5. Partitioning networks based on topological similarity")
    
    # Create different types of networks
    np.random.seed(42)
    
    # Network 1: Clustered network (high β₀)
    G1 = nx.Graph()
    
    # Create three clusters
    cluster1 = range(0, 5)
    cluster2 = range(5, 10)
    cluster3 = range(10, 15)
    
    # Add edges within clusters
    for c in [cluster1, cluster2, cluster3]:
        for i in c:
            for j in c:
                if i != j:
                    G1.add_edge(i, j, weight=0.9)
    
    # Add a few inter-cluster edges
    G1.add_edge(4, 5, weight=0.3)
    G1.add_edge(9, 10, weight=0.3)
    
    # Network 2: Cyclic network (high β₁)
    G2 = nx.Graph()
    
    # Create two cycles
    cycle1 = list(range(0, 8))
    cycle2 = list(range(8, 15))
    
    # Add edges to form cycles
    for i, c in enumerate(cycle1):
        G2.add_edge(c, cycle1[(i+1) % len(cycle1)], weight=0.9)
    
    for i, c in enumerate(cycle2):
        G2.add_edge(c, cycle2[(i+1) % len(cycle2)], weight=0.9)
    
    # Connect the cycles with one edge
    G2.add_edge(0, 8, weight=0.3)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot clustered network
    ax = axes[0]
    ax.set_title("Clustered Network (High β₀)")
    
    pos1 = nx.spring_layout(G1, seed=42)
    
    # Color nodes by cluster
    node_colors = []
    for node in G1.nodes():
        if node in cluster1:
            node_colors.append('blue')
        elif node in cluster2:
            node_colors.append('red')
        else:
            node_colors.append('green')
    
    # Draw nodes
    nx.draw_networkx_nodes(G1, pos1, node_color=node_colors, 
                          node_size=100, alpha=0.8, ax=ax)
    
    # Draw edges with weight-based width
    for u, v, data in G1.edges(data=True):
        weight = data.get('weight', 0.5)
        width = weight * 3
        if weight > 0.5:  # Intra-cluster edge
            nx.draw_networkx_edges(G1, pos1, edgelist=[(u, v)], 
                                  width=width, alpha=0.7, ax=ax)
        else:  # Inter-cluster edge
            nx.draw_networkx_edges(G1, pos1, edgelist=[(u, v)], 
                                  width=width, style='dashed', alpha=0.7, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G1, pos1, font_size=8, ax=ax)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    
    # Plot cyclic network
    ax = axes[1]
    ax.set_title("Cyclic Network (High β₁)")
    
    pos2 = nx.spring_layout(G2, seed=42)
    
    # Color nodes by cycle
    node_colors = []
    for node in G2.nodes():
        if node in cycle1:
            node_colors.append('purple')
        else:
            node_colors.append('orange')
    
    # Draw nodes
    nx.draw_networkx_nodes(G2, pos2, node_color=node_colors, 
                          node_size=100, alpha=0.8, ax=ax)
    
    # Draw edges with weight-based width
    for u, v, data in G2.edges(data=True):
        weight = data.get('weight', 0.5)
        width = weight * 3
        if weight > 0.5:  # Intra-cycle edge
            nx.draw_networkx_edges(G2, pos2, edgelist=[(u, v)], 
                                  width=width, alpha=0.7, ax=ax)
        else:  # Inter-cycle edge
            nx.draw_networkx_edges(G2, pos2, edgelist=[(u, v)], 
                                  width=width, style='dashed', alpha=0.7, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G2, pos2, font_size=8, ax=ax)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_topology_applications.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nNetwork topology applications visualization saved to: {output_path}")

def explain_topological_partitioning():
    """
    Explain and visualize the topological partitioning process
    """
    print("\n=== TOPOLOGICAL NETWORK PARTITIONING ===")
    print("Topological partitioning combines trust-based partitioning with topological features:")
    print("1. Compute topological features for each network (Betti numbers, persistence)")
    print("2. Calculate topological compatibility between networks")
    print("3. Combine trust scores with topological compatibility")
    print("4. Make node migration decisions based on combined score")
    print("5. Analyze resulting networks to ensure topological consistency")
    
    # Create a visualization of the process
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Step 1: Initial networks
    ax = axes[0]
    ax.set_title("1. Initial Networks")
    
    # Create three networks
    G = nx.Graph()
    
    # Network A: Triangle (high transitivity)
    G.add_nodes_from([0, 1, 2], network='A')
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    # Network B: Line (low transitivity)
    G.add_nodes_from([3, 4, 5], network='B')
    G.add_edges_from([(3, 4), (4, 5)])
    
    # Network C: Star (hub-spoke)
    G.add_nodes_from([6, 7, 8, 9], network='C')
    G.add_edges_from([(6, 7), (6, 8), (6, 9)])
    
    # Set positions
    pos = {
        0: (0.2, 0.2), 1: (0.4, 0.4), 2: (0.2, 0.4),
        3: (0.7, 0.2), 4: (0.8, 0.2), 5: (0.9, 0.2),
        6: (0.5, 0.8), 7: (0.3, 0.8), 8: (0.5, 0.6), 9: (0.7, 0.8)
    }
    
    # Color by network
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['network'] == 'A':
            node_colors.append('blue')
        elif G.nodes[node]['network'] == 'B':
            node_colors.append('red')
        else:
            node_colors.append('green')
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add network labels
    ax.text(0.25, 0.1, "Network A\nβ₀=1, β₁=1", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    ax.text(0.8, 0.1, "Network B\nβ₀=1, β₁=0", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    ax.text(0.5, 0.9, "Network C\nβ₀=1, β₁=0", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 2: Compute topological compatibility
    ax = axes[1]
    ax.set_title("2. Topological Compatibility Matrix")
    
    # Create a compatibility matrix
    compat_matrix = np.array([
        [1.0, 0.5, 0.6],
        [0.5, 1.0, 0.9],
        [0.6, 0.9, 1.0]
    ])
    
    # Plot the matrix
    im = ax.imshow(compat_matrix, cmap='YlGnBu')
    
    # Add labels
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_yticklabels(['A', 'B', 'C'])
    
    # Add compatibility values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{compat_matrix[i, j]:.1f}", 
                   ha='center', va='center', fontsize=12)
    
    # Add explanation
    ax.text(0.5, -0.2, "Closer topological signatures = higher compatibility", 
           ha='center', va='center', transform=ax.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Step 3: Combine trust with topology
    ax = axes[2]
    ax.set_title("3. Combined Trust & Topology")
    
    # Create a simplified visualization of the combination
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Draw axes
    ax.plot([0, 100], [0, 0], 'k-', alpha=0.5)
    ax.plot([0, 0], [0, 100], 'k-', alpha=0.5)
    
    # Label axes
    ax.text(50, -10, "Trust Score", ha='center', va='top')
    ax.text(-10, 50, "Topological Compatibility", ha='right', va='center', rotation=90)
    
    # Draw the weighting curve for different values
    x = np.linspace(0, 100, 100)
    for weight, color, label in zip(
        [0.0, 0.5, 1.0], 
        ['blue', 'purple', 'red'],
        ['Trust Only', 'Equal Weight', 'Topology Only']
    ):
        if weight == 0.0:
            y = x
        elif weight == 1.0:
            y = np.ones_like(x) * 50
        else:
            y = (1 - weight) * x + weight * 50
        
        ax.plot(x, y, color=color, label=label, linewidth=2)
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Step 4: Network evolution
    ax = axes[3]
    ax.set_title("4. Network Evolution")
    
    # Create evolved network (e.g., node 5 moves to network C)
    G_evolved = G.copy()
    G_evolved.nodes[5]['network'] = 'C'  # Node 5 moves from B to C
    
    # Color by evolved network
    node_colors_evolved = []
    for node in G_evolved.nodes():
        if G_evolved.nodes[node]['network'] == 'A':
            node_colors_evolved.append('blue')
        elif G_evolved.nodes[node]['network'] == 'B':
            node_colors_evolved.append('red')
        else:
            node_colors_evolved.append('green')
    
    # Draw the evolved network
    nx.draw_networkx_nodes(G_evolved, pos, node_color=node_colors_evolved, 
                          node_size=100, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G_evolved, pos, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G_evolved, pos, font_size=8, ax=ax)
    
    # Highlight the moved node
    ax.scatter(pos[5][0], pos[5][1], s=200, facecolors='none', 
               edgecolors='green', linewidths=2)
    
    # Add network labels with updated Betti numbers
    ax.text(0.25, 0.1, "Network A\nβ₀=1, β₁=1", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    ax.text(0.8, 0.1, "Network B\nβ₀=1, β₁=0", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    ax.text(0.5, 0.9, "Network C\nβ₀=1, β₁=0", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add explanation of the move
    ax.text(0.8, 0.3, "Node 5\nmoved to\nNetwork C", ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', color='green'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "topological_partitioning_process.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTopological partitioning process visualization saved to: {output_path}")

#########################
# Main Function
#########################

if __name__ == "__main__":
    print("Topological Concepts for Network Analysis")
    print("========================================")
    
    try:
        # Basic topology concepts
        explain_simplicial_complex()
        explain_cech_vietoris_rips()
        explain_filtration()
        explain_betti_numbers()
        explain_persistence_diagram()
        
        # Network applications
        explain_network_topology_applications()
        explain_topological_partitioning()
        
        print("\nAll explanations and visualizations complete!")
        print(f"Check the folder: {output_dir} for visual guides")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkGenesis.py

This script generates dynamic network visualizations showing the evolution and
merging of network communities in 3D space. It creates animated visualizations of:

1. Network formation with growing connectivity
2. Network merging with cross-edges 
3. Topological simplices (triangles) that form during network evolution
4. Persistence homology landscapes showing topological features

The script produces several variations with different network densities and
community structures. Output animations are saved as GIF files in the
current directory.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from itertools import combinations
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# PARAMETERS
num_steps_main = 30
hold_frames = 4
total_frames = num_steps_main + hold_frames
n0, radius0 = 20, 0.25
n1, radius1 = 20, 0.25

# GEOMETRIC GRAPHS
A = nx.random_geometric_graph(n0, radius0, seed=1)
B = nx.random_geometric_graph(n1, radius1, seed=2)

scale = 8
offset_x = scale + 2
pos2d = {}
for i, (x, y) in nx.get_node_attributes(A, 'pos').items():
    pos2d[i] = ((x - 0.5) * scale, (y - 0.5) * scale)
for j, (x, y) in nx.get_node_attributes(B, 'pos').items():
    pos2d[j + n0] = ((x - 0.5) * scale + offset_x, (y - 0.5) * scale)

# CROSS-EDGES
cross_pairs = []
for u in A.nodes():
    x1, y1 = pos2d[u]
    for v in B.nodes():
        vid = v + n0
        x2, y2 = pos2d[vid]
        dist = np.hypot(x2 - x1, y2 - y1)
        cross_pairs.append((u, vid, dist))
cross_pairs.sort(key=lambda x: x[2])

final_graph = nx.disjoint_union(A, B)
for u, vid, _ in cross_pairs:
    final_graph.add_edge(u, vid)

# PLOT SETUP
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
zs = 0

def grow(G, t, T):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    edges = list(G.edges())
    count = int(len(edges) * (t + 1) / T)
    H.add_edges_from(edges[:count])
    return H

def merge_step(t):
    G = nx.disjoint_union(A, B)
    for u, vid, _ in cross_pairs[:t]:
        G.add_edge(u, vid)
    return G

def draw_simplices(G):
    triangles = []
    for trio in combinations(G.nodes(), 3):
        if G.has_edge(trio[0], trio[1]) and G.has_edge(trio[1], trio[2]) and G.has_edge(trio[0], trio[2]):
            pts = [(*pos2d[n], zs) for n in trio]
            triangles.append(pts)
    if triangles:
        poly = Poly3DCollection(triangles, facecolors='purple', alpha=0.6, linewidths=0)
        ax.add_collection3d(poly)

def draw_graph(G):
    draw_simplices(G)
    for u, v in G.edges():
        x1, y1 = pos2d[u]; x2, y2 = pos2d[v]
        ax.plot([x1, x2], [y1, y2], [zs, zs], color='white', linewidth=1.5)
    xs, ys = zip(*(pos2d[n] for n in G.nodes()))
    ax.scatter(xs, ys, zs, c='purple', s=60)

def update(frame):
    ax.clear()
    ax.set_facecolor('black')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    if frame < num_steps_main // 3:
        H = grow(A, frame, num_steps_main // 3)
    elif frame < 2 * (num_steps_main // 3):
        k = frame - num_steps_main // 3
        H = grow(B, k, num_steps_main // 3)
        H = nx.relabel_nodes(H, {u: u + n0 for u in B.nodes()})
    elif frame < num_steps_main:
        t = frame - 2 * (num_steps_main // 3)
        H = merge_step(t)
    else:
        H = final_graph

    draw_graph(H)
    step = min(frame + 1, num_steps_main)
    ax.set_title(f"Step {step}/{num_steps_main}", color='white')
    ax.set_axis_off()

# Create and save animation
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, 'network_evolution_3d_simplices_v5.gif')
anim = FuncAnimation(fig, update, frames=total_frames, interval=200)
writer = PillowWriter(fps=5)
anim.save(output_file, writer=writer)

print(f"Animation saved to {output_file}")

# Create variations with different network configurations
print("Creating variations with different network configurations...")

# Variation 1: More dense networks
def create_variation_1():
    # Create denser networks
    A_var1 = nx.random_geometric_graph(n0, radius0*1.5, seed=3)
    B_var1 = nx.random_geometric_graph(n1, radius1*1.5, seed=4)
    
    # Recalculate positions and cross edges
    pos2d_var1 = {}
    for i, (x, y) in nx.get_node_attributes(A_var1, 'pos').items():
        pos2d_var1[i] = ((x - 0.5) * scale, (y - 0.5) * scale)
    for j, (x, y) in nx.get_node_attributes(B_var1, 'pos').items():
        pos2d_var1[j + n0] = ((x - 0.5) * scale + offset_x, (y - 0.5) * scale)
    
    # Recalculate cross edges
    cross_pairs_var1 = []
    for u in A_var1.nodes():
        x1, y1 = pos2d_var1[u]
        for v in B_var1.nodes():
            vid = v + n0
            x2, y2 = pos2d_var1[vid]
            dist = np.hypot(x2 - x1, y2 - y1)
            cross_pairs_var1.append((u, vid, dist))
    cross_pairs_var1.sort(key=lambda x: x[2])
    
    # Update final graph
    final_graph_var1 = nx.disjoint_union(A_var1, B_var1)
    for u, vid, _ in cross_pairs_var1[:len(cross_pairs_var1)//2]:  # Add fewer cross edges
        final_graph_var1.add_edge(u, vid)
    
    # Create a variation-specific update function
    def update_var1(frame):
        ax.clear()
        ax.set_facecolor('black')
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        
        if frame < num_steps_main // 3:
            H = grow_var1(A_var1, frame, num_steps_main // 3)
        elif frame < 2 * (num_steps_main // 3):
            k = frame - num_steps_main // 3
            H = grow_var1(B_var1, k, num_steps_main // 3)
            H = nx.relabel_nodes(H, {u: u + n0 for u in B_var1.nodes()})
        elif frame < num_steps_main:
            t = frame - 2 * (num_steps_main // 3)
            H = merge_step_var1(t)
        else:
            H = final_graph_var1
        
        draw_graph_var1(H)
        step = min(frame + 1, num_steps_main)
        ax.set_title(f"Step {step}/{num_steps_main}", color='white')
        ax.set_axis_off()
        
        return ax
    
    # Create variation-specific helper functions
    def grow_var1(G, t, T):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        edges = list(G.edges())
        count = int(len(edges) * (t + 1) / T)
        H.add_edges_from(edges[:count])
        return H
    
    def merge_step_var1(t):
        G = nx.disjoint_union(A_var1, B_var1)
        for u, vid, _ in cross_pairs_var1[:t]:
            G.add_edge(u, vid)
        return G
    
    def draw_simplices_var1(G):
        triangles = []
        for trio in combinations(G.nodes(), 3):
            if G.has_edge(trio[0], trio[1]) and G.has_edge(trio[1], trio[2]) and G.has_edge(trio[0], trio[2]):
                pts = [(*pos2d_var1[n], zs) for n in trio]
                triangles.append(pts)
        if triangles:
            poly = Poly3DCollection(triangles, facecolors='purple', alpha=0.6, linewidths=0)
            ax.add_collection3d(poly)
    
    def draw_graph_var1(G):
        draw_simplices_var1(G)
        for u, v in G.edges():
            if u in pos2d_var1 and v in pos2d_var1:  # Make sure nodes are in pos2d_var1
                x1, y1 = pos2d_var1[u]; x2, y2 = pos2d_var1[v]
                ax.plot([x1, x2], [y1, y2], [zs, zs], color='white', linewidth=1.5)
        
        # Only get positions for nodes that exist in pos2d_var1
        node_positions = [(n, pos2d_var1[n]) for n in G.nodes() if n in pos2d_var1]
        if node_positions:
            xs, ys = zip(*[pos for _, pos in node_positions])
            ax.scatter(xs, ys, zs, c='purple', s=60)
    
    # Save new animation
    variation_file = os.path.join(output_dir, 'network_evolution_3d_simplices_variation1.gif')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    
    anim = FuncAnimation(fig, update_var1, frames=total_frames, interval=200)
    writer = PillowWriter(fps=5)
    anim.save(variation_file, writer=writer)
    plt.close(fig)
    
    print(f"Variation 1 saved to {variation_file}")

# Variation 2: Community-focused networks
def create_variation_2():
    # Create networks with more community structure
    A_var2 = nx.random_geometric_graph(n0 + 5, radius0*0.8, seed=5)
    B_var2 = nx.random_geometric_graph(n1 + 5, radius1*0.8, seed=6)
    
    # Recalculate positions and cross edges
    pos2d_var2 = {}
    for i, (x, y) in nx.get_node_attributes(A_var2, 'pos').items():
        pos2d_var2[i] = ((x - 0.5) * scale, (y - 0.5) * scale)
    for j, (x, y) in nx.get_node_attributes(B_var2, 'pos').items():
        pos2d_var2[j + n0 + 5] = ((x - 0.5) * scale + offset_x, (y - 0.5) * scale)
    
    # Recalculate cross edges - fewer cross edges to emphasize community structure
    cross_pairs_var2 = []
    for u in A_var2.nodes():
        x1, y1 = pos2d_var2[u]
        for v in B_var2.nodes():
            vid = v + n0 + 5
            x2, y2 = pos2d_var2[vid]
            dist = np.hypot(x2 - x1, y2 - y1)
            cross_pairs_var2.append((u, vid, dist))
    cross_pairs_var2.sort(key=lambda x: x[2])
    
    # Update final graph with fewer cross edges
    final_graph_var2 = nx.disjoint_union(A_var2, B_var2)
    for u, vid, _ in cross_pairs_var2[:len(cross_pairs_var2)//4]:  # Add even fewer cross edges
        final_graph_var2.add_edge(u, vid)
    
    # Create a variation-specific update function
    def update_var2(frame):
        ax.clear()
        ax.set_facecolor('black')
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        
        if frame < num_steps_main // 3:
            H = grow_var2(A_var2, frame, num_steps_main // 3)
        elif frame < 2 * (num_steps_main // 3):
            k = frame - num_steps_main // 3
            H = grow_var2(B_var2, k, num_steps_main // 3)
            H = nx.relabel_nodes(H, {u: u + n0 + 5 for u in B_var2.nodes()})
        elif frame < num_steps_main:
            t = frame - 2 * (num_steps_main // 3)
            H = merge_step_var2(t)
        else:
            H = final_graph_var2
        
        draw_graph_var2(H)
        step = min(frame + 1, num_steps_main)
        ax.set_title(f"Step {step}/{num_steps_main}", color='white')
        ax.set_axis_off()
        
        return ax
    
    # Create variation-specific helper functions
    def grow_var2(G, t, T):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        edges = list(G.edges())
        count = int(len(edges) * (t + 1) / T)
        H.add_edges_from(edges[:count])
        return H
    
    def merge_step_var2(t):
        G = nx.disjoint_union(A_var2, B_var2)
        for u, vid, _ in cross_pairs_var2[:t]:
            G.add_edge(u, vid)
        return G
    
    def draw_simplices_var2(G):
        triangles = []
        for trio in combinations(G.nodes(), 3):
            if G.has_edge(trio[0], trio[1]) and G.has_edge(trio[1], trio[2]) and G.has_edge(trio[0], trio[2]):
                # Make sure all nodes in the trio are in pos2d_var2
                if all(n in pos2d_var2 for n in trio):
                    pts = [(*pos2d_var2[n], zs) for n in trio]
                    triangles.append(pts)
        if triangles:
            poly = Poly3DCollection(triangles, facecolors='purple', alpha=0.6, linewidths=0)
            ax.add_collection3d(poly)
    
    def draw_graph_var2(G):
        draw_simplices_var2(G)
        for u, v in G.edges():
            if u in pos2d_var2 and v in pos2d_var2:  # Make sure nodes are in pos2d_var2
                x1, y1 = pos2d_var2[u]; x2, y2 = pos2d_var2[v]
                ax.plot([x1, x2], [y1, y2], [zs, zs], color='white', linewidth=1.5)
        
        # Only get positions for nodes that exist in pos2d_var2
        node_positions = [(n, pos2d_var2[n]) for n in G.nodes() if n in pos2d_var2]
        if node_positions:
            xs, ys = zip(*[pos for _, pos in node_positions])
            ax.scatter(xs, ys, zs, c='purple', s=60)
    
    # Save new animation
    variation_file = os.path.join(output_dir, 'network_evolution_3d_simplices_variation2.gif')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    
    anim = FuncAnimation(fig, update_var2, frames=total_frames, interval=200)
    writer = PillowWriter(fps=5)
    anim.save(variation_file, writer=writer)
    plt.close(fig)
    
    print(f"Variation 2 saved to {variation_file}")

# Run the variations
create_variation_1()
create_variation_2()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from itertools import combinations
from matplotlib.collections import PolyCollection

# Generate random point cloud
np.random.seed(42)
n_points = 30
points = np.random.rand(n_points, 2) * 2 - 1

# Compute pairwise distances
edges = []
for i, j in combinations(range(n_points), 2):
    dist = np.linalg.norm(points[i] - points[j])
    edges.append((i, j, dist))
edges.sort(key=lambda x: x[2])

# Union-Find for H0 barcode
parent = list(range(n_points))
rank = [0] * n_points
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

# Compute H0 bars
bars = []
for i, j, dist in edges:
    if union(i, j):
        bars.append((0.0, dist))
bars.append((0.0, 1.0))  # Cap infinite bar
bars.sort(key=lambda x: x[1], reverse=True)

# Radii and persistence landscape λ1
radii = np.linspace(0, 1.0, 50)
f_vals = np.zeros((len(bars), len(radii)))
for idx, (_, death) in enumerate(bars):
    for j, t in enumerate(radii):
        f_vals[idx, j] = max(0, min(t, death - t))
lam1 = f_vals.max(axis=0)

# Setup figure with adjusted margins
fig, (ax_pts, ax_bar, ax_land) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.1)

output_dir = os.path.dirname(os.path.abspath(__file__))
gif_path = os.path.join(output_dir, "network_complex_with_landscape_v2.gif")
writer = PillowWriter(fps=10)

def update(frame):
    r = radii[frame]
    ax_pts.clear(); ax_bar.clear(); ax_land.clear()
    
    # Network Complex subplot
    ax_pts.scatter(points[:, 0], points[:, 1], color='black')
    # Draw 2-simplices translucent grey
    triangles = []
    for u, v, w in combinations(range(n_points), 3):
        if (np.linalg.norm(points[u]-points[v]) <= r and
            np.linalg.norm(points[v]-points[w]) <= r and
            np.linalg.norm(points[u]-points[w]) <= r):
            triangles.append([points[u], points[v], points[w]])
    if triangles:
        coll = PolyCollection(triangles, facecolors='gray', alpha=0.3, edgecolors='none')
        ax_pts.add_collection(coll)
    # Draw edges
    for i, j, dist in edges:
        if dist <= r:
            ax_pts.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                        color='gray', alpha=0.5)
    ax_pts.set_title(f"Network Complex (r = {r:.2f})")
    ax_pts.set_aspect('equal'); ax_pts.axis('off')
    
    # H0 Persistence Barcode subplot
    for idx, (birth, death) in enumerate(bars):
        if r >= birth:
            end = min(r, death)
            ax_bar.hlines(idx, birth, end, linewidth=4, color='purple')
    ax_bar.set_xlim(0, 1.0); ax_bar.set_ylim(-1, len(bars))
    ax_bar.set_title("H0 Persistence Barcode"); ax_bar.set_xlabel("Radius"); ax_bar.set_yticks([])

    # Persistence Landscape subplot
    ax_land.plot(radii[:frame+1], lam1[:frame+1], color='darkblue', linewidth=2)
    ax_land.set_xlim(0, 1.0); ax_land.set_ylim(0, lam1.max() * 1.05)
    ax_land.set_title("Persistence Landscape λ1"); ax_land.set_xlabel("Radius"); ax_land.set_ylabel("λ1")

# Create and save animation
with writer.saving(fig, gif_path, dpi=100):
    for frame in range(len(radii)):
        update(frame)
        writer.grab_frame()

# Provide frame preview and link
frame_path = os.path.join(output_dir, "network_complex_with_landscape_v2_frame.png")
fig.savefig(frame_path)
print(f"Animation saved to {gif_path}")
print(f"Frame preview saved to {frame_path}")

# Create variations of the landscape visualization with different network topologies
print("Creating variations of the landscape visualization...")

# Variation 1: More complex network with higher density
def create_landscape_variation_1():
    # Generate denser point cloud with different seed
    np.random.seed(43)
    n_points = 35
    points_var1 = np.random.rand(n_points, 2) * 2 - 1
    
    # Compute pairwise distances
    edges_var1 = []
    for i, j in combinations(range(n_points), 2):
        dist = np.linalg.norm(points_var1[i] - points_var1[j])
        edges_var1.append((i, j, dist))
    edges_var1.sort(key=lambda x: x[2])
    
    # Union-Find for H0 barcode
    parent_var1 = list(range(n_points))
    rank_var1 = [0] * n_points
    
    # Reset union-find functions for new data
    def find_var1(x):
        while parent_var1[x] != x:
            parent_var1[x] = parent_var1[parent_var1[x]]
            x = parent_var1[x]
        return x
    
    def union_var1(x, y):
        rx, ry = find_var1(x), find_var1(y)
        if rx == ry:
            return False
        if rank_var1[rx] < rank_var1[ry]:
            parent_var1[rx] = ry
        elif rank_var1[rx] > rank_var1[ry]:
            parent_var1[ry] = rx
        else:
            parent_var1[ry] = rx
            rank_var1[rx] += 1
        return True
    
    # Compute H0 bars
    bars_var1 = []
    for i, j, dist in edges_var1:
        if union_var1(i, j):
            bars_var1.append((0.0, dist))
    bars_var1.append((0.0, 1.0))  # Cap infinite bar
    bars_var1.sort(key=lambda x: x[1], reverse=True)
    
    # Radii and persistence landscape λ1
    radii_var1 = np.linspace(0, 1.0, 50)
    f_vals_var1 = np.zeros((len(bars_var1), len(radii_var1)))
    for idx, (_, death) in enumerate(bars_var1):
        for j, t in enumerate(radii_var1):
            f_vals_var1[idx, j] = max(0, min(t, death - t))
    lam1_var1 = f_vals_var1.max(axis=0)
    
    # Setup figure with adjusted margins
    fig_var1, (ax_pts, ax_bar, ax_land) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Create variation file path
    var1_path = os.path.join(output_dir, "network_complex_with_landscape_variation1.gif")
    writer = PillowWriter(fps=10)
    
    def update_var1(frame):
        r = radii_var1[frame]
        ax_pts.clear(); ax_bar.clear(); ax_land.clear()
        
        # Network Complex subplot
        ax_pts.scatter(points_var1[:, 0], points_var1[:, 1], color='black')
        # Draw 2-simplices translucent grey
        triangles = []
        for u, v, w in combinations(range(n_points), 3):
            if (np.linalg.norm(points_var1[u]-points_var1[v]) <= r and
                np.linalg.norm(points_var1[v]-points_var1[w]) <= r and
                np.linalg.norm(points_var1[u]-points_var1[w]) <= r):
                triangles.append([points_var1[u], points_var1[v], points_var1[w]])
        if triangles:
            coll = PolyCollection(triangles, facecolors='darkblue', alpha=0.3, edgecolors='none')
            ax_pts.add_collection(coll)
        # Draw edges
        for i, j, dist in edges_var1:
            if dist <= r:
                ax_pts.plot([points_var1[i, 0], points_var1[j, 0]], 
                            [points_var1[i, 1], points_var1[j, 1]], 
                            color='navy', alpha=0.5)
        ax_pts.set_title(f"Dense Network Complex (r = {r:.2f})")
        ax_pts.set_aspect('equal'); ax_pts.axis('off')
        
        # H0 Persistence Barcode subplot
        for idx, (birth, death) in enumerate(bars_var1):
            if r >= birth:
                end = min(r, death)
                ax_bar.hlines(idx, birth, end, linewidth=4, color='darkblue')
        ax_bar.set_xlim(0, 1.0); ax_bar.set_ylim(-1, len(bars_var1))
        ax_bar.set_title("Dense Network H0 Persistence Barcode")
        ax_bar.set_xlabel("Radius"); ax_bar.set_yticks([])
        
        # Persistence Landscape subplot
        ax_land.plot(radii_var1[:frame+1], lam1_var1[:frame+1], color='darkblue', linewidth=2)
        ax_land.set_xlim(0, 1.0); ax_land.set_ylim(0, lam1_var1.max() * 1.05)
        ax_land.set_title("Dense Network Persistence Landscape λ1")
        ax_land.set_xlabel("Radius"); ax_land.set_ylabel("λ1")
    
    # Create and save animation
    with writer.saving(fig_var1, var1_path, dpi=100):
        for frame in range(len(radii_var1)):
            update_var1(frame)
            writer.grab_frame()
    
    # Save frame preview
    frame_path_var1 = os.path.join(output_dir, "network_complex_with_landscape_variation1_frame.png")
    fig_var1.savefig(frame_path_var1)
    plt.close(fig_var1)
    
    print(f"Variation 1 saved to {var1_path}")
    print(f"Variation 1 frame preview saved to {frame_path_var1}")

# Variation 2: Network with community structure
def create_landscape_variation_2():
    # Generate two separate clusters of points to simulate community structure
    np.random.seed(44)
    n_points1 = 15
    n_points2 = 15
    n_points = n_points1 + n_points2
    
    # First community - top left
    points1 = np.random.rand(n_points1, 2) * 0.8 - 1.0
    # Second community - bottom right
    points2 = np.random.rand(n_points2, 2) * 0.8 + 0.2
    
    # Combine points
    points_var2 = np.vstack([points1, points2])
    
    # Compute pairwise distances
    edges_var2 = []
    for i, j in combinations(range(n_points), 2):
        dist = np.linalg.norm(points_var2[i] - points_var2[j])
        edges_var2.append((i, j, dist))
    edges_var2.sort(key=lambda x: x[2])
    
    # Union-Find for H0 barcode
    parent_var2 = list(range(n_points))
    rank_var2 = [0] * n_points
    
    # Reset union-find functions for new data
    def find_var2(x):
        while parent_var2[x] != x:
            parent_var2[x] = parent_var2[parent_var2[x]]
            x = parent_var2[x]
        return x
    
    def union_var2(x, y):
        rx, ry = find_var2(x), find_var2(y)
        if rx == ry:
            return False
        if rank_var2[rx] < rank_var2[ry]:
            parent_var2[rx] = ry
        elif rank_var2[rx] > rank_var2[ry]:
            parent_var2[ry] = rx
        else:
            parent_var2[ry] = rx
            rank_var2[rx] += 1
        return True
    
    # Compute H0 bars
    bars_var2 = []
    for i, j, dist in edges_var2:
        if union_var2(i, j):
            bars_var2.append((0.0, dist))
    bars_var2.append((0.0, 1.0))  # Cap infinite bar
    bars_var2.sort(key=lambda x: x[1], reverse=True)
    
    # Radii and persistence landscape λ1
    radii_var2 = np.linspace(0, 1.5, 50)  # Larger radius to see community connection
    f_vals_var2 = np.zeros((len(bars_var2), len(radii_var2)))
    for idx, (_, death) in enumerate(bars_var2):
        for j, t in enumerate(radii_var2):
            f_vals_var2[idx, j] = max(0, min(t, death - t))
    lam1_var2 = f_vals_var2.max(axis=0)
    
    # Setup figure with adjusted margins
    fig_var2, (ax_pts, ax_bar, ax_land) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Create variation file path
    var2_path = os.path.join(output_dir, "network_complex_with_landscape_variation2.gif")
    writer = PillowWriter(fps=10)
    
    def update_var2(frame):
        r = radii_var2[frame]
        ax_pts.clear(); ax_bar.clear(); ax_land.clear()
        
        # Network Complex subplot
        # Color points by community
        ax_pts.scatter(points_var2[:n_points1, 0], points_var2[:n_points1, 1], color='darkred', label='Community 1')
        ax_pts.scatter(points_var2[n_points1:, 0], points_var2[n_points1:, 1], color='darkblue', label='Community 2')
        ax_pts.legend(loc='upper right', fontsize=8)
        
        # Draw 2-simplices translucent grey
        triangles = []
        for u, v, w in combinations(range(n_points), 3):
            if (np.linalg.norm(points_var2[u]-points_var2[v]) <= r and
                np.linalg.norm(points_var2[v]-points_var2[w]) <= r and
                np.linalg.norm(points_var2[u]-points_var2[w]) <= r):
                triangles.append([points_var2[u], points_var2[v], points_var2[w]])
        if triangles:
            coll = PolyCollection(triangles, facecolors='gray', alpha=0.3, edgecolors='none')
            ax_pts.add_collection(coll)
        
        # Draw edges
        for i, j, dist in edges_var2:
            if dist <= r:
                # Color edges - red for within community 1, blue for community 2, purple for cross-community
                if i < n_points1 and j < n_points1:
                    color = 'red'
                elif i >= n_points1 and j >= n_points1:
                    color = 'blue'
                else:
                    color = 'purple'
                
                ax_pts.plot([points_var2[i, 0], points_var2[j, 0]], 
                            [points_var2[i, 1], points_var2[j, 1]], 
                            color=color, alpha=0.5)
        
        ax_pts.set_title(f"Community Network Complex (r = {r:.2f})")
        ax_pts.set_aspect('equal'); ax_pts.axis('off')
        
        # H0 Persistence Barcode subplot
        for idx, (birth, death) in enumerate(bars_var2):
            if r >= birth:
                end = min(r, death)
                ax_bar.hlines(idx, birth, end, linewidth=4, color='green')
        ax_bar.set_xlim(0, 1.5); ax_bar.set_ylim(-1, len(bars_var2))
        ax_bar.set_title("Community Network H0 Persistence Barcode")
        ax_bar.set_xlabel("Radius"); ax_bar.set_yticks([])
        
        # Persistence Landscape subplot
        ax_land.plot(radii_var2[:frame+1], lam1_var2[:frame+1], color='green', linewidth=2)
        ax_land.set_xlim(0, 1.5); ax_land.set_ylim(0, lam1_var2.max() * 1.05)
        ax_land.set_title("Community Network Persistence Landscape λ1")
        ax_land.set_xlabel("Radius"); ax_land.set_ylabel("λ1")
    
    # Create and save animation
    with writer.saving(fig_var2, var2_path, dpi=100):
        for frame in range(len(radii_var2)):
            update_var2(frame)
            writer.grab_frame()
    
    # Save frame preview
    frame_path_var2 = os.path.join(output_dir, "network_complex_with_landscape_variation2_frame.png")
    fig_var2.savefig(frame_path_var2)
    plt.close(fig_var2)
    
    print(f"Variation 2 saved to {var2_path}")
    print(f"Variation 2 frame preview saved to {frame_path_var2}")

# Execute variations
create_landscape_variation_1()
create_landscape_variation_2()

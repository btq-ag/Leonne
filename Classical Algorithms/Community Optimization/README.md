# Community Optimization

> A Python toolkit for simulating, visualizing, and analyzing different types of network communities with statistical properties. This module enables the exploration of network topologies in the context of Topological Consensus Networks.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Network Types](#network-types)  
5. [Network Statistics](#network-statistics)  
6. [Blockchain Visualization](#blockchain-visualization)  
7. [API Reference](#api-reference)  
8. [Examples](#examples)  
9. [Future Extensions](#future-extensions)  
10. [References](#references)

---

## Introduction

The Community Optimization module provides tools for generating, visualizing, and analyzing different types of network communities, which are fundamental to understanding consensus dynamics in distributed systems. This module implements multiple network models inspired by neuroscience and complex systems theory, each with distinct topological properties that affect consensus formation and robustness.

The toolkit includes visualization of statistical properties, community evolution, and integration with blockchain structures, making it ideal for studying how network topology influences consensus mechanisms in distributed systems.

---

## Features

- **Multiple Network Types**: Generate random, small-world, hub, community-structured, and spatial networks
- **Statistical Analysis**: Measure and visualize clustering coefficients, path lengths, degree heterogeneity, and other network metrics
- **Animated Visualizations**: Observe network evolution and properties change over time
- **Blockchain Integration**: Visualize networks as layers in blockchain structures
- **Customizable Parameters**: Control network density, community structure, and other properties

---

## Installation

```bash
git clone https://github.com/btq-ag/CommunityOptimization.git
cd CommunityOptimization
pip install numpy networkx matplotlib scipy tqdm
```

---

## Network Types

The module implements five canonical network topologies:

1. **Random (Erdős–Rényi)**: Nodes connect randomly with probability p, resulting in a homogeneous structure with low clustering.

2. **Small-World (Watts–Strogatz)**: Combines regular structure with random shortcuts, creating high clustering with short average path lengths.

3. **Hub (Barabási–Albert)**: Scale-free networks with preferential attachment, where nodes connect to existing nodes with probability proportional to their degree.

4. **Community (Stochastic Block Model)**: Contains distinct groups with dense internal connections and sparse connections between groups.

5. **Spatial (Geometric)**: Nodes connect based on physical proximity in a geometric space, mimicking constraints of physical systems.

Each type has unique implications for consensus formation, trust propagation, and resistance to adversarial manipulation.

---

## Network Statistics

The `networkStatistics.py` module provides tools to analyze and visualize key network metrics:

- **Clustering Coefficient**: Measures the degree to which nodes tend to cluster together
- **Path Length**: Average shortest path between nodes
- **Degree Heterogeneity**: Variance-to-mean ratio of node degree distribution
- **Global Efficiency**: Inverse of the average shortest path length
- **Density**: Ratio of actual connections to possible connections

The module generates animated visualizations showing how these metrics evolve with increasing network connectivity across different network types.

```python
# Example usage
from networkStatistics import generate_networks, calculate_network_statistics, create_statistics_animation

networks = generate_networks(n_nodes=50, n_frames=20)
statistics = calculate_network_statistics(networks)
create_statistics_animation(statistics, n_frames=20)
```

The module also generates multiple variations of network visualizations to support comparative analysis of different network configurations.

### Network Evolution Visualizations

The `networkGenesis.py` script generates animations showing the evolution of network topologies with a focus on:

1. **3D Simplicial Complex Formation**: Visualizes how nodes form connected components and higher-order structures (triangles) over time
2. **Community Integration**: Shows how initially separate network communities merge together through cross-connections
3. **Persistence Diagrams**: Includes topological data analysis visualizations showing the lifespan of network features as connections form

```python
# Generate topological visualizations
python networkGenesis.py
```

This will create several animations in the Community Optimization folder:
- `network_evolution_3d_simplices_v5.gif` - Main 3D animation of network evolution
- `network_evolution_3d_simplices_variation1.gif` - Variation with denser networks
- `network_evolution_3d_simplices_variation2.gif` - Variation with stronger community structure
- `network_complex_with_landscape_v2.gif` - Network complex with persistence landscape
- `network_complex_with_landscape_variation1.gif` - Dense network complex with persistence landscape
- `network_complex_with_landscape_variation2.gif` - Community network complex with persistence landscape

### Topological Network Visualizations

The `topologicalNetworkVisualizer.py` script generates animations specifically focused on the topological aspects of consensus networks, which are more directly related to Topological Consensus Networks:

1. **Topological Consensus Formation**: Visualizes how consensus spreads through a network, forming higher-order simplicial structures as agreement increases
2. **Trust-Based Simplicial Complex**: Shows how trust thresholds affect the formation of simplicial complexes in a network
3. **Community Detection via Filtration**: Demonstrates how persistent homology filtration can identify community structures in networks

```python
# Generate topological network visualizations
python topologicalNetworkVisualizer.py
```

This will create the following animations directly in the Community Optimization folder:
- `topological_consensus_evolution.gif` - Visualizes consensus formation with higher-order simplices
- `trust_simplicial_complex.gif` - Shows trust-based simplicial complex formation
- `community_filtration.gif` - Demonstrates community detection through filtration

These visualizations help understand the topological foundations of consensus networks, illustrating how higher-order structures emerge during consensus formation and how trust relationships define the network's simplex structure.

---

## Blockchain Visualization

The `networkVisualizer.py` module creates visualizations that are directly related to topological consensus networks:

```python
from networkVisualizer import create_topological_consensus_animation, create_topological_simplices_animation, create_consensus_landscape_animation

# Create a consensus evolution animation for different network types
create_topological_consensus_animation(n_nodes=30, n_frames=30, network_type='small_world')

# Create a 3D visualization of network simplices evolution
create_topological_simplices_animation(n_nodes=40, n_frames=40)

# Create a visualization of network complex with consensus landscape
create_consensus_landscape_animation(n_points=30, n_frames=30)
```

These visualizations help understand:
- How consensus propagates through networks with different topologies
- The formation of topological structures (simplices) as networks evolve
- Relationships between persistence homology and network consensus
- How community structure affects consensus dynamics

---

## API Reference

### networkCommunities.py

- `create_random_network_animation(n, p_final, n_frames)`: Visualize Erdős–Rényi network formation
- `create_community_network_animation(n, n_communities, n_frames)`: Visualize stochastic block model evolution
- `create_small_world_network_animation(n, k, p_range, n_frames)`: Visualize Watts-Strogatz network rewiring
- `create_hub_network_animation(n, m_range, n_frames)`: Visualize Barabási-Albert network growth
- `create_spatial_network_animation(n, radius_range, n_frames)`: Visualize geometric network density changes

### networkStatistics.py

- `generate_networks(n_nodes, n_frames)`: Create multiple network types with increasing connectivity
- `calculate_network_statistics(networks)`: Compute key metrics for all network types and frames
- `create_statistics_animation(statistics, n_frames)`: Generate animated visualization of network metrics

### blockchainVisualizer.py

- `create_blockchain_network_visualization(n_blocks, n_nodes_per_block, n_frames)`: Visualize networks arranged as blockchain layers

### networkGenesis.py

- Functions to generate base networks with different properties for simulation and analysis

---

## Examples

### Analyzing Network Types for Consensus Robustness

```python
from networkStatistics import generate_networks, calculate_network_statistics
import numpy as np

networks = generate_networks(n_nodes=100, n_frames=1)
stats = calculate_network_statistics(networks)

# Compare clustering vs. path length tradeoffs
for net_type in stats:
    clustering = stats[net_type]['clustering'][-1]
    path_length = stats[net_type]['path_length'][-1]
    print(f"{net_type}: Clustering={clustering:.3f}, Path Length={path_length:.3f}")
```

### Creating a Multi-Layer Blockchain Visualization

```python
from blockchainVisualizer import create_blockchain_network_visualization

# Create a visualization with different network topologies as layers
create_blockchain_network_visualization(
    n_blocks=5,           # 5 blockchain layers
    n_nodes_per_block=15, # 15 nodes per layer
    n_frames=40           # 40 animation frames
)
```

---

## Future Extensions

- Quantum-enhanced community detection algorithms
- Trust propagation simulations across different network topologies
- Adaptive network models that evolve based on consensus history
- Integration with trust partitioning for dynamic community evolution
- Adversarial network analysis for security testing

---

## References

- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
- Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
- Newman, M. E. J. (2006). Modularity and community structure in networks. *Proceedings of the National Academy of Sciences*, 103(23), 8577-8582.
- Fortunato, S. (2010). Community detection in graphs. *Physics Reports*, 486(3-5), 75-174.
- Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, 5(2), 109-137.

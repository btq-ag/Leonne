# Blockchain Simulation

> A Python toolkit for visualizing and simulating blockchain network structures with dynamic topologies and consensus mechanisms for Topological Consensus Networks.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Blockchain Network Visualization](#blockchain-network-visualization)  
5. [Network Models](#network-models)  
6. [API Reference](#api-reference)  
7. [Examples](#examples)  
8. [Advanced Usage](#advanced-usage)  
9. [Output Files](#output-files)  
10. [Integration with Other Algorithms](#integration-with-other-algorithms)  
11. [Future Directions](#future-directions)  
12. [References](#references)

---

## Introduction

The Blockchain Simulation module provides tools for visualizing and analyzing blockchain network structures, with a focus on how different network topologies affect consensus and transaction validation. By representing blockchain layers as cylindrical structures with various network types, this module offers insights into the interactions between consensus sets across blockchain layers.

This toolkit supports the broader Topological Consensus Networks (TCN) framework by illustrating how trust propagates through blockchain structures and how different network topologies contribute to the security and efficiency of distributed consensus.

The module contains two main components:
1. **blockchainVisualizer.py**: Creates basic 3D visualizations of blockchain networks
2. **blockchainNetworkAnalyzer.py**: Extends the visualization with advanced network analysis features, integrating concepts from other algorithms in the project

---

## Features

- **3D Blockchain Visualization**: Represent blockchain layers as cylindrical structures with nodes and connections
- **Multiple Network Topologies**: Implement different network types (random, small-world, scale-free, community, spatial) for each blockchain layer
- **Animated Growth Visualization**: Observe the progressive building of blockchain layers and cross-layer connections
- **Inter-block Connections**: Visualize how nodes in one blockchain layer connect to nodes in adjacent layers
- **Customizable Parameters**: Control block size, node counts, and connection patterns
- **Trust Analysis**: Calculate and visualize trust relationships between nodes and blocks
- **Community Detection**: Identify and visualize community structures within blockchain networks
- **Enhanced Visualization**: Color-coded blocks based on network type with highlighted important features
- **Multiple Configurations**: Create mixed or homogeneous blockchain networks with various parameters

---

## Installation

```bash
git clone https://github.com/btq-ag/BlockchainSimulation.git
cd BlockchainSimulation
pip install numpy networkx matplotlib tqdm
```

---

## Blockchain Network Visualization

The `blockchainVisualizer.py` module creates 3D visualizations of blockchain networks as stacked cylindrical layers:

```python
from blockchainVisualizer import create_blockchain_network_visualization

# Create a visualization with 5 blocks, each containing 12 nodes
create_blockchain_network_visualization(n_blocks=5, n_nodes_per_block=12, n_frames=30)
```

The visualization demonstrates:
- Different network topologies for each blockchain layer (block)
- Intra-block connections between nodes within the same block
- Inter-block connections between nodes in adjacent blocks
- Progressive building of the blockchain structure

---

## Network Models

Each blockchain layer can implement one of five network models:

1. **Random Networks (Erdős–Rényi)**: Simple random connections with uniform probability, representing a baseline consensus model.

2. **Small-World Networks (Watts–Strogatz)**: High clustering with some random long-range connections, representing efficient local consensus with occasional shortcuts.

3. **Scale-Free Networks (Barabási–Albert)**: Preferential attachment creating hub structures, representing consensus networks with key validator nodes.

4. **Community Networks (Stochastic Block Model)**: Densely connected internal groups with sparse external connections, representing sub-consensus formations.

5. **Spatial Networks (Geometric)**: Distance-based connections, representing geographically constrained validation nodes.

These models allow for exploring how different consensus structures impact blockchain security, throughput, and resistance to attacks.

---

## API Reference

### blockchainVisualizer.py

- `create_blockchain_network_visualization(n_blocks, n_nodes_per_block, n_frames)`: Create an animated 3D visualization of networks arranged as blockchain layers.
  - Parameters:
    - `n_blocks`: Number of blockchain layers to generate
    - `n_nodes_per_block`: Number of nodes per blockchain layer
    - `n_frames`: Number of animation frames
  - Returns:
    - An animation object showing the progressive building of the blockchain structure

### blockchainNetworkAnalyzer.py

- `create_blockchain_network_visualization(n_blocks, n_nodes_per_block, n_frames, network_type, filename)`: Create an enhanced blockchain visualization with network analysis features.
  - Parameters:
    - `n_blocks`: Number of blockchain layers to generate
    - `n_nodes_per_block`: Number of nodes per blockchain layer
    - `n_frames`: Number of animation frames
    - `network_type`: Type of networks to create ('mixed', 'small_world', 'scale_free', 'community', 'random', 'spatial')
    - `filename`: Output filename for the animation
  - Returns:
    - An animation object showing the blockchain with network analysis

- Network generation functions:
  - `create_random_network`: Creates random networks with uniform connection probability
  - `create_small_world_network`: Creates networks with high clustering and short average path lengths
  - `create_scale_free_network`: Creates networks with power-law degree distribution
  - `create_community_network`: Creates networks with distinct community structures
  - `create_spatial_network`: Creates networks with connections based on physical proximity

- Trust analysis functions:
  - `calculate_trust_matrix`: Computes trust values between nodes based on network structure
  - `calculate_inter_block_trust`: Computes trust values between different blockchain blocks

---

## Examples

### Creating a Basic Blockchain Visualization

```python
from blockchainVisualizer import create_blockchain_network_visualization

# Create a blockchain with 5 different network types as layers
create_blockchain_network_visualization(
    n_blocks=5,           # 5 blockchain layers
    n_nodes_per_block=15, # 15 nodes per layer
    n_frames=40           # 40 animation frames
)
```

### Creating Enhanced Blockchain Visualizations with Network Analysis

```python
from blockchainNetworkAnalyzer import create_blockchain_network_visualization

# Create a mixed network type blockchain
create_blockchain_network_visualization(
    n_blocks=5,
    n_nodes_per_block=20,
    n_frames=30,
    network_type='mixed',
    filename='blockchain_mixed_networks.gif'
)

# Create a blockchain with all small-world networks
create_blockchain_network_visualization(
    n_blocks=4,
    n_nodes_per_block=15,
    n_frames=25,
    network_type='small_world',
    filename='blockchain_small_world.gif'
)
```

### Analyzing Inter-Block Connections

```python
# Future extension - code to analyze connectivity between blocks
# and identify critical cross-block validation paths
```

---

## Advanced Usage

### Custom Network Topologies

The blockchain visualization can be extended to support custom network topologies:

```python
# Future extension - code to define and integrate custom network topologies
# into the blockchain visualization framework
```

### Network Analysis Integration

The blockchain visualization includes network analysis features that can be further customized:

```python
# Example of custom trust function that could be added
def custom_trust_calculator(G, nodes, weight_factor=1.5):
    # Implementation of custom trust calculation
    pass
```

### Consensus Simulation

```python
# Future extension - code to simulate consensus protocols
# across the visualized blockchain structure
```

### Output Files

By default, the `blockchainNetworkAnalyzer.py` script creates four different blockchain visualizations:

1. `blockchain_mixed_networks.gif` - A blockchain with different network types per block
2. `blockchain_small_world.gif` - A blockchain composed entirely of small-world networks
3. `blockchain_scale_free.gif` - A blockchain composed entirely of scale-free networks
4. `blockchain_community.gif` - A blockchain composed entirely of community structure networks

All outputs are saved directly to the Blockchain Simulation folder.

## Integration with Other Algorithms

The blockchain network analyzer implementation builds on concepts from other algorithms in the project:

- **Community Optimization**: Uses community detection techniques similar to those in the Community Optimization module
- **Trust Partitioning**: Applies trust analysis methods inspired by the Trust Partitioning module
- **Network Graph Generator**: Uses various network generation models explored in the Graph Generator module
- **Generalized Permutations**: Can be extended to incorporate permutation-based shuffling for network generation

This integration creates a more comprehensive blockchain analysis framework that leverages the strengths of multiple algorithmic approaches.

---

## Future Directions

- Integration with actual blockchain transaction data
- Simulating attack scenarios and resilience testing
- Quantum-enhanced consensus models for blockchain layers
- Dynamic block generation based on transaction validation patterns
- Trust-weighted connection visualization between blocks
- Temporal analysis of blockchain evolution over extended periods
- Advanced community detection for multi-layer blockchain networks
- Incorporating concepts from topological data analysis
- Integration with other network analysis modules in the project
- Comparative analysis of different consensus mechanisms across network types

---

## References

- Nakamoto, S. (2008). Bitcoin: A peer-to-peer electronic cash system. *Decentralized Business Review*.
- Pass, R., & Shi, E. (2017). Hybrid consensus: Efficient consensus in the permissionless model. *31st International Symposium on Distributed Computing (DISC 2017)*.
- Wang, W., Hoang, D. T., Xiong, Z., Niyato, D., Wang, P., Hu, P., & Wen, Y. (2019). A survey on consensus mechanisms and mining management in blockchain networks. *IEEE Access*, 7, 22328-22370.
- Bano, S., Sonnino, A., Al-Bassam, M., Azouvi, S., McCorry, P., Meiklejohn, S., & Danezis, G. (2019). SoK: Consensus in the age of blockchains. *Proceedings of the 1st ACM Conference on Advances in Financial Technologies*.
- Morais, J. et al. Topological Consensus Networks for Cryptographic Proof.

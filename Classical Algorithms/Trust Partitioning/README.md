# Trust Partitioning with Topological Analysis

> A Python implementation of the Topological Network Partitioning algorithm for trust-based redistribution of nodes across consensus networks with advanced topological analysis.

---

## Table of Contents

- [Introduction](#introduction)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Mathematical Background](#mathematical-background)
  - [Trust Partitioning](#trust-partitioning)
  - [Topological Analysis](#topological-analysis)
- [Usage](#usage)
  - [Basic Trust Partitioning](#basic-trust-partitioning)
  - [Topological Analysis](#topological-analysis-usage)
  - [Combined Approach](#combined-approach)
- [Implementation Details](#implementation-details)
  - [Trust-Based Algorithm](#trust-based-algorithm)
  - [Topological Computation](#topological-computation)
  - [Visualization](#visualization)
- [Core Files](#core-files)
- [References](#references)

---

## Introduction

The **Trust Partitioner** algorithm dynamically repartitions a collection of consensus networks based on pairwise trust values and per-node security thresholds. Inspired by the _Topological Consensus Networks_ framework, this linear-time algorithm allows nodes to "jump" to more trusted networks or "abandon" their network when their internal trust falls below their own threshold.

This implementation extends the trust partitioning framework with computational topology techniques, specifically **persistent homology**, to analyze and partition networks based on their topological features. By incorporating topological analysis, the framework can capture higher-dimensional structures like loops (cycles), voids, and their persistence across different scales.

---

## Key Features

- **Trust‐driven partitioning**: Nodes evaluate trust sums to decide optimal network jumps.  
- **Jump & abandon phases**: Two‐step reconfiguration—migration followed by isolation.  
- **Linear complexity**: Scalable to large numbers of networks and nodes.  
- **Persistent homology**: Captures higher-dimensional topological features in networks.
- **Topological compatibility**: Enhances trust-based decisions with topological similarity.
- **Advanced visualization**: Visualize network topology with persistence diagrams and Betti curves.
- **Pluggable trust matrices**: Works with any precomputed asymmetric trust matrix.  
- **Python 3 only**: Pure‐Python implementation with standard dependencies.

---

## Installation

```bash
git clone https://github.com/btq-ag/TrustPartitioner.git
cd TrustPartitioner
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.6+
- NumPy, NetworkX, Matplotlib
- GUDHI library for persistent homology computation
- SciPy

---

## Mathematical Background

### Trust Partitioning

The trust partitioning framework operates on a collection of networks where nodes have pairwise trust relations. It uses two key steps:

1. **Jump Step**: Nodes compute their perspective of every other network and jump if:
   - The target network yields strictly lower average distrust
   - The network accepts the node based on its security threshold

2. **Abandon Step**: Nodes whose internal trust falls below their personal threshold form singletons

### Topological Analysis

Computational topology adds another dimension to network analysis:

1. **Simplicial Complexes**:
   - Represent higher-dimensional structures beyond the graph (edges)
   - Used to capture higher-order relationships between nodes

2. **Persistent Homology**:
   - Quantifies the topological features of a network across different scales
   - Measures:
     - β₀: Connected components
     - β₁: Cycles/loops
     - β₂: Voids/cavities

3. **Filtration**:
   - Process of growing the simplicial complex as the scale parameter increases
   - Allows tracking the birth and death of topological features

4. **Persistence Diagrams**:
   - Visualize the birth and death times of topological features
   - Significant features appear far from the diagonal

5. **Betti Numbers**:
   - β₀: Number of connected components
   - β₁: Number of 1-dimensional holes (cycles)
   - β₂: Number of 2-dimensional voids
   - βₙ: Number of n-dimensional features

---

## Usage

### Basic Trust Partitioning

```python
from trustPartitioner import networkPartitioner

# Define your consensus networks as lists of node IDs
networks = [
    [0, 1, 2],
    [3, 4],
    [5, 6, 7],
    [8, 9]
]

# Define the full system‐level trust matrix (10×10 NumPy array)
import numpy as np
systemTrust = np.random.rand(10, 10)
np.fill_diagonal(systemTrust, 0)  # zero self‐trust

# Define per‐node security thresholds
nodeSecurity = {i: 0.3 + 0.01*i for i in range(10)}

# Define per‐network security (most lenient half of member thresholds)
netSecurity = {i: max(sorted([nodeSecurity[n] for n in net])[len(net)//2:])
               for i, net in enumerate(networks)}

# Run partitioner
newNetworks, _, _, _ = networkPartitioner(
    networks,
    systemTrust,
    nodeSecurity,
    netSecurity
)
print("Repartitioned networks:", newNetworks)
```

### Topological Analysis Usage

```python
from topologicalPartitioner import compute_betti_numbers, extract_topological_features
import networkx as nx
import numpy as np

# Create a graph
G = nx.cycle_graph(5)  # A cycle graph has β₁ = 1

# Convert to distance matrix
A = nx.to_numpy_array(G)
D = 1 - A  # Convert adjacency to distance
np.fill_diagonal(D, 0)  # Self-distance = 0

# Compute persistent homology
result = compute_betti_numbers(D, max_dimension=2, max_edge_length=0.5)

# Extract features
features = extract_topological_features(D, max_dimension=2)

print(f"Betti numbers: β₀={features['betti0']}, β₁={features['betti1']}, β₂={features['betti2']}")
```

### Combined Approach

```python
# Import required modules
from topologicalPartitioner import topological_partitioner, compute_betti_numbers
import numpy as np

# Define example networks
networks = [[0, 1, 2], [3, 4], [5, 6, 7]]  # Three networks
n_nodes = sum(len(net) for net in networks)

# Create a trust matrix
trust_matrix = np.random.rand(n_nodes, n_nodes)
np.fill_diagonal(trust_matrix, 0)  # No trust to self

# Define security parameters
node_security = {i: np.random.uniform(0.1, 0.5) for i in range(n_nodes)}
network_security = {i: np.random.uniform(0.2, 0.6) for i in range(len(networks))}

# Run the topological partitioner
new_networks, _, _, _, topo_results = topological_partitioner(
    networks, trust_matrix, node_security, network_security, 
    max_dimension=2, topo_weight=0.5  # 50% weight on topology vs. trust
)

# Access topological results
global_features = topo_results["global_features"]
print(f"Global Betti numbers: β₀={global_features['betti0']}, " +
      f"β₁={global_features['betti1']}, β₂={global_features['betti2']}")
```

---

## Implementation Details

### Trust-Based Algorithm

```python
def r(trustAinB: np.ndarray, forward: bool = True) -> np.ndarray:
    """
    Compute r_i(B) or r_B(i):
      - forward=True:   r_i(B) = (1/|B|) * sum_j trustAinB[i,j]
      - forward=False:  r_B(i) = (1/|A|) * sum_j trustAinB[j,i]
    """
    size = trustAinB.shape[0] if forward else trustAinB.shape[1]
    data = trustAinB if forward else trustAinB.T
    return np.sum(data, axis=1) / size
```

The Jump Step implementation:

```python
# Inside networkPartitioner:
for n, net in enumerate(inputNetworks):
    # compute node‐to‐network trust sums for all candidates
    for node in net.copy():
        trusts = [r(block, True)[idx] for block in blockMatrices]
        best = int(np.argmin(trusts))
        if best != n and networkSums[best][idx] <= networksSecurity[best]:
            networksCopy[n].remove(node)
            networksCopy[best].append(node)
```

### Topological Computation

The implementation uses the GUDHI library to compute persistent homology:

```python
def compute_betti_numbers(distance_matrix, max_dimension=2, max_edge_length=1.0):
    # Initialize the Rips complex
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    
    # Create a simplex tree with simplices up to dimension max_dimension+1
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension+1)
    
    # Compute persistent homology
    persistence = simplex_tree.persistence()
    
    # Calculate Betti numbers
    betti_numbers = {}
    persistence_dict = {}
    
    for dim in range(max_dimension+1):
        persistence_intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        
        # Count intervals that are still alive at max_edge_length
        betti_count = sum(1 for (birth, death) in persistence_intervals 
                         if birth <= max_edge_length and 
                         (death > max_edge_length or death == float('inf')))
        
        betti_numbers[f'betti{dim}'] = betti_count
        persistence_dict[f'dim{dim}'] = persistence_intervals
    
    return {
        'betti_numbers': betti_numbers,
        'persistence': persistence_dict,
        'simplex_tree': simplex_tree
    }
```

### Visualization

The implementation provides multiple visualization types:

1. **Persistence Diagrams**: Plot birth vs. death times of topological features
2. **Betti Curves**: Plot how Betti numbers change with filtration parameter
3. **Network with Topology**: Visualize networks with topological features highlighted
4. **Partitioning Animations**: Show how networks evolve through the partitioning process

---

## Core Files

1. **Core Implementation**:
   - `trustPartitioner.py`: Basic trust-based partitioning algorithm
   - `topologicalPartitioner.py`: Main implementation of persistent homology and topological partitioning
   - `visualizeTopology.py`: Basic visualization tools for topological features
   - `visualize_topology_improved.py`: Enhanced visualization with better error handling
   - `advanced_topological_analysis.py`: Advanced tools for comparing different network types
   - `topological_concepts.py`: Educational visualizations explaining topological concepts

---

## References

1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*.
2. Carlsson, G. (2009). *Topology and data*.
3. Horak, D., Maletić, S., & Rajković, M. (2009). *Persistent homology of complex networks*.

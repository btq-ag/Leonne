![BTQ Logo](../Logos/BTQ\ logo_extended.png)

# BTQ Algorithms Hub

A comprehensive interface and visualizer for BTQ's classical and quantum algorithms, providing a unified way to access, run, and visualize all the algorithms from both the Classical Algorithms and Quantum Algorithms modules.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Classical Algorithms Hub](#classical-algorithms-hub)
- [Quantum Algorithms Hub](#quantum-algorithms-hub)
- [Available Commands](#available-commands)
- [Example Use Cases](#example-use-cases)
- [Visualization Types](#visualization-types)
- [Original Visualizers](#original-visualizers)
- [Integration with Other Modules](#integration-with-other-modules)
- [Advanced Usage](#advanced-usage)

---

## Overview

The BTQ Algorithms Hub is a centralized interface for accessing, exploring, and visualizing the various classical and quantum algorithms implemented in the BTQ project. It serves as a master control module that allows users to run and visualize algorithms from different modules without having to navigate through individual implementation files.

This hub brings together algorithms from multiple domains including:

- Network Graph Generation
- Consensus Mechanisms
- Trust Partitioning
- Topological Analysis
- Blockchain Simulation
- Community Optimization
- Generalized Permutations

The hub is divided into two main components:
1. **Classical Algorithms Hub**: Provides access to traditional classical algorithms
2. **Quantum Algorithms Hub**: Provides access to quantum-enhanced algorithms

---

## Getting Started

### Prerequisites

The hub requires the following Python packages:

```bash
pip install numpy matplotlib networkx tqdm scipy
```

For topological algorithms, additional packages may be required:

```bash
pip install gudhi
```

### Running the Hub

There are four main ways to use the BTQ Algorithms Hub:

1. **Through the Classical Command-Line Interface**:

```bash
python visualizer_cli.py
```

2. **Through the Quantum Command-Line Interface**:

```bash
python quantum_visualizer_cli.py
```

3. **Programmatically using the Classical Hub**:

```python
from classical_hub import ClassicalAlgorithmsHub

# Initialize the hub
hub = ClassicalAlgorithmsHub()

# List available algorithms
algorithms = hub.list_algorithms()
```

4. **Programmatically using the Quantum Hub**:

```python
from quantum_hub import QuantumAlgorithmsHub

# Initialize the hub
hub = QuantumAlgorithmsHub()

# List available algorithms
algorithms = hub.list_algorithms()
```

## Architecture

The BTQ Algorithms Hub is designed around four core components:

1. **`classical_hub.py`**: The core module that provides programmatic access to all classical algorithms
2. **`visualizer_cli.py`**: A command-line interface for exploring and running classical algorithms
3. **`quantum_hub.py`**: The core module that provides programmatic access to all quantum algorithms
4. **`quantum_visualizer_cli.py`**: A command-line interface for exploring and running quantum algorithms

The hub dynamically loads all Python modules from both the Classical Algorithms and Quantum Algorithms directories and makes their functions and classes available through a unified interface. It discovers algorithms at runtime, so new algorithms added to either directory are automatically available without modifying the hub code.

### Key Classes

- **`ClassicalAlgorithmsHub`**: The main class that discovers, loads, and provides access to all classical algorithms
- **`VisualizerCLI`**: The command-line interface for classical algorithms
- **`QuantumAlgorithmsHub`**: The main class that discovers, loads, and provides access to all quantum algorithms
- **`QuantumVisualizerCLI`**: The command-line interface for quantum algorithms

## Classical Algorithms Hub

The Classical Algorithms Hub provides access to traditional classical algorithms in the following categories:

- Network Graph Generator
- Consensus Iterations
- Trust Partitioning
- Community Optimization
- Generalized Permutations
- Blockchain Simulation

### Example Usage

```python
from classical_hub import ClassicalAlgorithmsHub

# Initialize the hub
hub = ClassicalAlgorithmsHub()

# List available algorithms
categories = hub.list_categories()
print("Available categories:", categories)

# Get algorithms in a category
trust_modules = hub.list_modules("Trust Partitioning")
print("Trust Partitioning modules:", trust_modules)

# Run a specific algorithm
network_partitioner = hub.get_algorithm("Trust Partitioning.trustPartitioner.networkPartitioner")
networks = [[0,1,2],[3,4],[5,6,7]]
trust_matrix = hub.run_algorithm("Network Graph Generator.networkGenerator.generate_random_trust_matrix", 8)
node_security = {i: 0.3+0.05*i for i in range(8)}
network_security = {i: 0.5 for i in range(3)}

new_networks, jumps, abandons = network_partitioner(networks, trust_matrix, node_security, network_security)
```

## Quantum Algorithms Hub

The Quantum Algorithms Hub provides access to quantum-enhanced algorithms in the following categories:

- Network Graph Generator
- Consensus Iterations
- Trust Partitioning
- Community Optimization
- Generalized Permutations
- Blockchain Simulation

Each quantum algorithm incorporates quantum principles such as:
- Quantum randomness
- Quantum entanglement simulation
- Quantum key distribution
- Quantum-inspired optimization

### Example Usage

```python
from quantum_hub import QuantumAlgorithmsHub

# Initialize the hub
hub = QuantumAlgorithmsHub()

# List available algorithms
categories = hub.list_categories()
print("Available categories:", categories)

# Get algorithms in a category
quantum_blockchain_modules = hub.list_modules("Blockchain Simulation")
print("Quantum Blockchain modules:", quantum_blockchain_modules)

# Run a specific algorithm
quantum_blockchain_visualizer = hub.get_algorithm(
    "Blockchain Simulation.quantumBlockchainVisualizer.create_quantum_blockchain_visualization"
)
# Create a quantum blockchain visualization with 5 blocks, each containing 12 nodes
quantum_blockchain_visualizer(
    n_blocks=5, 
    n_nodes_per_block=12, 
    n_frames=30,
    quantum_ratio=0.6,  # 60% of nodes are quantum-enabled
    show_compliance=True  # Show the QKD compliance graph
)
```

## Available Commands

When using the command-line interface (`visualizer_cli.py`), the following commands are available:

| Command | Description |
|---------|-------------|
| `categories` | List all algorithm categories |
| `modules [category]` | List modules in a category |
| `algos [category] [module]` | List algorithms in a category/module |
| `select category module algorithm` | Select an algorithm |
| `info [algorithm]` | Show information about an algorithm |
| `run [args...]` | Run the selected algorithm with arguments |
| `examples` | Show example commands for common tasks |
| `visualize [type]` | Run a predefined visualization |
| `clear` | Clear the screen |
| `help` | Show this help message |
| `exit` | Exit the program |

## Example Use Cases

### 1. Classical Trust Partitioner

```python
# CLI Method
> select Trust Partitioning trustPartitioner networkPartitioner
> run [[0,1,2],[3,4],[5,6,7]] np.random.rand(8,8) {i:0.3+0.05*i for i in range(8)} {i:0.5 for i in range(3)}

# Programmatic Method
from classical_hub import run_trust_partitioner
import numpy as np

networks = [[0,1,2],[3,4],[5,6,7]]
trust_matrix = np.random.rand(8,8)
node_security = {i: 0.3+0.05*i for i in range(8)}
network_security = {i: 0.5 for i in range(3)}

new_networks, jumps, abandons = run_trust_partitioner(
    networks, trust_matrix, node_security, network_security
)
```

### 2. Quantum Trust Partitioner

```python
# CLI Method
> select Trust Partitioning quantum_trust_partitioner quantumNetworkPartitioner
> run [[0,1,2],[3,4],[5]] {0:0.3,1:0.4,2:0.3,3:0.6,4:0.5,5:0.2} 0.5 16

# Programmatic Method
from quantum_hub import run_quantum_trust_partitioner

networks = [[0,1,2],[3,4],[5]]
q_sec = {0:0.3,1:0.4,2:0.3,3:0.6,4:0.5,5:0.2}

final, qTrust = run_quantum_trust_partitioner(
    networks,
    q_sec,
    complianceThresh=0.5,   # QKD bit-overlap threshold
    bitLength=16            # QRiNG string length
)
```

### 3. Classical Blockchain Visualization

```python
# CLI Method
> select Blockchain Simulation blockchainVisualizer create_blockchain_network_visualization
> run 5 10 30

# Programmatic Method
from classical_hub import run_blockchain_visualization

run_blockchain_visualization(n_blocks=5, n_nodes_per_block=10, n_frames=30)
```

### 4. Quantum Blockchain Visualization

```python
# CLI Method
> select Blockchain Simulation quantumBlockchainVisualizer create_quantum_blockchain_visualization
> run 5 12 30 0.6 True

# Programmatic Method
from quantum_hub import run_quantum_blockchain_visualization

run_quantum_blockchain_visualization(
    n_blocks=5, 
    n_nodes_per_block=12, 
    n_frames=30,
    quantum_ratio=0.6,  # 60% of nodes are quantum-enabled
    show_compliance=True  # Show the QKD compliance graph
)

# Generate analysis plots comparing quantum and classical approaches
from quantum_hub import QuantumAlgorithmsHub
hub = QuantumAlgorithmsHub()

create_entropy_plot = hub.get_algorithm(
    "Blockchain Simulation.quantumBlockchainVisualizer.create_quantum_vs_classical_entropy_plot"
)
create_security_comparison = hub.get_algorithm(
    "Blockchain Simulation.quantumBlockchainVisualizer.create_quantum_security_comparison"
)

create_entropy_plot()
create_security_comparison()
```

## Visualization Types

The hub provides several predefined visualizations that demonstrate key algorithms for both classical and quantum approaches:

### Classical Visualizations

| Type | Description |
|------|-------------|
| `consensus_network` | Visualizes a consensus network with nodes colored by their state |
| `blockchain` | Creates a 3D visualization of a blockchain with different network topologies per block |
| `topological_partitioning` | Visualizes the process of partitioning a network using topological features |
| `network_generator` | Demonstrates the consensus shuffle algorithm on a network edge set |

### Quantum Visualizations

| Type | Description |
|------|-------------|
| `quantum_consensus_network` | Visualizes a quantum-enhanced consensus network with quantum-secure connections |
| `quantum_blockchain` | Creates a 3D visualization of a quantum blockchain with quantum-enhanced security features |
| `quantum_permutations` | Demonstrates quantum-enhanced Fisher-Yates shuffle with improved randomness |
| `quantum_communities` | Visualizes quantum-enhanced community detection in complex networks |
| `quantum_graph` | Shows quantum-inspired network evolution with quantum random connections |

To run a visualization, use the `visualize` command followed by the type:

```
# Classical visualization
> visualize blockchain

# Quantum visualization
> visualize quantum_blockchain
```

## Original Visualizers

In addition to the integrated Classical Algorithms Hub, this directory also contains the original standalone visualizers:

### 🔗 Partition Evolution Visualizer

Simulates nodes "jumping" between clusters or being "abandoned," drawing each cluster as a fully connected subgraph over time.

![Partition Evolution](https://raw.githubusercontent.com/btq-ag/Leonne/main/Visualizer/gif1_partition_evolution.gif)

<details>
<summary>Mathematical Model</summary>

Maintain a partition of nodes \(V=\{v_1,\dots,v_n\}\) into clusters \(C_i\). At each step:
\[
\begin{aligned}
\text{Jump: }&v\in C_a \;\to\; C_a\setminus\{v\},\;C_b\cup\{v\},\\
\text{Abandon: }&v\to E\ (\text{sink cluster}).
\end{aligned}
\]
</details>

```python
from partition_visualizer import PartitionEvolution

clusters = {'A':[0,1,2],'B':[3,4],'C':[5,6,7],'D':[8,9]}
events = [
    ('jump',0,'B'),
    ('jump',3,'D'),
    ('abandon',4,None),
    # ...
]
positions = {'A':(-4,2),'B':(0,2),'C':(-4,-2),'D':(0,-2),'E':(4,0)}

viz = PartitionEvolution(clusters, events, positions, interval=800)
viz.animate("gif1_partition_evolution.gif")
```

### 🌐 Worldsheet of Network Histories Visualizer

Renders a **cobordism** of graph snapshots in 3D: each graph at time $t$ is drawn in plane $z=t$, forming a continuous "worldsheet."

![Worldsheet of Network Histories](https://raw.githubusercontent.com/btq-ag/Leonne/main/Visualizer/gif2_worldsheet.gif)

<details>
<summary>Mathematical Model</summary>

Given $G_t=(V,E_t)$, the worldsheet is

$$
W = \bigcup_{t=0}^T (G_t \times \{t\})
\subset \mathbb{R}^2\times[0,T],
$$

with edges $(u,v)\in E_t$ at height $z=t$.

</details>

```python
from worldsheet_visualizer import Worldsheet

# reuse clusters/events as above
viz = Worldsheet(clusters, events, positions, duration=10, fps=5)
viz.animate("gif2_worldsheet.gif")
```

### 📈 Network Complex Persistence Visualizer

Builds a Vietoris–Rips complex on random points in $[-1,1]^2$, showing:

* **Left**: edges with $\|x_i-x_j\|\le r$.
* **Center**: H₀ persistence barcode (merge radii $d_i$).
* **Right**: persistence landscape

![Network Complex Persistence](https://raw.githubusercontent.com/btq-ag/Leonne/main/Visualizer/network_complex_with_landscape_v2.gif)

### 🔬 Community Structure Visualizer

Visualizes different types of network community structures and their evolution, showing how nodes organize into communities based on connectivity patterns.

![Community Networks](https://raw.githubusercontent.com/btq-ag/Leonne/main/Visualizer/communityNetworks.png)

<details>
<summary>Mathematical Model</summary>

Communities are detected using modularity optimization, where the modularity $Q$ is defined as:

$$
Q = \frac{1}{2m}\sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j)$ is 1 if nodes $i$ and $j$ are in the same community and 0 otherwise.
</details>

## Integration with Other Modules

The BTQ Algorithms Hub is designed to work seamlessly with both the classical and quantum components of the BTQ project. It provides a unified framework that allows users to:

1. **Compare Classical and Quantum Approaches**: Run both classical and quantum versions of the same algorithm and compare their performance, security, and other metrics.
2. **Build Hybrid Systems**: Combine classical and quantum algorithms to create hybrid systems that leverage the strengths of both approaches.
3. **Transition from Classical to Quantum**: Provide a pathway for transitioning from classical implementations to quantum-enhanced ones as quantum technologies mature.

Integration points include:

- **Trust models**: Classical trust models can be enhanced with quantum randomness
- **Consensus mechanisms**: Hybrid classical-quantum consensus algorithms
- **Network partitioning**: Using quantum algorithms for optimized partitioning
- **Blockchain security**: Enhancing classical blockchains with quantum key distribution

## Advanced Usage

### 1. Extending with New Algorithms

To add a new algorithm to the hub, simply add your Python file to the appropriate subdirectory in either the Classical Algorithms or Quantum Algorithms directory. The hub will automatically discover and make available any functions or classes defined in your file.

### 2. Customizing Visualizations

For specialized visualizations, you can create custom visualization functions by extending either the `VisualizerCLI` or `QuantumVisualizerCLI` class:

```python
from quantum_visualizer_cli import QuantumVisualizerCLI

class CustomQuantumVisualizer(QuantumVisualizerCLI):
    def _visualize_custom_quantum_algorithm(self):
        # Implement your custom quantum visualization here
        pass
```

### 3. Batch Processing

For batch processing or automated workflows, you can use both hubs programmatically:

```python
from classical_hub import ClassicalAlgorithmsHub
from quantum_hub import QuantumAlgorithmsHub
import numpy as np

classical_hub = ClassicalAlgorithmsHub()
quantum_hub = QuantumAlgorithmsHub()

# Compare classical and quantum partitioning
classical_partitioner = classical_hub.get_algorithm("Trust Partitioning.trustPartitioner.networkPartitioner")
quantum_partitioner = quantum_hub.get_algorithm("Trust Partitioning.quantum_trust_partitioner.quantumNetworkPartitioner")

# Run comparative analysis
results = []
for i in range(10):
    trust_matrix = np.random.rand(8, 8)
    networks = [[0,1,2],[3,4],[5,6,7]]
    node_security = {i: 0.3+0.05*i for i in range(8)}
    network_security = {i: 0.5 for i in range(3)}
    
    # Classical run
    classical_networks, jumps, abandons = classical_partitioner(
        networks, trust_matrix, node_security, network_security
    )
    
    # Quantum run
    quantum_networks, quantum_trust = quantum_partitioner(
        networks, node_security, complianceThresh=0.5, bitLength=16
    )
    
    results.append({
        'classical': classical_networks,
        'quantum': quantum_networks
    })
```

---

## 🚀 Installation

```bash
git clone https://github.com/btq-ag/Leonne.git
cd Leonne/Visualizer
pip install -r requirements.txt
```

---

## 🖥️ Usage

Run any visualizer module:

```bash
# Classical visualizers
python visualizer_cli.py

# Quantum visualizers
python quantum_visualizer_cli.py

# Individual visualizers
python -m partition_visualizer --config config/partition.yaml
python -m worldsheet_visualizer --config config/worldsheet.yaml
python -m persistence_visualizer --points data/random.npy
python -m networkCommunities --config config/communities.yaml
```

Or import and call from your own script as shown above.

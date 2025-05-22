# Quantum Graph Generator Toolkit

> **QGGT** – A quantum-enhanced implementation that builds *valid bipartite graphs*
> from arbitrary degree sequences and explores the quantum probability landscape of
> edge-set permutations.  
> Part of the **Topological Consensus Networks (TCN)** quantum stack.

---

## Contents
1. [Why Quantum Edge Assignment?](#why-quantum-edge-assignment)  
2. [Feature Highlights](#feature-highlights)  
3. [Requirements](#requirements)  
4. [Hello, Quantum Bipartite World](#hello-quantum-bipartite-world)  
5. [Core Concepts](#core-concepts)  
   - Quantum-Enhanced Degree Sequences  
   - Quantum Feasibility Checks  
   - Quantum-Assisted Construction  
6. [Visualizations](#visualizations)  
   - Quantum Network Evolution Animations
   - Quantum vs Classical Edge Distribution
   - Quantum Permutation Stability Analysis
   - 3D Quantum Network Visualization
   - Quantum vs Classical Comparison
7. [Advanced Topics](#advanced-topics)  
   - Quantum Degenerate Edge Sets  
   - Quantum Probability Diagnostics

---

## Why Quantum Edge Assignment?

In Topological Consensus Networks, we need **balanced bipartite graphs** with enhanced randomness properties that are resistant to classical prediction. Quantum-inspired algorithms provide:

1. **Enhanced randomness** - Quantum-inspired randomization offers true unpredictability for security-critical applications
2. **Improved sampling efficiency** - Quantum-inspired sampling can explore the space of possible graphs more efficiently
3. **Entanglement modeling** - Allows us to model correlations between edges that are not possible with classical algorithms

A single integer sequence `[d₀,…,d_{u+v−1}]` uniquely specifies a graph **iff** certain constraints hold. Our quantum implementation enhances the classical approach by adding:

1. *Quantum verification* – check constraints with quantum-inspired algorithms
2. *Quantum construction* – produce edge sets with quantum-enhanced randomness
3. *Quantum exploration* – shuffle edges with quantum consensus mechanisms
4. *Quantum diagnostics* – visualize quantum vs classical probability distributions

---

## Feature Highlights

| 🔧  | Description |
| --- | ----------- |
| **Quantum verifier** | Quantum-enhanced check of Gale–Ryser inequalities. |
| **Quantum builder** | Non-deterministic construction of edge sets using quantum-inspired randomness. |
| **Quantum degeneracy removal** | `quantum_remove_degeneracy` leverages quantum concepts for improved uniqueness testing. |
| **Quantum permutation analytics** | `quantum_probability_distribution()` compares classical and quantum shuffling. |
| **Enhanced visualizations** | Advanced animations including 3D network evolution, stability analysis, and comparative visualizations. |
| **Quantum methods** | Multiple quantum-inspired methods (Hadamard, Phase, Bloch sphere) for diverse randomness properties. |

---

## Requirements

```bash
pip install numpy matplotlib sympy
```

Additional packages for quantum functionality are included directly in the implementation.

---

## Hello, Quantum Bipartite World

```python
from QuantumGraphVisualizer import quantum_sequence_verifier, quantum_edge_assignment

# degree sequence [U|V]  ->  U has 3 vertices, V has 3
deg_seq = [2,2,2, 2,3,1]   # length 6

u_size, v_size = 3, 3

if quantum_sequence_verifier(deg_seq, u_size, v_size):
    edges = quantum_edge_assignment(deg_seq, u_size, v_size)
    print("Quantum edge set:", edges)
else:
    print("Sequence not quantum-realisable!")
```

---

## Core Concepts

### Quantum-Enhanced Degree Sequences

Just as in classical graph theory, degree sequences define the connectivity of vertices. Our quantum approach uses quantum-inspired randomness to verify and construct these sequences, providing enhanced security properties for consensus networks.

### Quantum Feasibility Checks

The quantum verifier uses quantum-inspired algorithms to check the Gale-Ryser constraints, ensuring that a degree sequence can be realized as a bipartite graph. The quantum approach allows limited violations of constraints to enable a more complete exploration of the graph space.

### Quantum-Assisted Construction

Once verified, we use quantum-inspired algorithms to construct edge sets with enhanced randomness properties, ensuring they are resistant to classical prediction methods.

---

## Visualizations

The package includes enhanced visualizations that compare quantum and classical approaches:

### Quantum Network Evolution Animations

![Quantum Network Evolution](quantum_small_network_evolution_bloch.gif)

This advanced animation visualizes how quantum-inspired permutations evolve the edge set over time, with quantum-inspired visual effects. Different quantum methods (Bloch sphere, phase, etc.) produce distinct evolution patterns.

### Quantum vs Classical Edge Distribution

![Classical vs Quantum Distribution](classical_vs_quantum_distribution.png)

This visualization compares the statistical properties of edge values in classical and quantum graph assignments. The quantum approach shows a different distribution pattern due to the quantum-inspired randomness.

### Quantum Permutation Stability Analysis

![Quantum Permutation Stability](quantum_medium_permutation_stability.png)

This visualization shows how quantum permutations differ from classical ones in terms of stability across multiple shuffles, highlighting the unique properties of quantum-inspired randomness.

### 3D Quantum Network Visualization

![3D Quantum Network](quantum_medium_network_3d.gif)

This sophisticated 3D visualization demonstrates quantum effects like superposition and entanglement in network evolution, with quantum wave effects and probability clouds.

### Quantum vs Classical Comparison

![Quantum vs Classical Evolution](quantum_vs_classical_medium.gif)

Side-by-side animation comparing the evolution of networks under classical and quantum shuffling algorithms, highlighting the differences in behavior.

### Running Enhanced Visualizations

To generate all enhanced visualizations:

```python
# Run the enhanced visualizer
python run_quantum_visualizer.py
```

---

## Advanced Topics

### Quantum Degenerate Edge Sets

Just as in classical graph theory, multiple distinct edge sets can realize the same degree sequence. Our quantum approach provides methods to identify and analyze these degenerate configurations, leveraging quantum concepts to ensure better uniqueness properties.

### Quantum Probability Diagnostics

The quantum probability distribution analysis allows detailed examination of the statistical properties of quantum-enhanced edge assignments, providing insights into how they differ from classical approaches. This can be crucial for applications requiring true randomness, such as consensus network security.

# Quantum Blockchain Simulation

> A Python toolkit for visualizing and simulating quantum-enhanced blockchain network structures with quantum random number generation, quantum key distribution, and entropy addition for improved security and true randomness.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Quantum Enhancements](#quantum-enhancements)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Quantum Network Models](#quantum-network-models)  
6. [Visualizations](#visualizations)  
7. [Technical Details](#technical-details)  
8. [Comparison with Classical Blockchain](#comparison-with-classical-blockchain)  
9. [Future Directions](#future-directions)  
10. [References](#references)

---

## Introduction

The Quantum Blockchain Simulation module extends traditional blockchain visualization with quantum-enhanced features. It provides tools for modeling and visualizing how quantum phenomena can improve blockchain security, consensus mechanisms, and network resilience. By representing blockchain layers as quantum-enabled cylindrical structures, this module offers insights into the advantages quantum technology offers to distributed ledger systems.

This simulation supports the broader Quantum Consensus Networks (QCN) framework by illustrating how quantum resources like entanglement, superposition, and quantum randomness enhance the security and efficiency of distributed consensus.

---

## Quantum Enhancements

The quantum version of the blockchain simulation introduces several key quantum-inspired improvements:

1. **Quantum Random Number Generation (QRNG)**: True random number generation based on quantum measurement, replacing pseudo-random number generators for increased entropy and unpredictability in network connections.

2. **Quantum Key Distribution (QKD)**: Secure key exchange between nodes using quantum principles, enabling eavesdropping-resistant communication channels between blockchain blocks.

3. **Entropy Addition**: Combining classical and quantum randomness sources using bitwise XOR operations to ensure maximum entropy even when some sources are compromised.

4. **Compliance Graphs**: Visual representation of secure quantum communication channels, allowing validation of quantum communication integrity.

5. **Quantum Network Topologies**: Enhanced network models (random, small-world, scale-free, community, spatial) that incorporate quantum effects like tunneling and entanglement.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/btq-ag/QuantumBlockchainSimulation.git
cd QuantumBlockchainSimulation

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy networkx matplotlib tqdm
```

---

## Usage

To generate the quantum blockchain visualization:

```python
from quantumBlockchainVisualizer import create_quantum_blockchain_visualization

# Create a quantum blockchain visualization with 5 blocks, each containing 12 nodes
create_quantum_blockchain_visualization(
    n_blocks=5, 
    n_nodes_per_block=12, 
    n_frames=30,
    quantum_ratio=0.6,  # 60% of nodes are quantum-enabled
    show_compliance=True  # Show the QKD compliance graph
)

# Generate analysis plots comparing quantum and classical approaches
from quantumBlockchainVisualizer import create_quantum_vs_classical_entropy_plot, create_quantum_security_comparison

create_quantum_vs_classical_entropy_plot()
create_quantum_security_comparison()
```

Alternatively, run the scripts directly:

```bash
# Generate the blockchain visualization
python quantumBlockchainVisualizer.py

# Run comparative analysis between classical and quantum approaches
python quantumBlockchainAnalysis.py
```

The visualization script will generate:
- An animated GIF showing the blockchain structure with quantum features
- A still image of the final frame
- Analysis plots comparing quantum vs. classical entropy and security

The analysis script generates additional comparative visualizations:
- Randomness quality comparison (distribution and autocorrelation)
- QKD security analysis under different noise conditions
- Compliance graph security analysis
- Entropy addition benefits
- Consensus security probabilities
- Feature comparison table

All outputs are saved to the respective output directories (`quantum_network_animations` and `quantum_analysis`).

---

## Quantum Network Models

Each blockchain layer implements one of five quantum-enhanced network models:

1. **Quantum-Random Networks**: Uses true quantum randomness for connections, providing higher entropy than classical random networks with pseudo-random number generators.

2. **Quantum Small-World Networks**: Enhances the small-world network model with quantum long-range connections, enabling "quantum tunneling" between otherwise distant nodes.

3. **Quantum Scale-Free Networks**: Incorporates quantum preference attachment, where quantum nodes serve as preferred hubs, creating robustness against targeted attacks.

4. **Quantum Community Networks**: Implements quantum bridges between communities using quantum entanglement, facilitating secure inter-community consensus.

5. **Quantum Spatial Networks**: Adds quantum tunneling effects to geometric networks, allowing connections beyond typical distance constraints.

These quantum enhancements provide stronger security guarantees, increased randomness for consensus algorithms, and novel connection patterns impossible in classical networks.

---

## Visualizations

The toolkit generates three types of visualizations:

### 1. Blockchain Network Animation

An animated 3D visualization showing:
- Cylindrical blockchain layers with different quantum network topologies
- Distinct visual representation of quantum vs. classical nodes
- Quantum key distribution effects shown along connections
- Progressive building of the blockchain structure
- Visualization of the compliance graph showing QKD validation

### 2. Entropy Comparison Plot

A bar chart comparing:
- The entropy of classical pseudo-random number generation
- The entropy of quantum random number generation
- The theoretical maximum entropy (ideal distribution)

### 3. Security Scaling Comparison

A logarithmic plot showing:
- How security bit requirements scale with network size for classical networks
- How quantum-enhanced networks reduce security overhead
- The quantum advantage ratio as network size increases

---

## Technical Details

### Quantum Random Number Generation (QRNG)

In real quantum systems, QRNG relies on inherently unpredictable quantum phenomena such as:
- Photon path superposition through beam splitters
- Vacuum fluctuations
- Radioactive decay

Our simulation approximates this behavior with added quantum-like noise to represent measurement uncertainty. True QRNGs provide:
- Non-deterministic output
- Higher entropy per bit than classical PRNGs
- Resistance to prediction and algorithmic vulnerabilities

### Quantum Key Distribution (QKD)

We simulate the BB84 protocol:
1. Sender encodes random bits in random bases (X or Z)
2. Receiver measures in randomly chosen bases
3. Bases are compared, and matching-basis measurements are kept
4. A subset of bits are checked for eavesdropping
5. Remaining bits form a secure shared key

The simulation includes:
- Channel noise effects
- Eavesdropping detection
- Secure flag determination based on error rates

### Entropy Addition

Combining randomness sources uses the XOR operation:
- If X and Y are independent random sources, X⊕Y has entropy at least as high as max(H(X), H(Y))
- This ensures that even if some randomness sources are compromised, the combined source remains secure
- Implemented through bitwise XOR for bit strings and modular addition for floating-point values

### Compliance Graphs

Compliance graphs visualize which QKD links are validated as secure:
- Nodes represent blockchain participants
- Edges represent validated quantum communication channels
- Edge weight indicates security level
- Graph analysis reveals the fully-connected secure subgraph

---

## Comparison with Classical Blockchain

### Security Improvements

| Feature | Classical Blockchain | Quantum Blockchain |
|---------|---------------------|-------------------|
| Randomness Source | Pseudo-random algorithms (deterministic) | True quantum randomness (non-deterministic) |
| Key Exchange | Vulnerable to computational advances | Information-theoretically secure (QKD) |
| Consensus Security | O(n) security bit scaling | Improved scaling through quantum resources |
| Attack Resistance | Vulnerable to computational attacks | Resistant to both classical and most quantum attacks |
| Node Selection | Potentially predictable | Truly random and unpredictable |

### Visualization Differences

- **Classical**: Focuses on network topology and block connectivity
- **Quantum**: Highlights quantum nodes, secure QKD links, and compliance validation

### Performance Considerations

- Quantum enhancements come with additional computational overhead in simulation
- In real-world implementation, quantum hardware would be required for true benefits
- Hybrid solutions (classical networks with quantum randomness sources) offer a practical transition path

---

## Future Directions

1. **Quantum Zk-SNARKs**: Implement quantum zero-knowledge proofs for enhanced privacy

2. **Quantum Consensus Algorithms**: Develop full quantum consensus mechanisms based on entanglement

3. **Post-Quantum Cryptography Integration**: Combine with post-quantum cryptographic methods for comprehensive security

4. **Hardware Integration**: Adapt simulation to interface with actual quantum hardware (IBMQ, D-Wave, etc.)

5. **Scalability Analysis**: Study how quantum resources scale with increasing network size and complexity

---

## References

1. Bennett, C.H., Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing. Proceedings of IEEE International Conference on Computers, Systems and Signal Processing, 175-179.

2. Ekert, A.K. (1991). Quantum cryptography based on Bell's theorem. Physical Review Letters, 67(6), 661-663.

3. Quantum Consensus Networks. (2023). Distributed Consensus Network Documentation.

4. Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379-423.

5. Impagliazzo, R., Levin, L.A., Luby, M. (1989). Pseudo-random generation from one-way functions. Proceedings of the Twenty-First Annual ACM Symposium on Theory of Computing, 12-24.

6. Singh, S., et al. (2023). Proof-of-work based on quantum sampling problems. Quantum Network Research.

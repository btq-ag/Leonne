# Distributed Consensus Networks (DCN)

## Overview

This implementation provides a complete framework for Distributed Consensus Networks (DCN) as described in the research papers by Jeffrey Morais. DCN is an alternative to inefficient proof-of-work and proof-of-stake blockchain systems, offering a secure, computationally efficient approach to distributed consensus.

In the DCN framework, a closed network of known nodes is partitioned into secure, random consensus subsets using shared randomness and cryptographic protocols. The resulting proofs-of-consensus are timestamped and cryptographically verifiable, serving as a commodity independent of any stake.

## Key Features

- **Efficient Consensus**: Unlike proof-of-work, DCN allocates all nodes to consensus sets, ensuring full resource utilization
- **Security**: Random subset allocation cannot be compromised unless all nodes collude
- **Self-Enforcement**: Protocol follows self-enforcing rules where adversarial behavior results in self-exclusion
- **Cryptographic Proofs**: Signature-sets produced by the protocol act as timestamped, cryptographic proofs-of-consensus
- **Strategic Robustness**: Highly robust against collusive adversaries, offering subnetworks defense against denial-of-service and majority takeover

## Components

The DCN implementation consists of the following key components:

1. **ConsensusNode** (`consensus_node.py`): Core node implementation with cryptographic functions, consensus protocol steps, and proof-of-consensus generation

2. **ConsensusNetwork** (`consensus_network.py`): Network manager that coordinates nodes, facilitates communication, and executes the consensus protocol phases

3. **ConsensusVisualizer** (`consensus_visualizer.py`): Visualization utilities for network topology, consensus sets, and protocol dynamics

4. **ConsensusSecurityAnalyzer** (`consensus_security.py`): Security analysis tools to evaluate network robustness, simulate attacks, and analyze protocol security

5. **DCN Driver** (`dcn_driver.py`): Main driver script demonstrating the DCN implementation with examples and interactive mode

## Protocol Phases

The DCN protocol operates in the following sequential phases:

1. **Bid Phase**: Nodes create bid requests for consensus on transactions
2. **Accept Phase**: Nodes accept or reject bids based on validation rules
3. **Consensus Phase**: Global random seed is computed and used to determine consensus sets
4. **Vote Phase**: Nodes in consensus sets vote on transactions
5. **Compliance Phase**: Nodes vote on the compliance of other nodes
6. **Proof Phase**: Compliant nodes generate proofs of consensus
7. **Ledger Phase**: Transaction ledger is updated based on consensus

## Usage

### Basic Usage

```python
from consensus_node import ConsensusNode
from consensus_network import ConsensusNetwork
from consensus_visualizer import ConsensusVisualizer

# Create a consensus network with 10 nodes
network = ConsensusNetwork(num_nodes=10)

# Submit a transaction
tx_data = {'sender': 'user_1', 'receiver': 'user_2', 'amount': 50}
tx_id = network.submit_transaction(tx_data)

# Run a consensus round
results = network.run_consensus_round()

# Visualize the network and consensus
visualizer = ConsensusVisualizer(network)
visualizer.plot_network_topology()
visualizer.plot_consensus_sets(results['consensus_sets'])
```

### Running the Demo

The interactive demo can be run using the `dcn_driver.py` script:

```
python dcn_driver.py --interactive
```

Other command-line options include:

- `--nodes N`: Set the number of nodes in the network
- `--transactions N`: Set the number of transactions to create
- `--rounds N`: Set the number of consensus rounds to run
- `--visualize`: Generate visualizations
- `--security`: Perform security analysis

Example:

```
python dcn_driver.py --nodes 15 --transactions 20 --rounds 5 --visualize --security
```

## Mathematical Foundation

The DCN framework is built on several mathematical concepts:

1. **Random Subset Sampling**: Using cryptographic techniques to securely allocate nodes into random consensus sets
2. **Distributed Randomness**: Nodes contribute random seeds which are combined to form a global seed that cannot be manipulated
3. **Consensus Assignment Graphs**: Representing the assignment of nodes to consensus sets as graph structures
4. **Strategic Robustness**: Analyzing the security of the network against various attack strategies

## Security Features

The implementation includes robust security features:

- **Cryptographic Signatures**: All messages and transactions are cryptographically signed
- **Compliance Voting**: Nodes vote on the compliance of other nodes to detect misbehavior
- **Attack Simulation**: The security analyzer can simulate various attack scenarios
- **Strategic Analysis**: Tools for evaluating network vulnerability and security metrics

## References

1. DCN: Distributed Consensus Networks
2. DCN Blockchains: Decentralized Cryptographic Proofs for Blockchains
3. Topological Consensus Networks, Jeffrey Morais

## Author

Jeffrey Morais, BTQ

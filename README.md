![ui](Logos/logo_extended.png)

<h4 align="center">Modular Consensus Networks</h4>


<div align="center">

[![CI](https://github.com/btq-ag/Leonne/actions/workflows/ci.yml/badge.svg)](https://github.com/btq-ag/Leonne/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![BTQ Core](https://img.shields.io/badge/BTQ-Core-ffd700?style=flat&logo=github)](https://github.com/btq-ag)
[![Léonne Stable](https://img.shields.io/badge/L%C3%A9onne-Stable-brightgreen?style=flat&logo=abstract&logoColor=white)](https://github.com/btq-ag/Leonne)
[![BTQ Site](https://img.shields.io/badge/BTQ-Site-0052cc?style=flat&logo=digitalocean&logoColor=white)](https://www.btq.com/)

</div>

Léonne (IPA: /leɪˈɔn/) is a collection of topological algorithms that computes cryptographic proof for multiple simulated blockchain transactions using modular consensus networks. It is designed with post-quantum protocols that take into account network histories and dynamically partitions networks to optimize consensus rates while being robust to dishonest parties colluding to have majority.  

## Table of Contents

- [Getting Started](#getting-started)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)


## Getting Started

Léonne enables you to run consensus and network partitioning algorithms for a collection of known consensus networks with the goal of performing parallel cryptographic proof for simulated multi-blockchain environments. 

- Visit our [algorithms modules](https://github.com/btq-ag/Leonne/tree/main/Classical%20Algorithms) to get started with consensus protocols.
- Visit the [theoretical showcase](https://github.com/btq-ag/Leonne/tree/main/Documentation) which represents the mathematical and algorithmic background needed to run the algorithms.

## Features

* Built-in topological trust partitioning to give you intelligent consensus features such as: autonomous splitting, diagnostics and recombination.
* Modal consensus modes supported as first class citizen (Proof-of-Consensus, and toggleable)
* Built-in remote development support inspired by distributed test-nets. Enjoy the benefits of a "local" experience, and seamlessly gain the full power of a remote cluster.
* Built-in analytics terminal, so you can execute trust-related commands in your workspace without leaving Léonne.

![Landscape](https://github.com/btq-ag/Leonne/blob/main/Plots/network_complex_with_landscape_variation1.gif)
![Community](https://github.com/btq-ag/Leonne/blob/main/Plots/topological_partitioning_complex_w5.gif)
![Blockchain](https://github.com/btq-ag/Leonne/blob/main/Plots/blockchain_network_visualization.gif)
![Network Evolution](https://github.com/btq-ag/Leonne/blob/main/Plots/network_evolution_3d_simplices_v5.gif)

## Architecture

| Module | Description |
|--------|-------------|
| [Classical Algorithms](Classical%20Algorithms/) | Trust partitioning, consensus iterations, blockchain simulation, community optimization, generalized permutations, and network graph generation |
| [Quantum Algorithms](Quantum%20Algorithms/) | Quantum-enhanced counterparts with QRNG, QKD, entanglement simulation, and post-quantum key generation |
| [Documentation](Documentation/) | Mathematical foundations: simplicial complexes, persistent homology, cobordism theory, and TQFT extensions |
| [Extension](Extension/) | Continuous-time cobordism formulations and persistent homology of network histories |
| [Visualizer](Visualizer/) | Unified hub interfaces and interactive CLIs for running and visualizing all algorithms |

## Installation

```bash
pip install -e .
```

For topological analysis features (persistent homology, Betti numbers):

```bash
pip install -e ".[topology]"
```

For all optional dependencies:

```bash
pip install -e ".[all]"
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Léonne is released under the [MIT License](LICENSE). Copyright (c) 2024-2026 BTQ Technologies.

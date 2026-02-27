"""
conftest.py

Shared fixtures and path configuration for the Leonne test suite.
"""

import sys
import os
import pytest
import numpy as np

# Add module directories to sys.path so tests can import from them
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSICAL_DIRS = [
    os.path.join(ROOT, "Classical Algorithms", "Trust Partitioning"),
    os.path.join(ROOT, "Classical Algorithms", "Network Graph Generator"),
    os.path.join(ROOT, "Classical Algorithms", "Generalized Permutations"),
    os.path.join(ROOT, "Classical Algorithms", "Consensus Iterations"),
    os.path.join(ROOT, "Classical Algorithms", "Blockchain Simulation"),
    os.path.join(ROOT, "Classical Algorithms", "Community Optimization"),
]

QUANTUM_DIRS = [
    os.path.join(ROOT, "Quantum Algorithms", "Trust Partitioning"),
    os.path.join(ROOT, "Quantum Algorithms", "Consensus Iterations"),
    os.path.join(ROOT, "Quantum Algorithms", "Blockchain Simulation"),
    os.path.join(ROOT, "Quantum Algorithms", "Community Optimization"),
    os.path.join(ROOT, "Quantum Algorithms", "Generalized Permutations"),
    os.path.join(ROOT, "Quantum Algorithms", "Network Graph Generator"),
]

EXTENSION_DIRS = [
    os.path.join(ROOT, "Extension"),
    os.path.join(ROOT, "Extension", "Excess"),
]

for d in CLASSICAL_DIRS + QUANTUM_DIRS + EXTENSION_DIRS:
    if d not in sys.path and os.path.isdir(d):
        sys.path.insert(0, d)


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def twoNetworkSetup():
    """Deterministic two-network trust configuration from trustPartitioner.py."""
    networks = [[0, 1, 2], [3, 4]]

    trustNinN = np.array([
        [0, 1 / 2, 1 / 4],
        [1 / 3, 0, 1 / 4],
        [1 / 2, 1, 0],
    ])
    trustNinM = np.array([
        [1, 1 / 5],
        [1 / 3, 1 / 2],
        [1 / 3, 1],
    ])
    trustMinN = np.array([
        [1 / 7, 1 / 2, 1 / 3],
        [1, 1 / 4, 1 / 2],
    ])
    trustMinM = np.array([
        [0, 1 / 4],
        [1 / 3, 0],
    ])

    systemTrust = np.hstack((
        np.vstack((trustNinN, trustMinN)),
        np.vstack((trustNinM, trustMinM)),
    ))

    nodeSecurity = {0: 1 / 6, 1: 1 / 2, 2: 1 / 3, 3: 1 / 2, 4: 1 / 4}

    networkSecurity = {}
    for idx, net in enumerate(networks):
        vals = sorted([nodeSecurity[n] for n in net])
        networkSecurity[idx] = np.max(vals[len(vals) // 2:])

    return networks, systemTrust, nodeSecurity, networkSecurity


@pytest.fixture
def threeNetworkSetup():
    """Three-network setup with random trust (seeded)."""
    np.random.seed(99)
    networks = [[0, 1, 2], [3, 4], [5]]
    systemTrust = np.random.rand(6, 6)
    nodeSecurity = {0: 1 / 6, 1: 1 / 2, 2: 1 / 3, 3: 1 / 2, 4: 1 / 4, 5: 1 / 7}

    networkSecurity = {}
    for idx, net in enumerate(networks):
        vals = sorted([nodeSecurity[n] for n in net])
        networkSecurity[idx] = np.max(vals[len(vals) // 2:])

    return networks, systemTrust, nodeSecurity, networkSecurity

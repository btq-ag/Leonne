"""
test_quantumTrust.py

Tests for quantum-enhanced trust functions: QRNG simulation,
trust matrix construction, and node distribution.
"""

import numpy as np
import random
import pytest

from quantum_trust_partitioner import (
    simulate_quantum_random_bit,
    generate_quantum_random_number,
    create_initial_quantum_trust_matrix,
    distribute_nodes_to_networks,
)


# ---------------------------------------------------------------------------
# simulate_quantum_random_bit()
# ---------------------------------------------------------------------------

class TestQuantumRandomBit:
    """Tests for simulated quantum random bit generation."""

    def test_returnsBinaryValue(self):
        """Output is always 0 or 1."""
        for _ in range(100):
            bit = simulate_quantum_random_bit()
            assert bit in (0, 1)

    def test_bothValuesAppear(self):
        """Over many samples, both 0 and 1 appear."""
        random.seed(42)
        bits = [simulate_quantum_random_bit() for _ in range(200)]
        assert 0 in bits
        assert 1 in bits


# ---------------------------------------------------------------------------
# generate_quantum_random_number()
# ---------------------------------------------------------------------------

class TestQuantumRandomNumber:
    """Tests for quantum-inspired random number generation."""

    def test_defaultEightBitRange(self):
        """8-bit number is in [0, 255]."""
        random.seed(42)
        for _ in range(50):
            n = generate_quantum_random_number(bits=8)
            assert 0 <= n <= 255

    def test_singleBitRange(self):
        """1-bit number is 0 or 1."""
        random.seed(42)
        for _ in range(50):
            n = generate_quantum_random_number(bits=1)
            assert n in (0, 1)

    def test_sixteenBitRange(self):
        """16-bit number is in [0, 65535]."""
        random.seed(42)
        for _ in range(50):
            n = generate_quantum_random_number(bits=16)
            assert 0 <= n <= 65535

    def test_returnsInteger(self):
        """Return type is int."""
        result = generate_quantum_random_number(bits=4)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# create_initial_quantum_trust_matrix()
# ---------------------------------------------------------------------------

class TestQuantumTrustMatrix:
    """Tests for quantum trust matrix construction."""

    def test_matrixShape(self):
        """Trust matrix is n x n."""
        np.random.seed(42)
        random.seed(42)
        mat = create_initial_quantum_trust_matrix(10, 'random')
        assert mat.shape == (10, 10)

    def test_selfTrustIsOne(self):
        """Diagonal entries (self-trust) are 1.0."""
        np.random.seed(42)
        random.seed(42)
        mat = create_initial_quantum_trust_matrix(8, 'random')
        np.testing.assert_array_equal(np.diag(mat), np.ones(8))

    def test_valuesInUnitInterval(self):
        """All trust values are in [0, 1]."""
        np.random.seed(42)
        random.seed(42)
        mat = create_initial_quantum_trust_matrix(10, 'small_world')
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)

    def test_supportedNetworkTypes(self):
        """All four network types produce valid matrices."""
        for netType in ('random', 'small_world', 'scale_free', 'community'):
            np.random.seed(42)
            random.seed(42)
            mat = create_initial_quantum_trust_matrix(12, netType)
            assert mat.shape == (12, 12)

    def test_unknownTypeRaises(self):
        """Invalid network type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown network type"):
            create_initial_quantum_trust_matrix(5, 'nonexistent')


# ---------------------------------------------------------------------------
# distribute_nodes_to_networks()
# ---------------------------------------------------------------------------

class TestDistributeNodes:
    """Tests for node-to-network distribution."""

    def test_balancedDistributionCoversAllNodes(self):
        """Balanced mode assigns every node exactly once."""
        random.seed(42)
        nets = distribute_nodes_to_networks(12, 3, 'balanced')
        allNodes = sorted([n for sub in nets for n in sub])
        assert allNodes == list(range(12))

    def test_balancedIsRoughlyEven(self):
        """Balanced mode produces networks of similar size."""
        random.seed(42)
        nets = distribute_nodes_to_networks(12, 3, 'balanced')
        sizes = [len(n) for n in nets]
        assert max(sizes) - min(sizes) <= 1

    def test_skewedDistributionCoversAllNodes(self):
        """Skewed mode still assigns every node."""
        random.seed(42)
        nets = distribute_nodes_to_networks(10, 3, 'skewed')
        allNodes = sorted([n for sub in nets for n in sub])
        assert allNodes == list(range(10))

    def test_skewedFirstNetworkIsLargest(self):
        """Skewed mode makes the first network the largest."""
        random.seed(42)
        nets = distribute_nodes_to_networks(20, 4, 'skewed')
        assert len(nets[0]) >= len(nets[1])

    def test_randomDistributionCoversAllNodes(self):
        """Random mode assigns every node."""
        random.seed(42)
        nets = distribute_nodes_to_networks(15, 4, 'random')
        allNodes = sorted([n for sub in nets for n in sub])
        assert allNodes == list(range(15))

    def test_numberOfNetworksMatchesRequest(self):
        """Output list length equals requested number of networks."""
        nets = distribute_nodes_to_networks(10, 5, 'balanced')
        assert len(nets) == 5

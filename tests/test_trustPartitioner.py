"""
test_trustPartitioner.py

Tests for the classical trust-based network partitioning algorithm.
Covers trust functions r(), rr(), networkCombiner(), and networkPartitioner().
"""

import numpy as np
import pytest

from trustPartitioner import r, rr, networkCombiner, networkPartitioner


# ---------------------------------------------------------------------------
# r() - node-level trust
# ---------------------------------------------------------------------------

class TestNodeTrust:
    """Tests for the r() function computing node-to-network trust."""

    def test_forwardTrustShape(self):
        """Forward trust returns one value per row."""
        mat = np.array([[0, 0.5], [0.3, 0]])
        result = r(mat, forward=True)
        assert result.shape == (2,)

    def test_reverseTrustShape(self):
        """Reverse trust returns one value per column."""
        mat = np.array([[0, 0.5, 0.1], [0.3, 0, 0.4]])
        result = r(mat, forward=False)
        assert result.shape == (3,)

    def test_forwardTrustValues(self):
        """Hand-computed forward trust for a 3x2 matrix."""
        mat = np.array([
            [1.0, 0.2],
            [0.3, 0.5],
            [0.3, 1.0],
        ])
        result = r(mat, forward=True)
        # r_i(B) = (1/|A|) * sum of row i
        expected = np.array([
            (1 / 3) * (1.0 + 0.2),
            (1 / 3) * (0.3 + 0.5),
            (1 / 3) * (0.3 + 1.0),
        ])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_reverseTrustValues(self):
        """Hand-computed reverse trust for a 2x3 matrix."""
        mat = np.array([
            [0.1, 0.5, 0.3],
            [1.0, 0.25, 0.5],
        ])
        result = r(mat, forward=False)
        # r_A(j) = (1/|B|) * sum of column j (after transpose)
        expected = np.array([
            (1 / 3) * (0.1 + 1.0),
            (1 / 3) * (0.5 + 0.25),
            (1 / 3) * (0.3 + 0.5),
        ])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_zeroMatrixReturnsZeros(self):
        """Zero trust matrix yields zero trust values."""
        mat = np.zeros((4, 3))
        result = r(mat, forward=True)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_identityLikeTrust(self):
        """Square identity-like matrix: each row sums to 1."""
        mat = np.eye(3)
        result = r(mat, forward=True)
        expected = np.full(3, 1 / 3)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# rr() - network-level trust
# ---------------------------------------------------------------------------

class TestNetworkTrust:
    """Tests for the rr() function computing network-to-network trust."""

    def test_forwardNetworkTrust(self):
        """r(A,B) is the max of node-level forward trusts."""
        trustAB = np.array([[0.5, 0.1], [0.2, 0.3], [0.8, 0.4]])
        trustBA = np.array([[0.3, 0.6, 0.1], [0.7, 0.2, 0.5]])
        result = rr(trustAB, trustBA, forward=True)
        nodeTrusts = r(trustAB, forward=True)
        assert result == np.max(nodeTrusts)

    def test_reverseNetworkTrust(self):
        """r(B,A) is the max of node-level forward trusts from B's perspective."""
        trustAB = np.array([[0.5, 0.1], [0.2, 0.3]])
        trustBA = np.array([[0.3, 0.9], [0.7, 0.2]])
        result = rr(trustAB, trustBA, forward=False)
        nodeTrusts = r(trustBA, forward=True)
        assert result == np.max(nodeTrusts)

    def test_symmetricTrust(self):
        """For symmetric trust, forward and reverse should be equal."""
        mat = np.array([[0.5, 0.3], [0.3, 0.5]])
        fwd = rr(mat, mat, forward=True)
        rev = rr(mat, mat, forward=False)
        np.testing.assert_allclose(fwd, rev, atol=1e-12)


# ---------------------------------------------------------------------------
# networkCombiner()
# ---------------------------------------------------------------------------

class TestNetworkCombiner:
    """Tests for networkCombiner() merging networks and empty sets."""

    def test_basicCombination(self):
        """Combines two networks with one isolated node."""
        nets = [[2, 0, 1], [4, 3]]
        emptySets = {0: [], 1: [], 2: [], 3: [], 4: [5]}
        result = networkCombiner(nets, emptySets)
        # Each sub-list should be sorted
        for subNet in result:
            assert subNet == sorted(subNet)
        # All original nodes + isolated node should appear
        allNodes = [n for sub in result for n in sub]
        assert sorted(allNodes) == [0, 1, 2, 3, 4, 5]

    def test_noEmptySets(self):
        """When no nodes abandon, output equals sorted input networks."""
        nets = [[3, 1], [0, 2]]
        emptySets = {0: [], 1: [], 2: [], 3: []}
        result = networkCombiner(nets, emptySets)
        assert result == [[1, 3], [0, 2]]

    def test_allNodesAbandoned(self):
        """When all networks are empty, each node becomes its own network."""
        nets = [[], []]
        emptySets = {0: [0], 1: [1], 2: [2]}
        result = networkCombiner(nets, emptySets)
        allNodes = sorted([n for sub in result for n in sub])
        assert allNodes == [0, 1, 2]

    def test_emptyInputs(self):
        """Empty networks and empty sets returns empty list."""
        result = networkCombiner([], {})
        assert result == []


# ---------------------------------------------------------------------------
# networkPartitioner()
# ---------------------------------------------------------------------------

class TestNetworkPartitioner:
    """Tests for the full trust-based network partitioning algorithm."""

    def test_returnsCorrectStructure(self, twoNetworkSetup):
        """Partitioner returns (networks, trust, nodeSec, netSec) tuple."""
        nets, trust, nodeSec, netSec = twoNetworkSetup
        result = networkPartitioner(nets, trust, nodeSec, netSec)
        assert len(result) == 4
        newNets, retTrust, retNodeSec, retNetSec = result
        assert isinstance(newNets, list)
        np.testing.assert_array_equal(retTrust, trust)

    def test_allNodesPreserved(self, twoNetworkSetup):
        """Total node count is preserved after partitioning."""
        nets, trust, nodeSec, netSec = twoNetworkSetup
        totalBefore = sum(len(n) for n in nets)
        newNets, *_ = networkPartitioner(nets, trust, nodeSec, netSec)
        totalAfter = sum(len(n) for n in newNets)
        assert totalAfter == totalBefore

    def test_nodesUniqueAfterPartitioning(self, twoNetworkSetup):
        """Every node appears exactly once after partitioning."""
        nets, trust, nodeSec, netSec = twoNetworkSetup
        newNets, *_ = networkPartitioner(nets, trust, nodeSec, netSec)
        allNodes = [n for sub in newNets for n in sub]
        assert len(allNodes) == len(set(allNodes))

    def test_threeNetworkPartitioning(self, threeNetworkSetup):
        """Partitioner works with three networks."""
        nets, trust, nodeSec, netSec = threeNetworkSetup
        newNets, *_ = networkPartitioner(nets, trust, nodeSec, netSec)
        allNodes = sorted([n for sub in newNets for n in sub])
        assert allNodes == [0, 1, 2, 3, 4, 5]

    def test_networksAreSorted(self, twoNetworkSetup):
        """Each sub-network in the output is sorted."""
        nets, trust, nodeSec, netSec = twoNetworkSetup
        newNets, *_ = networkPartitioner(nets, trust, nodeSec, netSec)
        for net in newNets:
            assert net == sorted(net)

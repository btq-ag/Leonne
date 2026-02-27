"""
test_graphGenerator.py

Tests for graph generation, edge set operations, and sequence verification.
Covers sequenceVerifier(), edgeAssignment(), removeDegeneracy(), and consensusShuffle().
"""

import numpy as np
import pytest

from graphGenerator import (
    sequenceVerifier,
    edgeAssignment,
    removeDegeneracy,
    consensusShuffle,
    testEdgeSet,
    largeEdgeSet,
    trivialEdgeSet,
)


# ---------------------------------------------------------------------------
# sequenceVerifier()
# ---------------------------------------------------------------------------

class TestSequenceVerifier:
    """Tests for the Gale-Ryser bipartite degree sequence verifier."""

    def test_trivialSequenceIsValid(self):
        """[1,1,1,1,1,1] with U=3, V=3 satisfies both constraints."""
        assert sequenceVerifier([1, 1, 1, 1, 1, 1], 3, 3, extraInfo=False) is True

    def test_nonTrivialSequenceIsValid(self):
        """[2,2,2,2,3,1] with U=3, V=3 is satisfiable."""
        assert sequenceVerifier([2, 2, 2, 2, 3, 1], 3, 3, extraInfo=False) is True

    def test_unsatisfiableSequenceReturnsFalse(self):
        """[6,2,1,3,3,3] with U=3, V=3 violates boundedness."""
        assert sequenceVerifier([6, 2, 1, 3, 3, 3], 3, 3, extraInfo=False) is False

    def test_unequalSumsReturnFalse(self):
        """U-sum != V-sum should fail the conservation constraint."""
        assert sequenceVerifier([3, 3, 3, 1, 1, 1], 3, 3, extraInfo=False) is False

    def test_singleNodeSequence(self):
        """Minimal case: one node on each side, degree 1."""
        assert sequenceVerifier([1, 1], 1, 1, extraInfo=False) is True

    def test_zeroSequence(self):
        """All-zero degree sequence is trivially satisfiable."""
        assert sequenceVerifier([0, 0, 0, 0], 2, 2, extraInfo=False) is True


# ---------------------------------------------------------------------------
# edgeAssignment()
# ---------------------------------------------------------------------------

class TestEdgeAssignment:
    """Tests for greedy edge assignment from valid degree sequences."""

    def test_trivialAssignment(self):
        """Trivial sequence produces correct number of edges."""
        edges = edgeAssignment([1, 1, 1, 1, 1, 1], 3, 3, extraInfo=False)
        assert edges is not False
        assert len(edges) == 3  # sum of U-degrees = 3

    def test_nonTrivialAssignment(self):
        """Non-trivial sequence produces edges matching total degree."""
        edges = edgeAssignment([2, 2, 2, 2, 3, 1], 3, 3, extraInfo=False)
        assert edges is not False
        assert len(edges) == 6  # sum of U-degrees = 2+2+2 = 6

    def test_invalidSequenceReturnsFalse(self):
        """Unsatisfiable sequence returns False."""
        result = edgeAssignment([6, 2, 1, 3, 3, 3], 3, 3, extraInfo=False)
        assert result is False

    def test_edgesAreValidPairs(self):
        """Each edge is a [u, v] pair with indices in range."""
        edges = edgeAssignment([1, 1, 1, 1, 1, 1], 3, 3, extraInfo=False)
        for edge in edges:
            assert len(edge) == 2
            assert 0 <= edge[0] < 3
            assert 0 <= edge[1] < 3


# ---------------------------------------------------------------------------
# removeDegeneracy()
# ---------------------------------------------------------------------------

class TestRemoveDegeneracy:
    """Tests for removing duplicate edge sets."""

    def test_removesDuplicates(self):
        """Identical edge sets (regardless of order) are deduplicated."""
        edgeSets = [
            [[1, 2], [3, 4]],
            [[3, 4], [1, 2]],  # same as first, different order
            [[5, 6]],
        ]
        result = removeDegeneracy(edgeSets)
        assert len(result) == 2

    def test_noDuplicates(self):
        """All unique edge sets are preserved."""
        edgeSets = [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ]
        result = removeDegeneracy(edgeSets)
        assert len(result) == 3

    def test_emptyInput(self):
        """Empty input returns empty output."""
        assert removeDegeneracy([]) == []

    def test_singleEdgeSet(self):
        """Single edge set is returned as-is."""
        edgeSets = [[[1, 2], [3, 4]]]
        result = removeDegeneracy(edgeSets)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# consensusShuffle()
# ---------------------------------------------------------------------------

class TestConsensusShuffle:
    """Tests for the consensus-preserving edge shuffle."""

    def test_outputShapeMatchesInput(self):
        """Shuffled edge set has the same shape as input."""
        np.random.seed(42)
        result = consensusShuffle(testEdgeSet, extraInfo=False)
        assert result.shape == (len(testEdgeSet), 2)

    def test_preservesVColumn(self):
        """The V column (second column) is unchanged by the shuffle."""
        np.random.seed(42)
        original = [[1, 2], [2, 1], [2, 3], [3, 2], [4, 4]]
        result = consensusShuffle(original, extraInfo=False)
        originalV = sorted([e[1] for e in original])
        resultV = sorted(result[:, 1].tolist())
        assert originalV == resultV

    def test_preservesUElements(self):
        """The U column contains the same elements (possibly reordered)."""
        np.random.seed(42)
        original = [[1, 2], [2, 1], [2, 3], [3, 2], [4, 4]]
        result = consensusShuffle(original, extraInfo=False)
        originalU = sorted([e[0] for e in original])
        resultU = sorted(result[:, 0].tolist())
        assert originalU == resultU

    def test_trivialEdgeSetIsStable(self):
        """Trivial set [[1,1],[2,2],[3,3]] has identity as only valid shuffle."""
        np.random.seed(42)
        result = consensusShuffle(trivialEdgeSet, extraInfo=False)
        assert result.shape == (3, 2)

    def test_deterministicWithSeed(self):
        """Same seed produces same result."""
        np.random.seed(123)
        r1 = consensusShuffle(largeEdgeSet.copy(), extraInfo=False)
        np.random.seed(123)
        r2 = consensusShuffle(largeEdgeSet.copy(), extraInfo=False)
        np.testing.assert_array_equal(r1, r2)

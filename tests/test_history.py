"""
test_history.py

Tests for the OperationHistory class from the Extension module.
"""

import pytest

# Import from Extension/Excess where the actual implementation lives
from history import OperationHistory


class TestOperationHistory:
    """Tests for the OperationHistory class."""

    def test_initializesEmpty(self):
        """New history starts with no operations."""
        hist = OperationHistory()
        assert hist.operations == []

    def test_recordSingleOperation(self):
        """Recording an operation adds it to the list."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {"nodes": 10}, "success")
        assert len(hist.operations) == 1
        assert hist.operations[0]["type"] == "partition"

    def test_recordMultipleOperations(self):
        """Multiple operations are stored in order."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {}, "r1")
        hist.record_operation("merge", 2.0, {}, "r2")
        hist.record_operation("split", 3.0, {}, "r3")
        assert len(hist.operations) == 3
        assert [op["type"] for op in hist.operations] == ["partition", "merge", "split"]

    def test_getHistoryReturnsAll(self):
        """get_history() with no filters returns everything."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {}, "r1")
        hist.record_operation("merge", 2.0, {}, "r2")
        result = hist.get_history()
        assert len(result) == 2

    def test_filterByType(self):
        """Filtering by operation_type returns only matching entries."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {}, "r1")
        hist.record_operation("merge", 2.0, {}, "r2")
        hist.record_operation("partition", 3.0, {}, "r3")
        result = hist.get_history(operation_type="partition")
        assert len(result) == 2
        assert all(op["type"] == "partition" for op in result)

    def test_filterByStartTime(self):
        """start_time filter excludes earlier operations."""
        hist = OperationHistory()
        hist.record_operation("a", 1.0, {}, None)
        hist.record_operation("b", 5.0, {}, None)
        hist.record_operation("c", 10.0, {}, None)
        result = hist.get_history(start_time=5.0)
        assert len(result) == 2
        assert result[0]["type"] == "b"

    def test_filterByEndTime(self):
        """end_time filter excludes later operations."""
        hist = OperationHistory()
        hist.record_operation("a", 1.0, {}, None)
        hist.record_operation("b", 5.0, {}, None)
        hist.record_operation("c", 10.0, {}, None)
        result = hist.get_history(end_time=5.0)
        assert len(result) == 2
        assert result[-1]["type"] == "b"

    def test_filterByTimeRange(self):
        """Combined start_time and end_time narrows the window."""
        hist = OperationHistory()
        for i in range(10):
            hist.record_operation("op", float(i), {}, i)
        result = hist.get_history(start_time=3.0, end_time=6.0)
        timestamps = [op["timestamp"] for op in result]
        assert all(3.0 <= t <= 6.0 for t in timestamps)

    def test_filterByTypeAndTime(self):
        """All filters compose correctly."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {}, "r1")
        hist.record_operation("merge", 2.0, {}, "r2")
        hist.record_operation("partition", 3.0, {}, "r3")
        hist.record_operation("partition", 5.0, {}, "r4")
        result = hist.get_history(operation_type="partition", start_time=2.0, end_time=4.0)
        assert len(result) == 1
        assert result[0]["result"] == "r3"

    def test_noMatchReturnsEmpty(self):
        """Filters that match nothing return an empty list."""
        hist = OperationHistory()
        hist.record_operation("partition", 1.0, {}, "r1")
        result = hist.get_history(operation_type="nonexistent")
        assert result == []

    def test_parametersStored(self):
        """Operation parameters are correctly stored and retrievable."""
        hist = OperationHistory()
        params = {"nodes": 10, "threshold": 0.5, "networks": [[0, 1], [2, 3]]}
        hist.record_operation("partition", 1.0, params, "ok")
        assert hist.operations[0]["parameters"] == params

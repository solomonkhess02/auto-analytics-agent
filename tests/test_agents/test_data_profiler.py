"""
Unit tests for DataProfilerAgent.

Code execution and the summary LLM call are mocked so these tests are
deterministic and require no network.
"""

import json
from unittest.mock import MagicMock

from agents.data_profiler import DataProfilerAgent
from core.state import PipelineState


def _profiler_returning(stats: dict):
    """Build a profiler whose code execution yields the given stats JSON."""
    agent = DataProfilerAgent()
    agent._generate_and_execute_code = MagicMock(return_value=("<code>", json.dumps(stats)))
    summary = MagicMock()
    summary.content = "A concise three-sentence summary."
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = summary
    return agent


def test_profiler_fills_gaps_and_resolves_classification_target():
    stats = {
        "shape": [5, 3],
        "columns": ["age", "city", "target"],
        "dtypes": {"age": "int64", "city": "object", "target": "int64"},
        "missing_values": {"age": 0, "city": 0, "target": 0},
        "missing_percentages": {"age": 0.0, "city": 0.0, "target": 0.0},
        "unique_counts": {"age": 5, "city": 3, "target": 2},
        "descriptive_stats": {"age": {"mean": 35.0}},
        "sample_rows": [{"age": 25, "city": "NY", "target": 0}],
        "correlations": {"age": {"age": 1.0}},
    }
    agent = _profiler_returning(stats)
    state = PipelineState(dataset_path="data/x.csv", task_type="auto",
                          messages=[], errors=[], warnings=[])

    result = agent.run(state)

    assert "data_profile" in result
    profile = result["data_profile"]

    # Previously-empty fields are now populated from the profiling output.
    assert profile["descriptive_stats"] == {"age": {"mean": 35.0}}
    assert profile["sample_rows"] == [{"age": 25, "city": "NY", "target": 0}]
    assert profile["correlations"] == {"age": {"age": 1.0}}

    # Target/task type resolved up front and propagated to top-level state.
    assert result["target_column"] == "target"
    assert result["task_type"] == "classification"
    assert profile["target_column"] == "target"
    assert profile["task_type"] == "classification"


def test_profiler_infers_regression_for_continuous_target():
    stats = {
        "shape": [100, 2],
        "columns": ["feature", "price"],
        "dtypes": {"feature": "float64", "price": "float64"},
        "missing_values": {},
        "missing_percentages": {},
        "unique_counts": {"feature": 100, "price": 95},
        "descriptive_stats": {},
        "sample_rows": [],
        "correlations": {},
    }
    agent = _profiler_returning(stats)
    state = PipelineState(dataset_path="data/x.csv", task_type="auto",
                          messages=[], errors=[], warnings=[])

    result = agent.run(state)

    assert result["target_column"] == "price"
    assert result["task_type"] == "regression"


def test_profiler_respects_user_supplied_target():
    stats = {
        "shape": [10, 3],
        "columns": ["a", "b", "c"],
        "dtypes": {"a": "int64", "b": "int64", "c": "int64"},
        "missing_values": {},
        "missing_percentages": {},
        "unique_counts": {"a": 2, "b": 10, "c": 9},
        "descriptive_stats": {},
        "sample_rows": [],
        "correlations": {},
    }
    agent = _profiler_returning(stats)
    state = PipelineState(dataset_path="data/x.csv", target_column="a", task_type="auto",
                          messages=[], errors=[], warnings=[])

    result = agent.run(state)

    # Honors the user's choice instead of defaulting to the last column.
    assert result["target_column"] == "a"
    assert result["task_type"] == "classification"  # 2 unique values

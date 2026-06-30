"""
Unit tests for FeatureEngineerAgent.select_features().

The LLM is mocked so these tests are deterministic and require no network.
"""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.state import PipelineState
from agents.feature_engineer import FeatureEngineerAgent


@pytest.fixture
def engineered_csv(tmp_path):
    """A small engineered dataset with one clearly useless (zero-variance) column."""
    path = tmp_path / "engineered.csv"
    df = pd.DataFrame({
        "useful_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "useful_b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "noise": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "target": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    })
    df.to_csv(path, index=False)
    return str(path)


def _agent_with_drop_decision(drop_list):
    """Build an agent whose mocked LLM returns a fixed feature-selection decision."""
    agent = FeatureEngineerAgent()
    response = MagicMock()
    response.content = json.dumps({"features_to_drop": drop_list, "reasoning": "test reasoning"})
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = response
    return agent


def test_select_features_drops_requested_column(engineered_csv):
    agent = _agent_with_drop_decision(["noise"])
    state = PipelineState(
        engineered_dataset_path=engineered_csv,
        target_column="target",
        task_type="classification",
        feature_report={"engineer_summary": "Base summary."},
        messages=[], errors=[], warnings=[],
    )

    result = agent.select_features(state)

    assert "feature_report" in result
    report = result["feature_report"]
    assert "noise" in report["features_dropped"]
    assert "target" not in report["final_feature_list"]
    assert {"useful_a", "useful_b", "noise"} <= set(report["feature_importances"].keys())
    # importances are JSON-serializable floats
    assert all(isinstance(v, float) for v in report["feature_importances"].values())

    df = pd.read_csv(engineered_csv)
    assert "noise" not in df.columns
    assert "target" in df.columns


def test_select_features_never_drops_target(engineered_csv):
    agent = _agent_with_drop_decision(["target", "useful_a"])
    state = PipelineState(
        engineered_dataset_path=engineered_csv,
        target_column="target",
        task_type="classification",
        messages=[], errors=[], warnings=[],
    )

    agent.select_features(state)

    df = pd.read_csv(engineered_csv)
    assert "target" in df.columns  # target must survive even if the LLM asks to drop it


def test_select_features_keeps_all_when_drop_too_aggressive(engineered_csv):
    # Asking to drop both real features would leave only target + noise -> rejected.
    agent = _agent_with_drop_decision(["useful_a", "useful_b", "noise"])
    state = PipelineState(
        engineered_dataset_path=engineered_csv,
        target_column="target",
        task_type="classification",
        messages=[], errors=[], warnings=[],
    )

    result = agent.select_features(state)

    assert result["feature_report"]["features_dropped"] == []
    df = pd.read_csv(engineered_csv)
    assert set(df.columns) == {"useful_a", "useful_b", "noise", "target"}


def test_select_features_missing_path_warns():
    agent = FeatureEngineerAgent()
    agent.llm = MagicMock()
    state = PipelineState(messages=[], errors=[], warnings=[])

    result = agent.select_features(state)

    assert "warnings" in result
    assert "feature_report" not in result
    agent.llm.invoke.assert_not_called()

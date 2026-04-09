import os
import pytest
import pandas as pd
from core.state import PipelineState
from agents.data_cleaner import DataCleanerAgent

@pytest.fixture
def dirty_dataset_path(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
        "age": [25, None, 35, 45],
        "salary": [50k, 60k, None, 80k]
    })
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    return str(path)

# This is a placeholder test. Full integration testing requires LLM mocking or actual LLM access.
def test_cleaner_agent_instantiation():
    agent = DataCleanerAgent()
    assert agent.name == "Cleaner"

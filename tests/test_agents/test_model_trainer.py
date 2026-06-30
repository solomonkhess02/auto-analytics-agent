import pytest
import os
import pandas as pd
from core.state import PipelineState
from agents.model_trainer import ModelTrainerAgent


@pytest.fixture
def mock_engineered_state():
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "city_London": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "target": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
    })
    df.to_csv("data/mock_engineered.csv", index=False)

    return PipelineState(
        dataset_path="data/mock_engineered.csv",
        cleaned_dataset_path="data/mock_engineered.csv",
        engineered_dataset_path="data/mock_engineered.csv",
        target_column="target",
        task_type="classification",
        messages=[],
        errors=[],
        warnings=[]
    )


@pytest.mark.integration
def test_model_trainer_run(mock_engineered_state):
    agent = ModelTrainerAgent()
    result = agent.run(mock_engineered_state)

    assert "errors" not in result
    assert "training_results" in result
    assert "model_artifacts_dir" in result
    assert result["target_column"] == "target"
    assert result["task_type"] == "classification"

    results = result["training_results"]
    assert len(results) >= 3
    for r in results:
        assert "model_name" in r
        assert "model_type" in r
        assert "hyperparameters" in r
        assert "training_time_seconds" in r
        assert "cross_val_scores" in r
        assert "cross_val_mean" in r

    # Check model artifacts exist
    artifacts_dir = result["model_artifacts_dir"]
    assert os.path.exists(artifacts_dir)
    assert os.path.exists(os.path.join(artifacts_dir, "train.csv"))
    assert os.path.exists(os.path.join(artifacts_dir, "test.csv"))

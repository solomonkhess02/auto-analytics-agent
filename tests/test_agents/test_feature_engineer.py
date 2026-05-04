import pytest
from core.state import PipelineState
from agents.feature_engineer import FeatureEngineerAgent
import pandas as pd
import os

@pytest.fixture
def mock_state():
    # Create a dummy cleaned dataset
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "city": ["New York", "London", "Paris", "London", "New York"],
        "target": [0, 1, 0, 1, 1],
        "date_joined": ["2023-01-01", "2023-02-15", "2023-03-10", "2023-04-20", "2023-05-05"]
    })
    df.to_csv("data/mock_cleaned.csv", index=False)
    
    return PipelineState(
        dataset_path="data/mock_cleaned.csv",
        cleaned_dataset_path="data/mock_cleaned.csv",
        target_column="target",
        task_type="classification",
        data_profile={
            "shape": (5, 4),
            "dtypes": {"age": "int64", "city": "object", "target": "int64", "date_joined": "object"},
            "missing_percentages": {},
            "correlations": {},
            "profiler_summary": "A small mock dataset with categorical and datetime columns."
        },
        human_feedback="None",
        messages=[],
        errors=[],
        warnings=[]
    )

@pytest.mark.integration
def test_feature_engineer_generate_plan(mock_state):
    agent = FeatureEngineerAgent()
    result = agent.generate_feature_plan(mock_state)
    
    assert "errors" not in result
    assert "feature_plan" in result
    plan = result["feature_plan"]
    assert "datetime_features" in plan
    assert "categorical_encoding" in plan
    assert "numerical_scaling" in plan

@pytest.mark.integration
def test_feature_engineer_execute_plan(mock_state):
    agent = FeatureEngineerAgent()
    
    # 1. Generate plan
    plan_result = agent.generate_feature_plan(mock_state)
    mock_state.update(plan_result)
    
    # 2. Execute plan
    exec_result = agent.execute_plan(mock_state)
    
    assert "errors" not in exec_result
    assert "engineered_dataset_path" in exec_result
    assert "feature_report" in exec_result
    
    # Verify file was created
    assert os.path.exists(exec_result["engineered_dataset_path"])
    df = pd.read_csv(exec_result["engineered_dataset_path"])
    
    # Original shape was (5,4). With OneHot for city and datetime extraction, columns should increase.
    # The exact columns depend on LLM output, but target should be preserved
    assert "target" in df.columns

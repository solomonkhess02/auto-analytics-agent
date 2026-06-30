import pytest
import os
import pandas as pd
from core.state import PipelineState
from agents.model_trainer import ModelTrainerAgent
from agents.evaluator import EvaluatorAgent


@pytest.fixture
def mock_trained_state():
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "city_London": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "target": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
    })
    df.to_csv("data/mock_engineered.csv", index=False)

    state = PipelineState(
        dataset_path="data/mock_engineered.csv",
        cleaned_dataset_path="data/mock_engineered.csv",
        engineered_dataset_path="data/mock_engineered.csv",
        target_column="target",
        task_type="classification",
        messages=[],
        errors=[],
        warnings=[]
    )

    trainer = ModelTrainerAgent()
    trainer_results = trainer.run(state)
    state.update(trainer_results)
    return state


@pytest.mark.integration
def test_evaluator_run(mock_trained_state):
    agent = EvaluatorAgent()
    result = agent.run(mock_trained_state)

    assert "errors" not in result
    assert "evaluation_report" in result
    
    report = result["evaluation_report"]
    assert report["best_model_name"] in ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    assert "metrics" in report
    metrics = report["metrics"]
    assert "accuracy" in metrics
    
    assert report["report_html_path"] == os.path.join("reports", "evaluation_report.html")
    assert os.path.exists(report["report_html_path"])
    
    # Verify plots were generated
    assert len(report["plot_paths"]) > 0
    for plot_path in report["plot_paths"]:
        assert os.path.exists(plot_path)

    assert len(report["natural_language_report"]) > 0

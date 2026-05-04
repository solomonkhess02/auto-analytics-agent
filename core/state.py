from typing import TypedDict, Annotated, Optional
import operator

class DataProfile(TypedDict):
    """Output of the Data Profiler Agent."""
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    descriptive_stats: dict
    unique_counts: dict[str, int]
    sample_rows: list[dict]
    correlations: dict
    target_column: Optional[str]
    task_type: Optional[str]
    profiler_summary: str

class CleaningReport(TypedDict):
    """Output of the Data Cleaner Agent."""
    actions_taken: list[str]
    columns_dropped: list[str]
    columns_modified: dict[str, str]
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int
    cleaning_code: str
    cleaner_summary: str

class FeatureReport(TypedDict):
    """Output of the Feature Engineer Agent."""
    features_created: list[str]
    features_dropped: list[str]
    encoding_applied: dict[str, str]
    scaling_applied: dict[str, str]
    feature_importances: dict[str, float]
    final_feature_list: list[str]
    feature_code: str
    engineer_summary: str

class TrainingResult(TypedDict):
    """Result from training a single model."""
    model_name: str
    model_type: str
    hyperparameters: dict
    training_time_seconds: float
    cross_val_scores: list[float]
    cross_val_mean: float

class EvaluationReport(TypedDict):
    """Output of the Evaluator Agent."""
    best_model_name: str
    metrics: dict[str, float]
    confusion_matrix: Optional[list[list[int]]]
    classification_report: Optional[str]
    feature_importances: dict[str, float]
    plot_paths: list[str]
    natural_language_report: str
    report_html_path: str

class PipelineState(TypedDict):
    """Shared state dictionary for the LangGraph orchestrator."""
    dataset_path: str
    target_column: Optional[str]
    task_type: str
    user_instructions: Optional[str]

    data_profile: Optional[DataProfile]
    cleaning_plan: Optional[dict]
    feature_plan: Optional[dict]
    human_feedback: Optional[str]
    cleaning_report: Optional[CleaningReport]
    feature_report: Optional[FeatureReport]
    training_results: Optional[list[TrainingResult]]
    evaluation_report: Optional[EvaluationReport]

    current_phase: str
    messages: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]

    cleaned_dataset_path: Optional[str]
    engineered_dataset_path: Optional[str]
    model_artifacts_dir: Optional[str]

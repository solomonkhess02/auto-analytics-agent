"""
PipelineState — The shared memory for all agents in the pipeline.

WHAT THIS FILE DOES:
    Every agent reads from and writes to this state object. It's the single
    source of truth that flows through the entire LangGraph pipeline.

WHY TypedDict:
    LangGraph requires state to be a TypedDict so it knows the exact shape
    of data flowing between agent nodes. This also gives us IDE autocomplete
    and type checking.

HOW IT WORKS:
    1. User uploads a dataset → initial state is created with dataset_path
    2. Profiler agent reads dataset_path → writes data_profile
    3. Cleaner agent reads data_profile → writes cleaning_report
    4. Engineer agent reads cleaning_report → writes feature_report
    5. Trainer agent reads feature_report → writes training_results
    6. Evaluator reads everything → writes evaluation_report
"""

from typing import TypedDict, Annotated, Optional
import operator


# ============================================================================
# SUB-STATE: Data Profiler Output
# ============================================================================

class DataProfile(TypedDict):
    """Output of the Data Profiler Agent.

    Contains everything we learn about the raw dataset:
    its shape, columns, data types, missing values, correlations, etc.
    The Cleaner Agent reads this to decide its cleaning strategy.
    """
    shape: tuple[int, int]                    # (num_rows, num_columns)
    columns: list[str]                        # List of column names
    dtypes: dict[str, str]                    # Column name → data type
    missing_values: dict[str, int]            # Column name → count of missing values
    missing_percentages: dict[str, float]     # Column name → % missing (0-100)
    descriptive_stats: dict                   # Mean, std, min, max per column
    unique_counts: dict[str, int]             # Column name → number of unique values
    sample_rows: list[dict]                   # First 5 rows as list of dicts
    correlations: dict                        # Correlation matrix as nested dict
    target_column: Optional[str]              # Detected or user-specified target column
    task_type: Optional[str]                  # "classification" | "regression" | "auto"
    profiler_summary: str                     # LLM-generated natural language summary


# ============================================================================
# SUB-STATE: Data Cleaner Output
# ============================================================================

class CleaningReport(TypedDict):
    """Output of the Data Cleaner Agent.

    Documents every cleaning action taken: what was dropped, modified,
    imputed, and the before/after statistics.
    """
    actions_taken: list[str]                  # Human-readable list of actions
    columns_dropped: list[str]                # Columns removed from dataset
    columns_modified: dict[str, str]          # Column → description of change
    rows_before: int                          # Row count before cleaning
    rows_after: int                           # Row count after cleaning
    missing_before: int                       # Total missing values before
    missing_after: int                        # Total missing values after (should be 0)
    cleaning_code: str                        # The actual Python code that was executed
    cleaner_summary: str                      # LLM explanation of decisions


# ============================================================================
# SUB-STATE: Feature Engineer Output
# ============================================================================

class FeatureReport(TypedDict):
    """Output of the Feature Engineer Agent.

    Documents all feature transformations: new features created,
    encoding/scaling applied, and feature importance rankings.
    """
    features_created: list[str]               # New columns added
    features_dropped: list[str]               # Columns removed (low importance)
    encoding_applied: dict[str, str]          # Column → encoding type used
    scaling_applied: dict[str, str]           # Column → scaler type used
    feature_importances: dict[str, float]     # Column → importance score
    final_feature_list: list[str]             # Final columns used for training
    feature_code: str                         # The actual Python code that was executed
    engineer_summary: str                     # LLM explanation of decisions


# ============================================================================
# SUB-STATE: Model Training Output (one per model trained)
# ============================================================================

class TrainingResult(TypedDict):
    """Result from training a single model.

    The Trainer Agent trains multiple models and creates one of these
    for each. The Evaluator then compares them to pick the best.
    """
    model_name: str                           # Human-readable name, e.g. "Random Forest"
    model_type: str                           # Class name, e.g. "RandomForestClassifier"
    hyperparameters: dict                     # Best hyperparameters found
    training_time_seconds: float              # How long training took
    cross_val_scores: list[float]             # Score from each CV fold
    cross_val_mean: float                     # Average CV score


# ============================================================================
# SUB-STATE: Evaluation & Report Output
# ============================================================================

class EvaluationReport(TypedDict):
    """Output of the Evaluator Agent.

    Contains the final evaluation metrics, visualizations, and the
    LLM-generated natural language report.
    """
    best_model_name: str                      # Name of the winning model
    metrics: dict[str, float]                 # {"accuracy": 0.94, "f1": 0.92, ...}
    confusion_matrix: Optional[list[list[int]]]   # For classification tasks
    classification_report: Optional[str]           # sklearn classification report
    feature_importances: dict[str, float]     # Top features for best model
    plot_paths: list[str]                     # Paths to generated plot images
    natural_language_report: str              # Full LLM-written analysis report
    report_html_path: str                     # Path to the HTML report file


# ============================================================================
# MAIN STATE: The complete pipeline state
# ============================================================================

class PipelineState(TypedDict):
    """The complete shared state for the entire pipeline.

    This is the 'notebook' that all agents read from and write to.
    LangGraph passes this automatically between agent nodes.

    FIELD TYPES:
        - Regular fields: Each agent can overwrite the value
        - Annotated[list, operator.add]: Append-only — new items are
          ADDED to the list, never replacing existing items.
          This ensures we never lose messages/errors from earlier agents.
    """

    # --- Input (set when user uploads a dataset) ---
    dataset_path: str                                   # Path to uploaded CSV/Excel
    target_column: Optional[str]                        # User-specified or auto-detected
    task_type: str                                      # "classification", "regression", "auto"
    user_instructions: Optional[str]                    # Any special user requests

    # --- Agent Outputs (None until each agent runs) ---
    data_profile: Optional[DataProfile]                 # Written by Profiler
    cleaning_plan: Optional[dict]                       # Written by Cleaner (Phase 1)
    human_feedback: Optional[str]                       # Written by User (Human-in-the-loop)
    cleaning_report: Optional[CleaningReport]           # Written by Cleaner (Phase 2)
    feature_report: Optional[FeatureReport]             # Written by Engineer
    training_results: Optional[list[TrainingResult]]    # Written by Trainer
    evaluation_report: Optional[EvaluationReport]       # Written by Evaluator

    # --- Pipeline Metadata ---
    current_phase: str                                  # Current pipeline phase name
    messages: Annotated[list[dict], operator.add]       # Append-only activity log
    errors: Annotated[list[str], operator.add]          # Append-only error log
    warnings: Annotated[list[str], operator.add]        # Append-only warning log

    # --- Intermediate File Paths ---
    cleaned_dataset_path: Optional[str]                 # Written by Cleaner
    engineered_dataset_path: Optional[str]              # Written by Engineer
    model_artifacts_dir: Optional[str]                  # Written by Trainer

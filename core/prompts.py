"""
Prompts — Centralized brain for the agents.

WHAT THIS FILE DOES:
    Contains all the prompt templates used by our agents.
    Separating prompts from logic makes it easier to tune the AI's behavior.
"""

# ============================================================================
# General / Base Agent Prompts
# ============================================================================

NO_CODE_FOUND_PROMPT = """Failed to find Python code in your previous response. 
Please provide the code wrapped in ```python ... ``` blocks.

Original prompt:
{prompt}"""

RETRY_CODE_PROMPT = """The Python code you generated failed to execute. You must fix it.

## Failed Code (Attempt {attempt})
```python
{failed_code}
```

## Error Output
```text
{error}
```

Please analyze the error, identify the bug, and provide the corrected Python code.
Output ONLY the corrected code inside ```python ... ``` blocks, with no conversational filler."""

# ============================================================================
# Data Profiler Prompts
# ============================================================================

PROFILER_CODE_PROMPT = """You are a Data Scientist Agent. Analyze the dataset at '{dataset_path}'.

Write a Python script that:
1. Loads the dataset using pandas.
2. Calculates the following statistics:
   - shape (list of 2 integers: [rows, columns])
   - columns (list of strings)
   - dtypes (dictionary mapping column names to string representations of their types)
   - missing_values (dictionary mapping column names to missing value counts)
   - missing_percentages (dictionary mapping column names to percentage of missing values as floats)
   - unique_counts (dictionary mapping column names to number of unique values)
3. Outputs the final result EXACTLY as a valid JSON string using `json.dumps()`.
4. Ensure the output only contains the JSON string and absolutely nothing else. Print the JSON at the very end.

IMPORTANT: Use proper error handling when reading the file."""

PROFILER_SUMMARY_PROMPT = """You are a Data Science consultant. Review these dataset statistics:
Shape: {shape}
Missing Values: {missing_values}
Data Types: {dtypes}

Write a concise, professional 3-sentence summary of this dataset.
Mention its size, any data quality issues (like missing values), and the general composition of the columns."""


# ============================================================================
# Data Cleaner Prompts
# ============================================================================

CLEANER_PLAN_PROMPT = """You are a Data Cleaning Planner. Review the dataset profile below to identify issues.

Profile:
{profile_summary}

Create a structured JSON plan to clean this dataset.
The plan MUST be a dictionary with these string keys (at minimum):
- "drop_columns": list of columns to drop completely (e.g. ones with too many missing values or zero variance).
- "impute_missing": dictionary describing how to handle remaining missing values per column.
- "type_conversions": dictionary for fixing incorrect types.
- "reasoning": a short sentence explaining these decisions.

Output EXACTLY valid JSON, nothing else."""

CLEANER_CODE_PROMPT = """You are a Python Data Engineer. Write a script to clean a dataset based on the following plan.

Dataset path: {dataset_path}
Cleaned output path: {output_path}
Human Feedback/Instructions: {human_feedback}

Cleaning Plan:
{cleaning_plan}

Instructions:
1. Load dataset with pandas.
2. Apply the cleaning plan (and adjust carefully for any Human Feedback).
3. Save the final cleaned dataframe to the output path using `df.to_csv(..., index=False)`.
4. Print a JSON object describing before/after stats: {{"missing_before": x, "missing_after": y, "rows_before": a, "rows_after": b}}
5. Provide ONLY the Python code in ```python ... ``` blocks.
"""


# ============================================================================
# Feature Engineer Prompts
# ============================================================================

FEATURE_ENGINEER_PLAN_PROMPT = """You are a Senior ML Engineer planning feature engineering for a dataset.
Review the data profile and the cleaned data info below carefully.

Task Type: {task_type}
Target Column: {target_column}
Profile Summary:
{profile_summary}

Create a structured JSON plan to engineer features for this dataset.
The plan MUST be a dictionary with these string keys:
- "datetime_features": list of date/time columns to extract features from (year, month, day, etc.).
- "categorical_encoding": dictionary detailing columns to encode and method (e.g., {{"city": "one_hot", "priority": "ordinal"}}).
- "numerical_scaling": description of how numerical columns should be scaled (e.g., standard, minmax).
- "new_features": list of specific new interaction or ratio features to create.
- "reasoning": a short sentence explaining these decisions.

Output EXACTLY valid JSON, nothing else."""

FEATURE_ENGINEER_EXECUTE_PROMPT = """You are a Python Data Engineer. Write a script to engineer features based on the following plan.

Dataset path: {cleaned_dataset_path}
Output dataset path (MUST save to this): {engineered_dataset_path}
Target column (DO NOT scale/encode this): {target_column}
Human Feedback/Instructions: {human_feedback}

Feature Engineering Plan:
{feature_plan}

Instructions:
1. Load dataset with pandas.
2. Apply the feature engineering plan, adjusting carefully for any Human Feedback. Use sklearn preprocessing where appropriate (OneHotEncoder, StandardScaler, etc.).
3. Save the final engineered dataframe to the output path using `df.to_csv(..., index=False)`.
4. Print a JSON object describing before/after stats: {{"features_before": x, "features_after": y, "new_columns": [...], "dropped_columns": [...]}}
5. Provide ONLY the Python code in ```python ... ``` blocks.
"""

FEATURE_SELECTION_PROMPT = """You are an ML Engineer performing feature selection.
Analyze the provided feature importance scores and multicollinearity metrics.

Feature Importances:
{feature_importances}

Task:
Determine which features should be dropped to improve model generalization and reduce multicollinearity.

Output a structured JSON dictionary with:
- "features_to_drop": list of string column names to drop.
- "reasoning": explanation of why these were chosen to be dropped.
Output EXACTLY valid JSON, nothing else."""

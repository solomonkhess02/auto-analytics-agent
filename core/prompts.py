"""
Prompt templates used by the LLM agents.
"""

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
   - descriptive_stats (dictionary from `df.describe(include='all')`, i.e. a dict of column -> {{stat: value}})
   - sample_rows (list of the first 5 rows, each as a dictionary of column -> value)
   - correlations (dictionary from `df.corr(numeric_only=True)`, i.e. a nested dict of column -> {{column: float}}; use an empty dict {{}} if there are no numeric columns)
3. Outputs the final result EXACTLY as a valid JSON string using `json.dumps()`.
4. Ensure the output only contains the JSON string and absolutely nothing else. Print the JSON at the very end.

IMPORTANT:
- Use proper error handling when reading the file.
- Replace every NaN / NaT / infinite value with null so the printed string is strictly valid JSON (e.g. round-trip through `df.where(pd.notnull(df), None)` for sample rows, and pass `default=str` to `json.dumps`)."""

PROFILER_SUMMARY_PROMPT = """You are a Data Science consultant. Review these dataset statistics:
Shape: {shape}
Missing Values: {missing_values}
Data Types: {dtypes}

Write a concise, professional 3-sentence summary of this dataset.
Mention its size, any data quality issues (like missing values), and the general composition of the columns."""




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




FEATURE_ENGINEER_PLAN_PROMPT = """You are a Senior ML Engineer planning feature engineering for a dataset.
Review the data profile and the cleaned data info below carefully.

Task Type: {task_type}
Target Column: {target_column}
Profile Summary:
{profile_summary}

Create a structured JSON plan to engineer features for this dataset.
The plan MUST be a dictionary with these string keys:
- "datetime_features": list of date/time columns to extract features from (year, month, day, etc.).
- "categorical_encoding": dictionary detailing columns to encode and method (e.g., {{"city": "one_hot", "priority": "ordinal"}}). Do not encode unique identifiers (like 'id', 'name').
- "numerical_scaling": description of how numerical columns should be scaled (e.g., standard, minmax).
- "new_features": list of specific new interaction or ratio features to create.
- "reasoning": a short sentence explaining these decisions.

IMPORTANT RULES:
1. Never encode unique identifier columns (e.g. 'id', 'name'). They should be ignored or dropped.
2. If calculating new features (like ratios) from numerical columns, specify that they must be calculated BEFORE scaling the numerical columns.

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
2. Apply the feature engineering plan, adjusting carefully for any Human Feedback. 
   - IMPORTANT: Calculate any `new_features` (like ratios or sums) FIRST, BEFORE applying any numerical scaling to the base columns.
   - IMPORTANT: Do not apply encoding or scaling to unique identifier columns like 'id' or 'name'.
   - Use sklearn preprocessing where appropriate (OneHotEncoder, StandardScaler, etc.).
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


MODEL_TRAINING_PROMPT = """You are a Python ML Engineer. Write a script to train multiple machine learning models.

Engineered dataset path: {engineered_dataset_path}
Target column: {target_column}
Task type: {task_type}
Artifacts directory: {artifacts_dir}

Instructions:
1. Load the engineered dataset with pandas.
2. Separate features (X) and target (y). Drop any unique identifiers (like 'id', 'name') if they are still present.
3. Split the data into train and test sets (test_size=0.2, random_state=42) and save them to '{artifacts_dir}/train.csv' and '{artifacts_dir}/test.csv' respectively.
4. Based on the task type ({task_type}), select at least 3 models to train:
   - For classification: RandomForestClassifier, GradientBoostingClassifier, and LogisticRegression.
   - For regression: RandomForestRegressor, GradientBoostingRegressor, and LinearRegression.
5. For each model:
   - Use cross_val_score (with cv=3) on the training set to get cross-validation scores. Use accuracy/ROC-AUC for classification and R2/negative RMSE for regression.
   - Fit the model on the full training set.
   - Measure training time in seconds.
   - Save the trained model to the artifacts directory as '{artifacts_dir}/' + model_name_sanitized + '.joblib'. Use standard joblib.dump.
   - Record the EXACT path you saved the model to and include it as "model_file" in the output (see format below).
6. Print a JSON list of training results at the very end in this exact format:
   [
     {{
       "model_name": "Random Forest",
       "model_type": "RandomForestClassifier",
       "model_file": "{artifacts_dir}/random_forest.joblib",
       "hyperparameters": {{"n_estimators": 100, ...}},
       "training_time_seconds": 1.25,
       "cross_val_scores": [0.82, 0.85, 0.81],
       "cross_val_mean": 0.8267
     }},
     ...
   ]
7. Provide ONLY the Python code in ```python ... ``` blocks.
"""


MODEL_EVALUATION_PROMPT = """You are a Python Data Scientist. Write a script to evaluate the best model and generate visual plots and reports.

Artifacts directory: {artifacts_dir}
Target column: {target_column}
Task type: {task_type}
Best model name: {best_model_name}
Best model file: {best_model_file}
Report HTML path: {report_html_path}

Instructions:
1. Load the test dataset '{artifacts_dir}/test.csv'.
2. Load the best trained model using joblib from '{best_model_file}'.
3. Separate features (X_test) and target (y_test).
4. Run predictions on the test set.
5. Calculate evaluation metrics on the test set:
   - For classification: accuracy, precision, recall, f1_score, and confusion matrix.
   - For regression: r2, mae, mse, rmse.
6. Get feature importances or model coefficients. Create a bar chart showing the top 10 feature importances/coefficients, and save it to '{artifacts_dir}/feature_importance.png'.
7. Generate plots based on task type:
   - For classification: Generate and save confusion matrix heatmap to '{artifacts_dir}/confusion_matrix.png'.
   - For regression: Generate and save actual vs predicted scatter plot to '{artifacts_dir}/actual_vs_predicted.png'.
8. Generate a professional HTML report and save it to '{report_html_path}'. The HTML report should be beautiful, modern (styled with a clean dark/light UI or inline CSS), and display:
   - Evaluation metrics table.
   - Top 10 feature importances.
   - Embedded images of the generated plots (using relative paths or base64, relative path to `{artifacts_dir}` is preferred).
9. Print a JSON object describing the evaluation results:
   {{
     "best_model_name": "{best_model_name}",
     "metrics": {{"accuracy": x, "precision": y, ...}},
     "confusion_matrix": [[...], ...],
     "classification_report": "...",
     "feature_importances": {{"col1": 0.45, "col2": 0.2, ...}},
     "plot_paths": ["{artifacts_dir}/feature_importance.png", ...],
     "report_html_path": "{report_html_path}"
   }}
10. Provide ONLY the Python code in ```python ... ``` blocks.
"""


EVALUATION_SUMMARY_PROMPT = """You are a Senior Data Scientist. Review these model evaluation metrics:
Best Model: {best_model_name}
Task Type: {task_type}
Metrics: {metrics}
Top Features: {top_features}

Write a concise, professional 3-sentence summary of the model performance.
Mention the best model, its key performance metric value on the test set, and what features were most influential in the model's predictions.
"""

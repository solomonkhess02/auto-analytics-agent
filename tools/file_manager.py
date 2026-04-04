"""
File Manager — Safe file operations for agents.

WHAT THIS FILE DOES:
    Provides functions for agents to read/write datasets and artifacts.
    All file operations are restricted to allowed directories (data/, reports/)
    so agents can't accidentally access system files.

WHY NOT JUST USE pandas.read_csv() DIRECTLY:
    1. Path validation — agents can only access data/ and reports/
    2. Auto-creates directories if they don't exist
    3. Consistent error handling across all agents
    4. Central place to add logging, size limits, etc. later

USAGE:
    from tools.file_manager import load_dataset, save_dataset, get_project_root

    df = load_dataset("data/sales.csv")
    save_dataset(df, "data/cleaned_sales.csv")
"""

import os
import pandas as pd
import pickle
from typing import Optional


# --- Project paths ---
# These are set relative to the project root directory

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    # This file is at tools/file_manager.py, so root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir() -> str:
    """Get the path to the data/ directory."""
    return os.path.join(get_project_root(), "data")


def get_reports_dir() -> str:
    """Get the path to the reports/ directory."""
    return os.path.join(get_project_root(), "reports")


# --- Path Safety ---

# Only these directories are allowed for agent file operations
ALLOWED_DIRS = ["data", "reports", "tmp"]


def _validate_path(filepath: str) -> str:
    """
    Validate that a file path is within an allowed directory.

    This prevents agents from reading/writing to system files or
    other sensitive locations. Only data/, reports/, and tmp/ are allowed.

    Args:
        filepath: Path to validate (can be relative or absolute)

    Returns:
        The absolute, validated path

    Raises:
        PermissionError: If the path is outside allowed directories
    """
    # Convert to absolute path
    if not os.path.isabs(filepath):
        filepath = os.path.join(get_project_root(), filepath)

    filepath = os.path.abspath(filepath)
    project_root = get_project_root()

    # On Windows, check if both paths are on the same drive
    # (os.path.relpath crashes if they're on different drives like C: vs D:)
    if os.path.splitdrive(filepath)[0].lower() != os.path.splitdrive(project_root)[0].lower():
        raise PermissionError(
            f"Access denied: path is outside the project drive. "
            f"Attempted to access: {filepath}"
        )

    # Check if the path is within an allowed directory
    relative = os.path.relpath(filepath, project_root)
    top_dir = relative.split(os.sep)[0]

    if top_dir not in ALLOWED_DIRS:
        raise PermissionError(
            f"Access denied: agents can only access {ALLOWED_DIRS} directories. "
            f"Attempted to access: {filepath}"
        )

    return filepath


# --- Dataset Operations ---

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset into a pandas DataFrame.

    Args:
        filepath: Path to the dataset (relative to project root or absolute)
                  Supports .csv, .xlsx, .xls

    Returns:
        pandas DataFrame with the loaded data

    Example:
        df = load_dataset("data/sales.csv")
        print(df.shape)  # (10000, 15)
    """
    filepath = _validate_path(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # Choose the right pandas reader based on file extension
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Use .csv, .xlsx, or .xls"
        )


def save_dataset(df: pd.DataFrame, filepath: str, index: bool = False) -> str:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df:       The DataFrame to save
        filepath: Where to save it (relative to project root or absolute)
        index:    Whether to include the DataFrame index in the CSV

    Returns:
        The absolute path where the file was saved

    Example:
        save_dataset(cleaned_df, "data/cleaned_sales.csv")
    """
    filepath = _validate_path(filepath)

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_csv(filepath, index=index)
    return filepath


# --- Artifact Operations ---

def save_artifact(obj: object, filepath: str) -> str:
    """
    Save a Python object (model, scaler, encoder) as a pickle file.

    This is how we save trained ML models and preprocessing objects
    so they can be loaded later for predictions.

    Args:
        obj:      Any Python object (sklearn model, scaler, etc.)
        filepath: Where to save it (must end in .pkl)

    Returns:
        The absolute path where the file was saved

    Example:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        save_artifact(model, "reports/models/random_forest.pkl")
    """
    filepath = _validate_path(filepath)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    return filepath


def load_artifact(filepath: str) -> object:
    """
    Load a previously saved Python object from a pickle file.

    Args:
        filepath: Path to the .pkl file

    Returns:
        The deserialized Python object

    Example:
        model = load_artifact("reports/models/random_forest.pkl")
        predictions = model.predict(X_test)
    """
    filepath = _validate_path(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")

    with open(filepath, "rb") as f:
        return pickle.load(f)


# --- File Utilities ---

def read_file(filepath: str) -> str:
    """
    Read a text file and return its contents as a string.

    Args:
        filepath: Path to the text file

    Returns:
        File contents as a string

    Example:
        report = read_file("reports/summary.md")
    """
    filepath = _validate_path(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_file(content: str, filepath: str) -> str:
    """
    Write text content to a file.

    Args:
        content:  The text to write
        filepath: Where to save it

    Returns:
        The absolute path where the file was saved

    Example:
        write_file("# Analysis Report\n...", "reports/report.md")
    """
    filepath = _validate_path(filepath)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def list_files(directory: str, extension: Optional[str] = None) -> list[str]:
    """
    List all files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to list (e.g., "data", "reports")
        extension: Optional filter like ".csv" or ".pkl"

    Returns:
        List of filenames

    Example:
        csv_files = list_files("data", extension=".csv")
        # ["sales.csv", "customers.csv"]
    """
    dir_path = _validate_path(directory)

    if not os.path.isdir(dir_path):
        return []

    files = os.listdir(dir_path)

    if extension:
        files = [f for f in files if f.endswith(extension)]

    return sorted(files)

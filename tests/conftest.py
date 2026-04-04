"""
Shared test fixtures for pytest.
"""

import pytest
import pandas as pd
import numpy as np
import os


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV dataset for testing."""
    data = {
        "age": [25, 30, np.nan, 45, 50, 35, 28, 40, 33, 55],
        "salary": [50000, 60000, 70000, 80000, 90000, 55000, 48000, 75000, 62000, 95000],
        "city": ["NY", "LA", "NY", "SF", "LA", np.nan, "SF", "NY", "LA", "SF"],
        "purchased": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

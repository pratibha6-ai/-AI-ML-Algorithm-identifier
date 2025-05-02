import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_dataset(file) -> Tuple[pd.DataFrame, str]:
    """Load and validate dataset from uploaded file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise ValueError(
                "Unsupported file format. Please upload CSV or Excel file.")

        if df.empty:
            raise ValueError("The uploaded file is empty.")

        return df, "Success"
    except Exception as e:
        return None, str(e)


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic statistics about the dataset."""
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns":
        len(df.select_dtypes(include=['object']).columns),
        "missing_values": df.isnull().sum().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # in MB
    }
    return stats


def identify_target_variable_type(series: pd.Series) -> str:
    """Identify the type of target variable."""
    if series.dtype in ['int64', 'float64']:
        if len(series.unique()) < 10:
            return 'classification'
        return 'regression'
    return 'classification'

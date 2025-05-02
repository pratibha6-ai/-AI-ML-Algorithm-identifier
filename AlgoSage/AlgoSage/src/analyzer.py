import pandas as pd
from typing import Dict, Any

class DatasetAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_features(self) -> Dict[str, Any]:
        """Analyze dataset features and characteristics."""
        analysis = {
            "data_shape": self.df.shape,
            "feature_types": {
                "numerical": list(self.df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical": list(self.df.select_dtypes(include=['object']).columns)
            },
            "missing_values": self.df.isnull().sum().to_dict(),
            "unique_values": {col: self.df[col].nunique() for col in self.df.columns},
            "skewness": self.df.select_dtypes(include=['int64', 'float64']).skew().to_dict()
        }
        return analysis
    
    def get_complexity_score(self) -> float:
        """Calculate dataset complexity score."""
        numerical_cols = len(self.df.select_dtypes(include=['int64', 'float64']).columns)
        categorical_cols = len(self.df.select_dtypes(include=['object']).columns)
        unique_vals_ratio = sum(self.df[col].nunique() / len(self.df) 
                              for col in self.df.columns) / len(self.df.columns)
        
        complexity = (0.4 * numerical_cols + 
                     0.3 * categorical_cols + 
                     0.3 * unique_vals_ratio)
        return min(complexity, 1.0)

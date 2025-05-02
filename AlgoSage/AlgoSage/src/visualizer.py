from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_feature_importance_plot(self, feature_importance: Dict[str, float]):
        """Create feature importance visualization."""
        fig = px.bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            title='Feature Importance',
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        return fig

    def create_data_distribution_plot(self, column: str):
        """Create distribution plot for a numeric column."""
        fig = px.histogram(
            self.df, x=column,
            title=f'Distribution of {column}',
            marginal='box'
        )
        return fig

    def create_algorithm_comparison_plot(self, recommendations: List[Dict[str, Any]]):
        """Create algorithm comparison visualization."""
        fig = go.Figure(data=[
            go.Bar(
                x=[algo['name'] for algo in recommendations],
                y=[algo['score'] for algo in recommendations],
                text=[f"{algo['score']:.2f}" for algo in recommendations],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Algorithm Recommendation Scores',
            xaxis_title='Algorithm',
            yaxis_title='Compatibility Score',
            yaxis_range=[0, 1]
        )
        return fig
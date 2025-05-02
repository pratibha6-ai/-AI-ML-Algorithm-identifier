import streamlit as st
from typing import Dict, Any, List

class AlgorithmRecommender:
    def __init__(self):
        self.algorithms = {
            'classification': [
                {'name': 'Random Forest', 'suitable_for': ['medium to large datasets', 'mixed feature types'], 'pros': ['handles non-linear relationships', 'robust to outliers'], 'sklearn_name': 'RandomForestClassifier'},
                {'name': 'Logistic Regression', 'suitable_for': ['linear relationships', 'binary classification'], 'pros': ['simple', 'interpretable'], 'sklearn_name': 'LogisticRegression'},
                {'name': 'XGBoost', 'suitable_for': ['complex relationships', 'large datasets'], 'pros': ['high performance', 'handles missing values'], 'sklearn_name': 'XGBClassifier'},
                {'name': 'Support Vector Machine (SVM)', 'suitable_for': ['small to medium datasets', 'high-dimensional data'], 'pros': ['effective in high dimensions', 'robust with kernel trick'], 'sklearn_name': 'SVC'},
                {'name': 'K-Nearest Neighbors (KNN)', 'suitable_for': ['small datasets', 'non-parametric approaches'], 'pros': ['simple', 'no training phase'], 'sklearn_name': 'KNeighborsClassifier'},
                {'name': 'Naive Bayes', 'suitable_for': ['text classification', 'probabilistic classification'], 'pros': ['fast', 'handles categorical features well'], 'sklearn_name': 'GaussianNB'},
                {'name': 'Gradient Boosting', 'suitable_for': ['complex relationships', 'moderate datasets'], 'pros': ['high accuracy', 'handles missing data'], 'sklearn_name': 'GradientBoostingClassifier'},
                {'name': 'Neural Networks (MLP)', 'suitable_for': ['deep learning tasks', 'large datasets'], 'pros': ['captures complex relationships', 'adaptive'], 'sklearn_name': 'MLPClassifier'},
                {'name': 'AdaBoost', 'suitable_for': ['boosting weak learners', 'binary classification'], 'pros': ['improves weak models', 'adaptive'], 'sklearn_name': 'AdaBoostClassifier'},
                {'name': 'Decision Tree', 'suitable_for': ['small datasets', 'interpretable models'], 'pros': ['simple to understand', 'no need for feature scaling'], 'sklearn_name': 'DecisionTreeClassifier'}
            ],
            'regression': [
                {'name': 'Linear Regression', 'suitable_for': ['linear relationships', 'continuous data'], 'pros': ['simple', 'interpretable'], 'sklearn_name': 'LinearRegression'},
                {'name': 'Random Forest Regressor', 'suitable_for': ['non-linear relationships', 'complex datasets'], 'pros': ['handles non-linearity', 'robust to outliers'], 'sklearn_name': 'RandomForestRegressor'},
                {'name': 'XGBoost Regressor', 'suitable_for': ['complex relationships', 'large datasets'], 'pros': ['high performance', 'handles missing values'], 'sklearn_name': 'XGBRegressor'},
                {'name': 'Support Vector Regressor (SVR)', 'suitable_for': ['high-dimensional data', 'complex relationships'], 'pros': ['effective in high dimensions', 'robust with kernel trick'], 'sklearn_name': 'SVR'},
                {'name': 'K-Nearest Neighbors Regressor (KNN)', 'suitable_for': ['small datasets', 'non-parametric approaches'], 'pros': ['simple', 'no training phase'], 'sklearn_name': 'KNeighborsRegressor'},
                {'name': 'Gradient Boosting Regressor', 'suitable_for': ['complex relationships', 'moderate datasets'], 'pros': ['high accuracy', 'handles missing data'], 'sklearn_name': 'GradientBoostingRegressor'},
                {'name': 'Neural Networks (MLP Regressor)', 'suitable_for': ['deep learning tasks', 'large datasets'], 'pros': ['captures complex relationships', 'adaptive'], 'sklearn_name': 'MLPRegressor'},
                {'name': 'AdaBoost Regressor', 'suitable_for': ['boosting weak learners', 'non-linear regression'], 'pros': ['improves weak models', 'adaptive'], 'sklearn_name': 'AdaBoostRegressor'},
                {'name': 'Decision Tree Regressor', 'suitable_for': ['small datasets', 'interpretable models'], 'pros': ['simple to understand', 'no need for feature scaling'], 'sklearn_name': 'DecisionTreeRegressor'}
            ]
        }

    def recommend_algorithms(self, analysis: Dict[str, Any], problem_type: str) -> List[Dict[str, Any]]:
        """Recommend algorithms based on dataset analysis."""
        recommendations = []
        available_algorithms = self.algorithms[problem_type]
        
        for algo in available_algorithms:
            score = self._calculate_algorithm_score(analysis, algo)
            recommendations.append({**algo, 'score': score})
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]

    def _calculate_algorithm_score(self, analysis: Dict[str, Any], algorithm: Dict[str, Any]) -> float:
        """Calculate compatibility score for an algorithm."""
        score = 0.0
        
        rows = analysis['data_shape'][0]
        if rows < 1000:
            score += 0.3 if algorithm['name'] in ['Logistic Regression', 'Linear Regression'] else 0.1
        elif rows < 10000:
            score += 0.3 if algorithm['name'] in ['Random Forest', 'XGBoost'] else 0.2
        else:
            score += 0.4 if algorithm['name'] == 'XGBoost' else 0.2
        
        if len(analysis['feature_types']['numerical']) > len(analysis['feature_types']['categorical']):
            score += 0.2 if algorithm['name'] in ['Linear Regression', 'XGBoost'] else 0.1
        else:
            score += 0.2 if algorithm['name'] == 'Random Forest' else 0.1
        
        return min(score, 1.0)

if __name__ == "__main__":
    main()



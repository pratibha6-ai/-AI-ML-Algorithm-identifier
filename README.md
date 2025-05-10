# 🤖 ML Algorithm Recommender System

The **ML Algorithm Recommender System** is an intelligent web application that automatically analyzes datasets and recommends suitable machine learning algorithms based on the dataset characteristics and the problem type.

> 🔍 Just upload your CSV/Excel dataset — the system does the rest.

## 🚀 Features

- 📊 **Dataset Analysis**: Automatically computes stats (rows, columns, missing values).
- 🎯 **Target Identification**: Select one or more target variables to determine the problem type (classification or regression).
- 🧠 **Intelligent Recommendations**: Suggests ML algorithms using rule-based scoring.
- 📈 **Visualizations**: Interactive Plotly charts to compare algorithm suitability and target distribution.
- 🔄 **Supports Multiple Formats**: Upload `.csv`, `.xlsx`, or `.xls` files.

---

## 🛠️ Technologies Used

| Area | Tools / Frameworks |
|------|---------------------|
| Frontend | [Streamlit](https://streamlit.io) |
| Backend Logic | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly |
| ML Knowledge Base | Rule-based engine |
| File Handling | CSV, Excel (via pandas) |

---

## 🧠 Algorithms Considered

The system currently supports over 10 ML algorithms:

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest
- SVM
- KNN
- Naive Bayes
- Gradient Boosting (XGBoost, LightGBM)
- MLPClassifier (Neural Net)
- AdaBoost
- Extra Trees Classifier

### Regression
- Linear Regression
- Ridge & Lasso
- Decision Tree Regressor
- Random Forest Regressor
- SVR
- KNN Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- MLPRegressor
- Extra Trees Regressor

---

## 📥 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ml-algo-recommender.git
   cd ml-algo-recommender

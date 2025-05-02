import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from src.utils import load_dataset, get_basic_stats, identify_target_variable_type
from src.analyzer import DatasetAnalyzer
from src.recommender import AlgorithmRecommender
from src.visualizer import DataVisualizer

# Load lottie from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(
        page_title="🤖 ML Algorithm Recommender",
        page_icon="🧠",
        layout="wide"
    )

    st.markdown("""
        <style>
            .main {background-color: #f5f7fa;}
            h1, h2, h3 {color: #37474f;}
            .stButton>button {background-color: #4CAF50; color: white;}
            .metric-container {
                background: #ffffff;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st_lottie(
            load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_9cyyl8i4.json"),
            height=250, speed=1, loop=True, quality="high"
        )
        st.markdown("### 🤖 Smart AI Assistant")
        st.caption("Upload dataset → get ML recommendations.")

    st.title("🔍 ML Algorithm Recommender")
    st.markdown("""
    <div style="font-size:18px;">
        Upload your dataset 📂 and get smart ML algorithm suggestions 🧠 
        based on automatic data analysis 📊.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📁 Upload your dataset (CSV or Excel)", 
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file:
        df, message = load_dataset(uploaded_file)

        if df is None:
            st.error(f"❌ Error loading dataset: {message}")
            return

        st.header("📄 Dataset Overview")
        stats = get_basic_stats(df)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📏 Rows", stats["rows"])
        with col2:
            st.metric("📐 Columns", stats["columns"])
        with col3:
            st.metric("🚫 Missing Values", stats["missing_values"])

        st.markdown("---")
        st.subheader("🎯 Select Target Variable(s)")
        target_variables = st.multiselect(
            "Choose one or more target variables",
            options=df.columns.tolist()
        )

        if st.button("🚀 Analyze Dataset") and target_variables:
            with st.spinner("Crunching the numbers... 🔎"):
                st_lottie(
                    load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json"),
                    height=150, speed=1, loop=True
                )
                analyzer = DatasetAnalyzer(df)
                analysis = analyzer.analyze_features()

                problem_types = {
                    var: identify_target_variable_type(df[var]) 
                    for var in target_variables
                }

                recommender = AlgorithmRecommender()
                recommendations = {
                    var: recommender.recommend_algorithms(analysis, problem_types[var]) 
                    for var in target_variables
                }

                visualizer = DataVisualizer(df)

            st.success("✅ Analysis Complete!")
            st.markdown("---")
            st.subheader("📊 Results & Recommendations")

            for idx, var in enumerate(target_variables):
                st.markdown(f"### 🔹 **{var}** — *{problem_types[var].title()} Problem*")

                st.markdown(f"#### 🧠 Recommended Algorithms for `{var}`")
                for i, algo in enumerate(recommendations[var], 1):
                    with st.expander(f"{i}. {algo['name']} (⭐ {algo['score']:.2f})"):
                        st.markdown("**✅ Best for:**")
                        st.markdown(" • " + "<br> • ".join(algo['suitable_for']), unsafe_allow_html=True)
                        st.markdown("**👍 Pros:**")
                        st.markdown(" • " + "<br> • ".join(algo['pros']), unsafe_allow_html=True)

                st.markdown("#### 📈 Visualizations")
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = visualizer.create_algorithm_comparison_plot(recommendations[var])
                    st.plotly_chart(fig1, use_container_width=True, key=f"algo_plot_{idx}")

                with col2:
                    if df[var].dtype in ['int64', 'float64']:
                        fig2 = visualizer.create_data_distribution_plot(var)
                        st.plotly_chart(fig2, use_container_width=True, key=f"dist_plot_{idx}")

if __name__ == "__main__":
    main()

"""Streamlit Web Application: Student Performance Prediction and Classification System.

Provides an intuitive, interactive interface for educators, researchers, and
students to analyze academic performance risk factors, make early predictions,
and explore model evaluation metrics.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

# Add project root to sys.path to guarantee clean imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    CLASS_DESCRIPTIONS,
    CLASS_LABELS,
    CONFUSION_MATRIX_PATH,
    EVALUATION_RESULTS_PATH,
    FEATURE_METADATA,
    GRADE_BINS,
    NUMERICAL_FEATURES,
    UCI_LANDING_PAGE,
)
from src.evaluate import load_evaluation_results
from src.predict import (
    load_classification_pipeline,
    load_regression_pipeline,
    predict_student,
)

# Page configuration
st.set_page_config(
    page_title="Student Performance AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_pipelines():
    """Cache trained machine learning pipelines in memory."""
    try:
        reg_pipe = load_regression_pipeline()
        clf_pipe = load_classification_pipeline()
        return reg_pipe, clf_pipe, None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_data
def get_cached_evaluation_results() -> Tuple[Dict[str, Any], str]:
    """Cache evaluation results JSON from disk."""
    try:
        results = load_evaluation_results(EVALUATION_RESULTS_PATH)
        return results, None
    except Exception as exc:
        return None, str(exc)


def render_sidebar():
    """Render the application navigation sidebar (100% offline)."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 12px 0; background-color: #f0f2f6; border-radius: 8px; margin-bottom: 12px;">
            <span style="font-size: 38px;">🎓</span>
            <h3 style="margin: 4px 0 0 0; color: #1E3A8A; font-weight: 700;">Student Success</h3>
            <p style="margin: 0; font-size: 12px; color: #64748B;">AI Early Warning Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Overview",
            "🎯 Predict Performance",
            "📊 Model Evaluation",
            "📖 About & Methodology",
        ],
        index=1,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**System Status:**\n- 🟢 Early-Prediction Pipeline Active\n- 📁 UCI Mathematics Cohort\n- 🔒 Leak-Free Feature Schema"
    )
    return choice


def render_overview():
    """Render Project Overview section."""
    st.title("🎓 Student Performance Prediction & Classification System")
    st.markdown(
        """
    Welcome to the **Student Performance Analysis & Early Warning System**.

    This tool is designed to assist academic advisors, educators, and institutions in identifying
    students who may benefit from early academic interventions before final grades are recorded.
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 📈 Regression Task")
        st.write(
            """
        - **Objective:** Estimate a student's final numerical grade on a continuous 0 to 20 scale.
        - **Target Variable:** `G3` (Final academic term grade).
        - **Early Prediction Protocol:** Historical term evaluations (`G1`, `G2`) and `G3` are
          strictly excluded from inputs. Only pre-period demographic, socio-economic, and behavioral factors are used.
        """
        )

    with col2:
        st.subheader("2. 🏷️ Classification Task")
        st.write(
            """
        - **Objective:** Categorize expected academic achievement into three actionable tiers:
          - **🔴 Low (0 – 9):** Below passing threshold; prioritized for proactive academic support.
          - **🟡 Medium (10 – 13):** Satisfactory performance meeting baseline graduation criteria.
          - **🟢 High (14 – 20):** Strong academic achievement and potential for honors.
        """
        )

    st.markdown("---")
    st.subheader("📚 Dataset & Transparency")
    st.markdown(
        f"""
    - **Source:** [UCI Machine Learning Repository: Student Performance Dataset]({UCI_LANDING_PAGE})
    - **Context:** Secondary education mathematics courses across two Portuguese schools.
    - **Independent Validation:** All machine learning models were developed with strict train/test separation
      and evaluated on held-out test data.
    """
    )


def render_prediction_page(reg_pipe, clf_pipe):
    """Render interactive student feature input form and prediction engine."""
    st.title("🎯 Predict Student Performance")
    st.markdown(
        "Enter the student's demographic, academic background, and behavioral information below. "
        "The system will compute both a continuous grade prediction and a multi-class performance tier."
    )

    if reg_pipe is None or clf_pipe is None:
        st.error(
            "⚠️ Trained model pipelines could not be loaded. Please ensure the models are trained "
            "by running `python -m src.train` in your project terminal."
        )
        return

    # Organize features into logical thematic tabs for usability
    st.subheader("Student Profile Input")
    tab_demo, tab_acad, tab_fam, tab_life = st.tabs(
        [
            "👤 Demographics & Identity",
            "🏫 School & Study Habits",
            "👨‍👩‍👦 Family & Environment",
            "⚽ Lifestyle & Well-being",
        ]
    )

    form_values = {}

    with tab_demo:
        c1, c2, c3 = st.columns(3)
        with c1:
            form_values["school"] = st.selectbox(
                FEATURE_METADATA["school"]["label"],
                CATEGORICAL_OPTIONS["school"],
                format_func=lambda x: "GP - Gabriel Pereira" if x == "GP" else "MS - Mousinho da Silveira",
                help=FEATURE_METADATA["school"]["help"],
            )
            form_values["sex"] = st.selectbox(
                FEATURE_METADATA["sex"]["label"],
                CATEGORICAL_OPTIONS["sex"],
                format_func=lambda x: "Female" if x == "F" else "Male",
                help=FEATURE_METADATA["sex"]["help"],
            )
        with c2:
            form_values["age"] = st.slider(
                FEATURE_METADATA["age"]["label"],
                min_value=FEATURE_METADATA["age"]["min"],
                max_value=FEATURE_METADATA["age"]["max"],
                value=FEATURE_METADATA["age"]["default"],
                help=FEATURE_METADATA["age"]["help"],
            )
            form_values["address"] = st.selectbox(
                FEATURE_METADATA["address"]["label"],
                CATEGORICAL_OPTIONS["address"],
                format_func=lambda x: "Urban" if x == "U" else "Rural",
                help=FEATURE_METADATA["address"]["help"],
            )
        with c3:
            form_values["famsize"] = st.selectbox(
                FEATURE_METADATA["famsize"]["label"],
                CATEGORICAL_OPTIONS["famsize"],
                format_func=lambda x: "Greater than 3 (>3)" if x == "GT3" else "Up to 3 (≤3)",
                help=FEATURE_METADATA["famsize"]["help"],
            )
            form_values["Pstatus"] = st.selectbox(
                FEATURE_METADATA["Pstatus"]["label"],
                CATEGORICAL_OPTIONS["Pstatus"],
                format_func=lambda x: "Living Together" if x == "T" else "Apart / Separated",
                help=FEATURE_METADATA["Pstatus"]["help"],
            )

    with tab_acad:
        c1, c2, c3 = st.columns(3)
        with c1:
            form_values["studytime"] = st.select_slider(
                FEATURE_METADATA["studytime"]["label"],
                options=[1, 2, 3, 4],
                value=FEATURE_METADATA["studytime"]["default"],
                format_func=lambda x: {
                    1: "< 2 hours/week",
                    2: "2 to 5 hours/week",
                    3: "5 to 10 hours/week",
                    4: "> 10 hours/week",
                }[x],
                help=FEATURE_METADATA["studytime"]["help"],
            )
            form_values["failures"] = st.number_input(
                FEATURE_METADATA["failures"]["label"],
                min_value=0,
                max_value=4,
                value=0,
                help=FEATURE_METADATA["failures"]["help"],
            )
            form_values["absences"] = st.number_input(
                FEATURE_METADATA["absences"]["label"],
                min_value=0,
                max_value=93,
                value=4,
                help=FEATURE_METADATA["absences"]["help"],
            )
        with c2:
            form_values["schoolsup"] = st.selectbox(
                FEATURE_METADATA["schoolsup"]["label"],
                CATEGORICAL_OPTIONS["schoolsup"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["schoolsup"]["help"],
            )
            form_values["paid"] = st.selectbox(
                FEATURE_METADATA["paid"]["label"],
                CATEGORICAL_OPTIONS["paid"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["paid"]["help"],
            )
            form_values["higher"] = st.selectbox(
                FEATURE_METADATA["higher"]["label"],
                CATEGORICAL_OPTIONS["higher"],
                index=1,
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["higher"]["help"],
            )
        with c3:
            form_values["reason"] = st.selectbox(
                FEATURE_METADATA["reason"]["label"],
                CATEGORICAL_OPTIONS["reason"],
                format_func=lambda x: x.capitalize(),
                help=FEATURE_METADATA["reason"]["help"],
            )
            form_values["traveltime"] = st.select_slider(
                FEATURE_METADATA["traveltime"]["label"],
                options=[1, 2, 3, 4],
                value=1,
                format_func=lambda x: {
                    1: "< 15 min",
                    2: "15 to 30 min",
                    3: "30 min to 1 hr",
                    4: "> 1 hour",
                }[x],
                help=FEATURE_METADATA["traveltime"]["help"],
            )

    with tab_fam:
        c1, c2 = st.columns(2)
        with c1:
            form_values["Medu"] = st.select_slider(
                FEATURE_METADATA["Medu"]["label"],
                options=[0, 1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    0: "None",
                    1: "Primary (4th grade)",
                    2: "5th to 9th grade",
                    3: "Secondary education",
                    4: "Higher education",
                }[x],
                help=FEATURE_METADATA["Medu"]["help"],
            )
            form_values["Mjob"] = st.selectbox(
                FEATURE_METADATA["Mjob"]["label"],
                CATEGORICAL_OPTIONS["Mjob"],
                format_func=lambda x: x.replace("_", " ").title(),
                help=FEATURE_METADATA["Mjob"]["help"],
            )
            form_values["guardian"] = st.selectbox(
                FEATURE_METADATA["guardian"]["label"],
                CATEGORICAL_OPTIONS["guardian"],
                format_func=lambda x: x.capitalize(),
                help=FEATURE_METADATA["guardian"]["help"],
            )
            form_values["famsup"] = st.selectbox(
                FEATURE_METADATA["famsup"]["label"],
                CATEGORICAL_OPTIONS["famsup"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["famsup"]["help"],
            )
        with c2:
            form_values["Fedu"] = st.select_slider(
                FEATURE_METADATA["Fedu"]["label"],
                options=[0, 1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    0: "None",
                    1: "Primary (4th grade)",
                    2: "5th to 9th grade",
                    3: "Secondary education",
                    4: "Higher education",
                }[x],
                help=FEATURE_METADATA["Fedu"]["help"],
            )
            form_values["Fjob"] = st.selectbox(
                FEATURE_METADATA["Fjob"]["label"],
                CATEGORICAL_OPTIONS["Fjob"],
                format_func=lambda x: x.replace("_", " ").title(),
                help=FEATURE_METADATA["Fjob"]["help"],
            )
            form_values["famrel"] = st.slider(
                FEATURE_METADATA["famrel"]["label"],
                min_value=1,
                max_value=5,
                value=4,
                help=FEATURE_METADATA["famrel"]["help"],
            )

    with tab_life:
        c1, c2, c3 = st.columns(3)
        with c1:
            form_values["activities"] = st.selectbox(
                FEATURE_METADATA["activities"]["label"],
                CATEGORICAL_OPTIONS["activities"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["activities"]["help"],
            )
            form_values["nursery"] = st.selectbox(
                FEATURE_METADATA["nursery"]["label"],
                CATEGORICAL_OPTIONS["nursery"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["nursery"]["help"],
            )
            form_values["internet"] = st.selectbox(
                FEATURE_METADATA["internet"]["label"],
                CATEGORICAL_OPTIONS["internet"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["internet"]["help"],
            )
        with c2:
            form_values["romantic"] = st.selectbox(
                FEATURE_METADATA["romantic"]["label"],
                CATEGORICAL_OPTIONS["romantic"],
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help=FEATURE_METADATA["romantic"]["help"],
            )
            form_values["freetime"] = st.slider(
                FEATURE_METADATA["freetime"]["label"],
                min_value=1,
                max_value=5,
                value=3,
                help=FEATURE_METADATA["freetime"]["help"],
            )
            form_values["goout"] = st.slider(
                FEATURE_METADATA["goout"]["label"],
                min_value=1,
                max_value=5,
                value=3,
                help=FEATURE_METADATA["goout"]["help"],
            )
        with c3:
            form_values["Dalc"] = st.slider(
                FEATURE_METADATA["Dalc"]["label"],
                min_value=1,
                max_value=5,
                value=1,
                help=FEATURE_METADATA["Dalc"]["help"],
            )
            form_values["Walc"] = st.slider(
                FEATURE_METADATA["Walc"]["label"],
                min_value=1,
                max_value=5,
                value=2,
                help=FEATURE_METADATA["Walc"]["help"],
            )
            form_values["health"] = st.slider(
                FEATURE_METADATA["health"]["label"],
                min_value=1,
                max_value=5,
                value=4,
                help=FEATURE_METADATA["health"]["help"],
            )

    st.markdown("---")
    predict_clicked = st.button("🔮 Generate Student Performance Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing student profile through trained pipelines..."):
            try:
                result = predict_student(form_values, reg_pipeline=reg_pipe, clf_pipeline=clf_pipe)

                st.success("✅ Prediction Generated Successfully")

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.subheader("Estimated Final Numerical Grade")
                    pred_grade = result["predicted_grade"]
                    st.metric(
                        label="Predicted Grade (0 to 20 scale)",
                        value=f"{pred_grade:.2f} / 20.0",
                        help="Continuous regression prediction",
                    )
                    st.progress(float(pred_grade) / 20.0)

                    if pred_grade < 10.0:
                        st.warning("⚠️ Estimated grade is below passing threshold (10.0). Proactive support advised.")
                    elif pred_grade < 14.0:
                        st.info("ℹ️ Estimated grade is within standard passing range (10.0–13.9).")
                    else:
                        st.success("🌟 Estimated grade is in high academic achievement tier (14.0–20.0).")

                with col_res2:
                    st.subheader("Performance Classification Tier")
                    category = result["predicted_category"]
                    cat_desc = result["category_description"]

                    badge_color = {
                        "Low": "🔴",
                        "Medium": "🟡",
                        "High": "🟢",
                    }.get(category, "⚪")

                    st.markdown(f"### {badge_color} Category: **{category}**")
                    st.markdown(f"*{cat_desc}*")

                    if result["class_probabilities"]:
                        st.markdown("**Predicted Class Probabilities:**")
                        prob_df = pd.DataFrame(
                            list(result["class_probabilities"].items()),
                            columns=["Category", "Probability"],
                        ).set_index("Category")
                        st.bar_chart(prob_df, height=180)

                st.info(
                    "📌 **Note:** Model predictions represent statistical estimates based on historical student cohorts. "
                    "They are intended as early warning guidance to assist human educators, not definitive determinations."
                )

            except Exception as exc:
                st.error(f"Error calculating prediction: {exc}")


def render_evaluation_page(eval_data):
    """Render saved model performance metrics and confusion matrix."""
    st.title("📊 Model Evaluation & Validation Report")
    st.markdown(
        "Performance benchmarks generated from offline evaluation on a **held-out test set (20% of data, N=79)**. "
        "These metrics assess generalized performance on unseen records."
    )

    if not eval_data:
        st.warning(
            "Evaluation results not found. Please train models first using `python -m src.train`."
        )
        return

    meta = eval_data.get("metadata", {})
    reg = eval_data.get("regression", {})
    clf = eval_data.get("classification", {})

    st.markdown("### 🏆 Final Selected Models")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Regression Algorithm:** `{reg.get('selected_model')}`")
    with c2:
        st.info(f"**Classification Algorithm:** `{clf.get('selected_model')}`")

    st.markdown("---")
    st.subheader("1. 📈 Regression Model Metrics")
    reg_test = reg.get("test_metrics", {})
    reg_base = reg.get("baseline_metrics", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Mean Absolute Error (MAE)",
            f"{reg_test.get('mae', 0):.4f}",
            delta=f"{-1 * (reg_test.get('mae', 0) - reg_base.get('mae', 0)):.4f} vs Baseline",
            delta_color="normal",
        )
    with col2:
        st.metric(
            "Root Mean Squared Error (RMSE)",
            f"{reg_test.get('rmse', 0):.4f}",
            delta=f"{-1 * (reg_test.get('rmse', 0) - reg_base.get('rmse', 0)):.4f} vs Baseline",
            delta_color="normal",
        )
    with col3:
        st.metric(
            "R² Score",
            f"{reg_test.get('r2', 0):.4f}",
            delta=f"{reg_test.get('r2', 0) - reg_base.get('r2', 0):.4f} vs Baseline",
            delta_color="normal",
        )

    # Cross-validation comparison table
    st.markdown("**Cross-Validation Comparison (5-Fold CV on Training Data):**")
    cv_reg_data = reg.get("cv_comparison", {})
    if cv_reg_data:
        df_cv_reg = pd.DataFrame.from_dict(cv_reg_data, orient="index")
        df_cv_reg.columns = ["CV MAE", "CV R²", "CV RMSE"]
        st.dataframe(df_cv_reg, use_container_width=True)

    st.markdown("---")
    st.subheader("2. 🏷️ Classification Model Metrics")
    clf_test = clf.get("test_metrics", {})
    clf_base = clf.get("baseline_metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Test Accuracy",
            f"{clf_test.get('accuracy', 0):.4f}",
            delta=f"{clf_test.get('accuracy', 0) - clf_base.get('accuracy', 0):.4f} vs Baseline",
        )
    with c2:
        st.metric(
            "F1-Score (Macro Average)",
            f"{clf_test.get('f1_macro', 0):.4f}",
            delta=f"{clf_test.get('f1_macro', 0) - clf_base.get('f1_macro', 0):.4f} vs Baseline",
        )
    with c3:
        st.metric(
            "F1-Score (Weighted Average)",
            f"{clf_test.get('f1_weighted', 0):.4f}",
            delta=f"{clf_test.get('f1_weighted', 0) - clf_base.get('f1_weighted', 0):.4f} vs Baseline",
        )
    with c4:
        st.metric(
            "Baseline Most-Frequent Accuracy",
            f"{clf_base.get('accuracy', 0):.4f}",
        )

    st.markdown("**Confusion Matrix (Test Set):**")
    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix Heatmap", use_container_width=True)
        else:
            cm = clf_test.get("confusion_matrix", [])
            df_cm = pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS)
            st.dataframe(df_cm)

    with col_cm2:
        st.markdown("**Per-Class Classification Report:**")
        report = clf_test.get("classification_report", {})
        if report:
            class_report_rows = []
            for label in CLASS_LABELS:
                if label in report:
                    row = report[label]
                    class_report_rows.append(
                        {
                            "Category": label,
                            "Precision": round(row.get("precision", 0), 3),
                            "Recall": round(row.get("recall", 0), 3),
                            "F1-Score": round(row.get("f1-score", 0), 3),
                            "Support (Students)": int(row.get("support", 0)),
                        }
                    )
            st.dataframe(pd.DataFrame(class_report_rows).set_index("Category"), use_container_width=True)


def render_about_page():
    """Render About, Methodology, and Limitations documentation."""
    st.title("📖 About & Technical Methodology")
    st.markdown(
        """
    ### 1. Dataset Context & Citation
    - **Dataset:** Student Performance (Mathematics Cohort)
    - **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
    - **Reference:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.*
      Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12.

    ### 2. Prediction Problem & Leakage Prevention
    - **Early-Prediction Scenario:** In realistic educational settings, predictive models are most valuable
      at the beginning of an academic year or course. Therefore, intermediate period grades (`G1`, `G2`)
      and final grade (`G3`) are strictly omitted from input features.
    - **Leak-Free Pipelines:** Preprocessing transformations (scaling, imputation, one-hot encoding)
      are contained within Scikit-learn `Pipeline` objects and fit solely on training partitions.

    ### 3. Classification Grade Boundaries
    The classification categories follow standard Portuguese educational achievement standards:
    - **Low (0 – 9):** Failing / Needs Support
    - **Medium (10 – 13):** Satisfactory / Average
    - **High (14 – 20):** Good to Excellent

    ### 4. Ethical Use & Responsible AI Considerations
    - **No Autonomous Decisions:** This system is an advisory tool for early intervention, never an autonomous grading mechanism.
    - **Socio-economic Sensitivity:** Demographic features (parental jobs, education, address) correlate with historical opportunity,
      not inherent ability. Interventions should focus on providing educational resources rather than penalizing background factors.
    """
    )


def main():
    """Application main entrypoint."""
    page = render_sidebar()

    reg_pipe, clf_pipe, pipe_err = get_cached_pipelines()
    eval_results, eval_err = get_cached_evaluation_results()

    if page == "🏠 Overview":
        render_overview()
    elif page == "🎯 Predict Performance":
        render_prediction_page(reg_pipe, clf_pipe)
    elif page == "📊 Model Evaluation":
        render_evaluation_page(eval_results)
    elif page == "📖 About & Methodology":
        render_about_page()


if __name__ == "__main__":
    main()

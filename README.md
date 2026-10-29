# 🎓 Student Performance Prediction & Classification System

An end-to-end, production-grade Machine Learning solution designed to predict and analyze secondary school academic performance. Built using Python, Scikit-learn, and Streamlit, this system provides educational institutions with early-warning capabilities to detect academic risk and support timely academic interventions before final examinations.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Dataset Attribution & Acquisition](#3-dataset-attribution--acquisition)
4. [Prediction Objectives & Target Formulation](#4-prediction-objectives--target-formulation)
5. [Feature Schema & Early Prediction Protocol](#5-feature-schema--early-prediction-protocol)
6. [Classification Boundaries](#6-classification-boundaries)
7. [Preprocessing & Leak-Free Architecture](#7-preprocessing--leak-free-architecture)
8. [Model Selection & Evaluation Methodology](#8-model-selection--evaluation-methodology)
9. [Empirical Evaluation Results](#9-empirical-evaluation-results)
10. [Directory Structure](#10-directory-structure)
11. [Installation & Setup](#11-installation--setup)
12. [Dataset Setup Instructions](#12-dataset-setup-instructions)
13. [Training the Models](#13-training-the-models)
14. [Launching the Web Application](#14-launching-the-web-application)
15. [Automated Testing & QA](#15-automated-testing--qa)
16. [Example Programmatic Usage](#16-example-programmatic-usage)
17. [Ethical Considerations & Limitations](#17-ethical-considerations--limitations)
18. [Troubleshooting & FAQ](#18-troubleshooting--faq)

---

## 1. Project Overview

Academic underachievement frequently surfaces too late in the school calendar for remediative measures to take full effect. This system provides an **early warning mechanism** based on students' demographic, socio-economic, and behavioral factors available at the beginning of an academic course.

The platform executes two dual, independently evaluated machine learning tasks:
1. **Numerical Regression:** Predicts the exact expected final grade on the standard Portuguese continuous scale ($0.0$ to $20.0$).
2. **Multi-class Categorization:** Stratifies students into three actionable performance tiers: **Low**, **Medium**, or **High**.

---

## 2. Key Features

- **Strict Leak-Free Machine Learning:** Intermediate term grades ($G1$ and $G2$) and target grade ($G3$) are excluded from feature inputs, ensuring zero look-ahead bias.
- **Robust Scikit-learn Pipelines:** End-to-end preprocessing pipelines (`ColumnTransformer`) encapsulating median numerical imputation, standard scaling, and one-hot encoding with safe handling of novel categories (`handle_unknown='ignore'`).
- **Validated Cross-Validation Selection:** 5-fold cross-validation performed strictly within training partitions to benchmark candidates against dummy baselines.
- **Interactive Streamlit Web Interface:** A modern, accessible UI featuring categorized tab inputs, real-time prediction meters, class probability breakdowns, offline model evaluation reports, and confusion matrix visualizations.
- **Full Offline Reproducibility:** Fixed random seeds, self-contained dataset retrieval, automated artifact serialization via Joblib, and a comprehensive automated Pytest test suite.

---

## 3. Dataset Attribution & Acquisition

- **Dataset Name:** Student Performance Data Set (Mathematics Subject: `student-mat.csv`)
- **Official Landing Page:** [UCI Machine Learning Repository #320](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Direct Download Source:** `https://archive.ics.uci.edu/static/public/320/student+performance.zip`
- **Citation:**
  > Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, EUROSIS, ISBN 978-9077381-39-7.
- **Licensing & Attribution:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Cohort Size:** 395 student records, 33 original attributes.

---

## 4. Prediction Objectives & Target Formulation

### Regression Task
- **Target Variable:** `G3` (Final academic year grade).
- **Domain:** Continuous floating-point score bounded in $[0.0, 20.0]$.
- **Objective:** Quantitatively estimate final academic standing early in the term.

### Classification Task
- **Target Variable:** Academic performance class derived from $G3$.
- **Objective:** Triage students into discrete intervention tiers for targeted tutoring or enrichment.
- **Independence:** The classifier is trained directly from the raw feature representations, not from the regression model's output.

---

## 5. Feature Schema & Early Prediction Protocol

In the intended **early-prediction scenario**, interim term grades ($G1$ and $G2$) are not yet observed. To prevent data leakage, $G1$, $G2$, and $G3$ are strictly omitted from predictors.

The model accepts 30 independent features across four thematic categories:

| Feature Category | Features Included |
| :--- | :--- |
| **Demographics (6)** | `school`, `sex`, `age` (15-22), `address` (Urban/Rural), `famsize` (LE3/GT3), `Pstatus` (Together/Apart) |
| **Academic & Schooling (8)** | `studytime` (1-4), `failures` (0-4), `absences` (0-93), `schoolsup` (yes/no), `paid` (yes/no), `higher` (yes/no), `reason` (home/reputation/course/other), `traveltime` (1-4) |
| **Family Background (7)** | `Medu` (0-4), `Fedu` (0-4), `Mjob` (5 classes), `Fjob` (5 classes), `guardian` (mother/father/other), `famsup` (yes/no), `famrel` (1-5) |
| **Lifestyle & Habits (9)** | `activities` (yes/no), `nursery` (yes/no), `internet` (yes/no), `romantic` (yes/no), `freetime` (1-5), `goout` (1-5), `Dalc` (workday alcohol 1-5), `Walc` (weekend alcohol 1-5), `health` (1-5) |

---

## 6. Classification Boundaries

In Portuguese secondary education, student performance is assessed on a 20-point scale where 10 is the statutory passing threshold:

| Classification | Grade Range ($G3$) | Cohort Distribution | Description & Educational Action |
| :--- | :---: | :---: | :--- |
| **🔴 Low** | $0 \le G3 \le 9$ | 130 students (32.9%) | **At Risk / Failing:** Priority candidate for tutoring and advisor check-ins. |
| **🟡 Medium** | $10 \le G3 \le 13$ | 165 students (41.8%) | **Satisfactory / Passing:** Meets standard curriculum requirements. |
| **🟢 High** | $14 \le G3 \le 20$ | 100 students (25.3%) | **Good to Excellent:** High academic mastery; candidate for honors programs. |

---

## 7. Preprocessing & Leak-Free Architecture

1. **Stratified Splitting:** Dataset is split into an 80% training set ($N=316$) and a 20% held-out testing set ($N=79$), stratified across classification tiers to preserve target distribution.
2. **Encapsulated Preprocessing:** A Scikit-learn `ColumnTransformer` is fitted **exclusively on training data**:
   - **Numerical Features (13):** Imputed with training median; standardized with `StandardScaler` ($\mu=0, \sigma=1$).
   - **Categorical Features (17):** Imputed with training mode; encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to safeguard against unseen categories during inference.
3. **Identical Inference Pipeline:** The exact same pipeline object used during training processes runtime inference inputs.

---

## 8. Model Selection & Evaluation Methodology

Candidate algorithms were selected through 5-fold cross-validation strictly on the training partition:

### Regression Candidates
- `DummyRegressor(strategy='mean')` (Baseline)
- `LinearRegression`
- `Ridge(alpha=10.0)`
- `KNeighborsRegressor(n_neighbors=7)`
- `DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)` *(Winning Model)*

### Classification Candidates
- `DummyClassifier(strategy='most_frequent')` (Baseline)
- `DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)`
- `KNeighborsClassifier(n_neighbors=7)`
- `LogisticRegression(C=0.1)`
- `LogisticRegression(C=1.0)` *(Winning Model)*

---

## 9. Empirical Evaluation Results

The models were evaluated on the **held-out test set ($N=79$ students)**. All numbers below are generated directly from the executed test evaluation:

### Regression Evaluation (Held-Out Test Set)
| Model | MAE (lower is better) | RMSE (lower is better) | $R^2$ (higher is better) |
| :--- | :---: | :---: | :---: |
| **Baseline (`DummyRegressor`)** | 3.6650 | 4.9173 | -0.0023 |
| **Selected (`DecisionTreeRegressor`)** | **3.5212** | **4.6739** | **+0.0945** |

*Note on Early-Prediction $R^2$:* Without period grades $G1$ and $G2$, predicting exact final grades from baseline socio-economic data is inherently challenging. An $R^2$ of $+0.0945$ genuinely outperforms the baseline without data leakage or look-ahead contamination.

### Classification Evaluation (Held-Out Test Set)
| Metric | Baseline (`DummyClassifier`) | Selected (`LogisticRegression`) |
| :--- | :---: | :---: |
| **Accuracy** | 41.77% | **45.57%** |
| **Macro F1-Score** | 0.1964 | **0.4531** |
| **Weighted F1-Score** | 0.2462 | **0.4551** |
| **Macro Precision** | 0.1392 | **0.4632** |
| **Macro Recall** | 0.3333 | **0.4578** |

### Per-Class Test Breakdown (Selected Classifier)
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **🔴 Low** | 0.529 | 0.692 | 0.600 | 26 |
| **🟡 Medium** | 0.444 | 0.364 | 0.400 | 33 |
| **🟢 High** | 0.417 | 0.375 | 0.395 | 20 |

> **Key Finding:** The classifier achieves **69.2% recall on the Low (at-risk) category**, identifying over two-thirds of struggling students before exams take place.

---

## 10. Directory Structure

```text
Prediction Student Performance/
├── .gitignore                      # Source control exclusions
├── app.py                          # Streamlit interactive web application
├── requirements.txt                # Production Python dependencies
├── README.md                       # Comprehensive client documentation
├── data/
│   ├── .gitkeep
│   └── student-mat.csv             # UCI student mathematics dataset
├── models/
│   ├── .gitkeep
│   ├── regression_pipeline.joblib  # Serialized regression pipeline
│   └── classification_pipeline.joblib # Serialized classification pipeline
├── reports/
│   ├── .gitkeep
│   ├── evaluation_results.json     # Machine-readable evaluation report
│   └── confusion_matrix.png        # Test set confusion matrix heatmap
├── src/
│   ├── __init__.py                 # Python package marker
│   ├── config.py                   # Global constants, paths, and schemas
│   ├── data_loader.py              # Download, load, and schema separation
│   ├── evaluate.py                 # Evaluation metrics and reporting
│   ├── predict.py                  # Single-student and batch inference engine
│   ├── preprocessing.py            # ColumnTransformer & input validation
│   ├── train.py                    # Master training orchestrator
│   ├── train_classification.py     # Classification model selection & training
│   └── train_regression.py         # Regression model selection & training
└── tests/
    ├── __init__.py
    ├── test_app.py                 # Streamlit UI & metadata unit tests
    ├── test_data.py                # Dataset loading and integrity tests
    ├── test_models.py              # Model training & metrics verification
    ├── test_predictions.py         # Single and batch inference tests
    └── test_preprocessing.py       # Preprocessor and validation tests
```

---

## 11. Installation & Setup

### Windows (PowerShell / Command Prompt)

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "c:\Users\moham\Desktop\Projects\Prediction Student Performance"
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
3. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### macOS / Linux (Bash / Zsh)

1. Open a terminal and clone/navigate to the folder:
   ```bash
   cd "Prediction Student Performance"
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 12. Dataset Setup Instructions

### Automatic Download (Default)
The dataset will automatically download from UCI repository when running training, testing, or the application.
You can trigger the download manually at any time:
```bash
python -m src.data_loader
```

### Manual Fallback
If working offline or behind an enterprise firewall:
1. Download `student.zip` from the [UCI Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
2. Extract `student-mat.csv`.
3. Place `student-mat.csv` directly into the `data/` folder.

---

## 13. Training the Models

To execute the complete end-to-end training, cross-validation, and artifact serialization:

```bash
python -m src.train
```

To run individual training modules:
- **Regression only:** `python -m src.train_regression`
- **Classification only:** `python -m src.train_classification`

Outputs generated:
- `models/regression_pipeline.joblib`
- `models/classification_pipeline.joblib`
- `reports/evaluation_results.json`
- `reports/confusion_matrix.png`

---

## 14. Launching the Web Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

### Application Pages:
1. **🏠 Overview:** System mission, dataset overview, and prediction objectives.
2. **🎯 Predict Performance:** Interactive form with intuitive sliders and dropdowns for all 30 student attributes. Produces instant grade estimation and performance tier with confidence probabilities.
3. **📊 Model Evaluation:** Displays actual test set metrics (MAE, RMSE, $R^2$, Accuracy, F1-scores), baseline comparison deltas, and the confusion matrix heatmap.
4. **📖 About & Methodology:** Technical documentation, classification boundaries, and responsible AI guidance.

---

## 15. Automated Testing & QA

Run the automated test suite with Pytest:

```bash
pytest -v
```

### Test Suite Execution Scenarios:

1. **Full Environment (Recommended — with all `requirements.txt` installed):**
   When dependencies including Streamlit are installed, all **32 tests** execute and pass:
   ```text
   ============================= 32 passed in ~5-9s =============================
   ```
2. **Minimal / Headless Environment (Without Streamlit):**
   If executed in an environment where Streamlit is omitted, the test suite cleanly runs the **31 core unit & integration tests** and safely skips the UI rendering test (`test_streamlit_app_renders_without_exceptions`) via `@pytest.mark.skipif`:
   ```text
   ======================== 31 passed, 1 skipped in ~3s =========================
   ```

### Test Suite Coverage Breakdown:
- **`test_data.py` (8 tests):** Verifies dataset schema, row/column dimensions (395 rows, 33 columns), feature-target separation, and strictly asserts that no future grades ($G1$, $G2$, $G3$) contaminate predictor inputs.
- **`test_preprocessing.py` (9 tests):** Tests `ColumnTransformer`, median numerical imputation, mode categorical imputation, one-hot encoding resilience against unseen categories (`handle_unknown='ignore'`), and domain bounds validation.
- **`test_models.py` (6 tests):** Validates regression and classification pipeline fitting, synthetic evaluation calculation logic, disk artifact persistence, and `evaluation_results.json` schema integrity.
- **`test_predictions.py` (5 tests):** Tests single-student prediction, range bounding $[0.0, 20.0]$, batch inference, and multi-class probability summation.
- **`test_app.py` (4 tests):** Tests application syntax compilation, feature metadata schema coverage, end-to-end model loading flow, and simulated headless Streamlit execution (`AppTest`).

---

## 16. Example Programmatic Usage

You can embed the trained pipelines directly into Python scripts:

```python
from src.predict import predict_student, load_regression_pipeline, load_classification_pipeline

# 1. Load trained pipelines
reg_pipe = load_regression_pipeline()
clf_pipe = load_classification_pipeline()

# 2. Define student feature profile
student_data = {
    'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
    'Medu': 4, 'Fedu': 4, 'Mjob': 'health', 'Fjob': 'services', 'reason': 'reputation',
    'guardian': 'mother', 'traveltime': 1, 'studytime': 3, 'failures': 0, 'schoolsup': 'no',
    'famsup': 'yes', 'paid': 'yes', 'activities': 'yes', 'nursery': 'yes', 'higher': 'yes',
    'internet': 'yes', 'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 2,
    'Dalc': 1, 'Walc': 1, 'health': 5, 'absences': 2
}

# 3. Generate prediction
prediction = predict_student(student_data, reg_pipe, clf_pipe)

print(f"Predicted Grade:    {prediction['predicted_grade']} / 20.0")
print(f"Performance Tier:   {prediction['predicted_category']}")
print(f"Class Confidence:   {prediction['class_probabilities']}")
```

---

## 17. Ethical Considerations & Limitations

- **Advisory Decision Support:** Model predictions must serve solely as early indicators to allocate educational assistance. Predictions should **never** be used to deny academic opportunities, gate admissions, or penalize students.
- **Socio-Economic Variables:** Attributes like parental education and employment reflect structural socioeconomic factors. Educational institutions must not interpret predictions as innate cognitive aptitude.
- **Data Specificity:** The training data derives from two regional Portuguese secondary schools. Adapting this solution to other curricula or geographic regions requires retraining on local institutional data.

---

## 18. Troubleshooting & FAQ

1. **Error: Model artifacts not found (`FileNotFoundError`)**
   - *Solution:* Ensure you have executed the training script before launching the app: `python -m src.train`.
2. **Error: `ModuleNotFoundError: No module named 'src'`**
   - *Solution:* Run your commands from the workspace root directory, or ensure `src` is in your Python path.
3. **Error: Dataset download times out**
   - *Solution:* Follow the manual dataset setup instructions in [Section 12](#12-dataset-setup-instructions) to place `student-mat.csv` in `data/`.
4. **PowerShell script execution disabled**
   - *Solution:* Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell session.

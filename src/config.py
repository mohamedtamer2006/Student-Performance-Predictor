"""Configuration module for the Student Performance Prediction and Classification System.

Defines reproducible constants, directory paths, schema definitions,
feature categories, target boundaries, and model hyperparameters.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base project directory (workspace root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Core subdirectories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset paths and remote source
DATA_FILE = DATA_DIR / "student-mat.csv"
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
UCI_LANDING_PAGE = "https://archive.ics.uci.edu/dataset/320/student+performance"

# Saved artifact paths
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_pipeline.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_pipeline.joblib"
EVALUATION_RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCES_PATH = REPORTS_DIR / "feature_importances.png"

# Reproducibility settings
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Targets and early-prediction exclusion rules
REGRESSION_TARGET = "G3"
FORBIDDEN_FEATURES = ["G1", "G2", "G3"]

# Classification Grade Boundaries (Portuguese Secondary Grading Scale 0 - 20)
# Low: 0 - 9 (Fail / Insufficient)
# Medium: 10 - 13 (Sufficient / Satisfactory)
# High: 14 - 20 (Good to Excellent)
GRADE_BINS = [-0.1, 9.5, 13.5, 20.0]
CLASS_LABELS = ["Low", "Medium", "High"]
CLASS_DESCRIPTIONS = {
    "Low": "Grade 0–9 (Needs Support / At Risk of Failing)",
    "Medium": "Grade 10–13 (Satisfactory / Passing)",
    "High": "Grade 14–20 (Good / High Academic Achievement)"
}

# Feature definitions for early prediction
CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]

NUMERICAL_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Allowed categorical values for validation & UI forms
CATEGORICAL_OPTIONS: Dict[str, List[str]] = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["no", "yes"],
    "famsup": ["no", "yes"],
    "paid": ["no", "yes"],
    "activities": ["no", "yes"],
    "nursery": ["no", "yes"],
    "higher": ["no", "yes"],
    "internet": ["no", "yes"],
    "romantic": ["no", "yes"]
}

# Friendly descriptions for UI
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "school": {"label": "School", "help": "Student's school: GP (Gabriel Pereira) or MS (Mousinho da Silveira)"},
    "sex": {"label": "Sex", "help": "Student's sex: F (Female) or M (Male)"},
    "age": {"label": "Age", "min": 15, "max": 22, "default": 17, "help": "Student age in years (15 to 22)"},
    "address": {"label": "Address Type", "help": "Home address type: U (Urban) or R (Rural)"},
    "famsize": {"label": "Family Size", "help": "Family size: LE3 (<= 3) or GT3 (> 3)"},
    "Pstatus": {"label": "Parent Cohabitation Status", "help": "T (Living together) or A (Apart)"},
    "Medu": {"label": "Mother's Education", "min": 0, "max": 4, "default": 2, "help": "0: None, 1: 4th grade, 2: 5th-9th grade, 3: Secondary, 4: Higher education"},
    "Fedu": {"label": "Father's Education", "min": 0, "max": 4, "default": 2, "help": "0: None, 1: 4th grade, 2: 5th-9th grade, 3: Secondary, 4: Higher education"},
    "Mjob": {"label": "Mother's Job", "help": "Mother's occupation category"},
    "Fjob": {"label": "Father's Job", "help": "Father's occupation category"},
    "reason": {"label": "Reason to Choose School", "help": "Close to home, school reputation, course preference, or other"},
    "guardian": {"label": "Student Guardian", "help": "Mother, father, or other"},
    "traveltime": {"label": "Home to School Travel Time", "min": 1, "max": 4, "default": 1, "help": "1: <15 min, 2: 15-30 min, 3: 30 min-1 hr, 4: >1 hr"},
    "studytime": {"label": "Weekly Study Time", "min": 1, "max": 4, "default": 2, "help": "1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs"},
    "failures": {"label": "Past Class Failures", "min": 0, "max": 4, "default": 0, "help": "Number of past class failures (0 to 4)"},
    "schoolsup": {"label": "Extra Educational Support", "help": "Extra educational school support (yes/no)"},
    "famsup": {"label": "Family Educational Support", "help": "Family educational support (yes/no)"},
    "paid": {"label": "Extra Paid Classes", "help": "Extra paid classes within the course subject (yes/no)"},
    "activities": {"label": "Extra-curricular Activities", "help": "Extra-curricular activities (yes/no)"},
    "nursery": {"label": "Attended Nursery School", "help": "Attended nursery school (yes/no)"},
    "higher": {"label": "Wants Higher Education", "help": "Wishes to take higher education (yes/no)"},
    "internet": {"label": "Internet Access at Home", "help": "Internet access at home (yes/no)"},
    "romantic": {"label": "In a Romantic Relationship", "help": "With a romantic relationship (yes/no)"},
    "famrel": {"label": "Family Relationship Quality", "min": 1, "max": 5, "default": 4, "help": "1 (Very bad) to 5 (Excellent)"},
    "freetime": {"label": "Free Time After School", "min": 1, "max": 5, "default": 3, "help": "1 (Very low) to 5 (Very high)"},
    "goout": {"label": "Going Out with Friends", "min": 1, "max": 5, "default": 3, "help": "1 (Very low) to 5 (Very high)"},
    "Dalc": {"label": "Workday Alcohol Consumption", "min": 1, "max": 5, "default": 1, "help": "1 (Very low) to 5 (Very high)"},
    "Walc": {"label": "Weekend Alcohol Consumption", "min": 1, "max": 5, "default": 2, "help": "1 (Very low) to 5 (Very high)"},
    "health": {"label": "Current Health Status", "min": 1, "max": 5, "default": 4, "help": "1 (Very bad) to 5 (Very good)"},
    "absences": {"label": "Number of School Absences", "min": 0, "max": 93, "default": 4, "help": "Number of school absences (0 to 93)"}
}

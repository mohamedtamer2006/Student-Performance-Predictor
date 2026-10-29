"""Tests for the Streamlit application structure, execution, and helper routines.
"""

import py_compile
from pathlib import Path
import pytest

try:
    import streamlit
    from streamlit.testing.v1 import AppTest
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.config import ALL_FEATURE_COLUMNS, BASE_DIR, FEATURE_METADATA
from src.evaluate import load_evaluation_results
from src.predict import (
    load_classification_pipeline,
    load_regression_pipeline,
    predict_student,
)


def test_app_py_syntax():
    """Verify that app.py compiles cleanly without any syntax errors."""
    app_path = BASE_DIR / "app.py"
    assert app_path.exists()
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None


def test_feature_metadata_covers_all_columns():
    """Verify every feature in ALL_FEATURE_COLUMNS has UI metadata defined."""
    for col in ALL_FEATURE_COLUMNS:
        assert col in FEATURE_METADATA, f"Missing metadata for feature '{col}'"
        assert "label" in FEATURE_METADATA[col]


def test_app_model_loading_and_predict_flow():
    """Verify the core inference logic that powers the Streamlit interface."""
    reg_pipe = load_regression_pipeline()
    clf_pipe = load_classification_pipeline()
    results_meta = load_evaluation_results()

    assert reg_pipe is not None
    assert clf_pipe is not None
    assert results_meta is not None

    test_student = {}
    for col in ALL_FEATURE_COLUMNS:
        meta = FEATURE_METADATA[col]
        if "default" in meta:
            test_student[col] = meta["default"]
        else:
            from src.config import CATEGORICAL_OPTIONS
            test_student[col] = CATEGORICAL_OPTIONS[col][0]

    prediction = predict_student(test_student, reg_pipeline=reg_pipe, clf_pipeline=clf_pipe)
    assert "predicted_grade" in prediction
    assert "predicted_category" in prediction
    assert 0.0 <= prediction["predicted_grade"] <= 20.0


@pytest.mark.skipif(not HAS_STREAMLIT, reason="Streamlit is not installed in the testing environment.")
def test_streamlit_app_renders_without_exceptions():
    """Verify that Streamlit AppTest executes app.py end-to-end without unhandled exceptions."""
    app_file = str(BASE_DIR / "app.py")
    at = AppTest.from_file(app_file, default_timeout=15)
    at.run()
    assert len(at.exception) == 0
    assert len(at.title) > 0

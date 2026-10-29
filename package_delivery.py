"""Packaging script for creating a pure, clean client deliverable ZIP archive.

Contains solely the project source code, dataset, trained models, reports,
tests, and documentation, strictly excluding any Git metadata or cache files.
"""

from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = BASE_DIR / "Student_Performance_Prediction_Clean.zip"

DELIVERY_FILES = [
    ".gitignore",
    "README.md",
    "app.py",
    "pytest.ini",
    "requirements.txt",
    "data/.gitkeep",
    "data/student-mat.csv",
    "models/.gitkeep",
    "models/classification_pipeline.joblib",
    "models/regression_pipeline.joblib",
    "reports/.gitkeep",
    "reports/confusion_matrix.png",
    "reports/evaluation_results.json",
    "src/__init__.py",
    "src/config.py",
    "src/data_loader.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/preprocessing.py",
    "src/train.py",
    "src/train_classification.py",
    "src/train_regression.py",
    "tests/__init__.py",
    "tests/test_app.py",
    "tests/test_data.py",
    "tests/test_models.py",
    "tests/test_predictions.py",
    "tests/test_preprocessing.py",
]


def build_clean_zip() -> Path:
    """Build the clean client delivery ZIP file."""
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    print(f"Creating clean delivery archive: {OUTPUT_ZIP.name}...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(DELIVERY_FILES):
            fpath = BASE_DIR / rel
            if fpath.exists():
                zf.write(fpath, rel)
                print(f"  [+] {rel}")
            else:
                print(f"  [!] Missing file: {rel}")

    file_size_kb = OUTPUT_ZIP.stat().st_size / 1024
    print(f"\nArchive successfully generated: {OUTPUT_ZIP.name}")
    print(f"Total files: {len(DELIVERY_FILES)} | Size: {file_size_kb:.2f} KB")
    return OUTPUT_ZIP


if __name__ == "__main__":
    build_clean_zip()

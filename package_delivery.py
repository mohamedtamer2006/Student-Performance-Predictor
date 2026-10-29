"""Packaging script for clean client deliverable ZIP.

Creates a distribution archive containing code, data, models, reports, tests,
and documentation, while strictly excluding Git metadata, cache directories,
and temporary files.
"""

import os
from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = BASE_DIR / "Student_Performance_Prediction_Clean.zip"

# Exact directories and root files to include
INCLUDE_ROOT_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "pytest.ini",
]

INCLUDE_DIRS = [
    "data",
    "models",
    "reports",
    "src",
    "tests",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    ".DS_Store",
    "Thumbs.db",
]


def is_excluded(path_str: str) -> bool:
    """Check if file or directory path contains excluded patterns."""
    parts = Path(path_str).parts
    for part in parts:
        for pattern in EXCLUDE_PATTERNS:
            if pattern in part:
                return True
        if part.endswith(".pyc") or part.endswith(".pyo"):
            return True
    return False


def build_clean_zip() -> Path:
    """Build the clean client delivery ZIP file."""
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    print(f"Creating clean delivery archive at: {OUTPUT_ZIP.name}...")
    added_count = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add root files
        for fname in INCLUDE_ROOT_FILES:
            fpath = BASE_DIR / fname
            if fpath.exists():
                arcname = f"Student_Performance_Prediction/{fname}"
                zf.write(fpath, arcname)
                added_count += 1
                print(f"  + {arcname}")

        # 2. Add directories
        for dname in INCLUDE_DIRS:
            dir_path = BASE_DIR / dname
            if not dir_path.exists():
                continue
            for root, dirs, files in os.walk(dir_path):
                # Filter out excluded subdirs in-place
                dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]
                for file in files:
                    file_path = Path(root) / file
                    if is_excluded(str(file_path)):
                        continue
                    rel_path = file_path.relative_to(BASE_DIR)
                    arcname = f"Student_Performance_Prediction/{rel_path.as_posix()}"
                    zf.write(file_path, arcname)
                    added_count += 1
                    print(f"  + {arcname}")

    file_size_kb = OUTPUT_ZIP.stat().st_size / 1024
    print(f"\nSuccessfully created clean archive: {OUTPUT_ZIP.name}")
    print(f"Total files: {added_count} | Size: {file_size_kb:.1f} KB")
    return OUTPUT_ZIP


if __name__ == "__main__":
    build_clean_zip()

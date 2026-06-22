from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "Makefile",
    "data/raw/stocks.csv",
    "data/processed/train_preprocessed.csv",
    "data/processed/test_preprocessed.csv",
    "data/processed/preprocessing_params.json",
    "models/best_model.joblib",
    "models/training_metadata.json",
    "src/__init__.py",
    "src/eda.py",
    "src/preprocessing.py",
    "src/training.py",
    "src/api.py",
    "tests/test_preprocessing.py",
    "tests/test_pipeline.py",
]


REQUIRED_README_SECTIONS = [
    "## 1. Problématique d'affaires",
    "## 2. Présentation du jeu de données",
    "## 3. Analyse exploratoire des données",
    "## 4. Prétraitement des données",
    "## 5. Sélection et entraînement du modèle",
    "## 6. API FastAPI et packaging du modèle",
    "## 7. Tests automatisés",
    "## 8. Docker",
    "## 9. Structure du projet",
    "## 10. Installation",
    "## 11. Exécution du projet",
    "## 12. Architecture Cloud",
]


REQUIRED_REQUIREMENTS = [
    "fastapi",
    "uvicorn",
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "pydantic",
    "pytest",
]


def check_required_paths() -> list[str]:
    errors = []

    for relative_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing required path: {relative_path}")

    return errors


def check_readme_sections() -> list[str]:
    errors = []
    readme_path = PROJECT_ROOT / "README.md"

    if not readme_path.exists():
        return ["README.md is missing"]

    readme_content = readme_path.read_text(encoding="utf-8")

    for section in REQUIRED_README_SECTIONS:
        if section not in readme_content:
            errors.append(f"Missing README section: {section}")

    return errors


def check_requirements() -> list[str]:
    errors = []
    requirements_path = PROJECT_ROOT / "requirements.txt"

    if not requirements_path.exists():
        return ["requirements.txt is missing"]

    try:
        requirements_content = requirements_path.read_text(encoding="utf-8-sig").lower()
    except UnicodeDecodeError:
        requirements_content = requirements_path.read_text(encoding="utf-16").lower()


    for dependency in REQUIRED_REQUIREMENTS:
        if dependency.lower() not in requirements_content:
            errors.append(f"Missing dependency in requirements.txt: {dependency}")

    return errors


def check_dockerfile() -> list[str]:
    errors = []
    dockerfile_path = PROJECT_ROOT / "Dockerfile"

    if not dockerfile_path.exists():
        return ["Dockerfile is missing"]

    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")

    if "EXPOSE 8000" not in dockerfile_content:
        errors.append("Dockerfile should expose port 8000")

    if "uvicorn" not in dockerfile_content:
        errors.append("Dockerfile should start the FastAPI app with uvicorn")

    if "src.api:app" not in dockerfile_content:
        errors.append("Dockerfile should reference src.api:app")

    return errors


def check_api_import() -> list[str]:
    errors = []

    result = subprocess.run(
        [sys.executable, "-c", "from src.api import app; print(app.title)"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        errors.append("FastAPI app cannot be imported from src.api")

    return errors


def run_pytest() -> list[str]:
    errors = []

    result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        errors.append("pytest failed")

    return errors


def main() -> int:
    print("=== Project Validation ===")

    checks = {
        "Required files and folders": check_required_paths,
        "README sections": check_readme_sections,
        "requirements.txt dependencies": check_requirements,
        "Dockerfile": check_dockerfile,
        "FastAPI import": check_api_import,
        "pytest": run_pytest,
    }

    all_errors = []

    for check_name, check_function in checks.items():
        print(f"\nChecking: {check_name}")
        errors = check_function()

        if errors:
            print("FAILED")
            for error in errors:
                print(f"  - {error}")
            all_errors.extend(errors)
        else:
            print("PASSED")

    print("\n=== Validation Summary ===")

    if all_errors:
        print(f"Project validation failed with {len(all_errors)} issue(s).")
        return 1

    print("Project validation passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
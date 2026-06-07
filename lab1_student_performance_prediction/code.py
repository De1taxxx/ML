"""Train models for predicting online course completion.

The script performs the full laboratory pipeline:
1. loads a CSV dataset;
2. preprocesses numeric and categorical features;
3. trains several machine learning models;
4. compares metrics;
5. saves the best model and analytical reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
RANDOM_STATE = 42
TARGET_COLUMN = "passed"


NUMERIC_FEATURES = [
    "age",
    "previous_experience",
    "study_hours_per_week",
    "lectures_viewed",
    "assignments_submitted",
    "avg_quiz_score",
    "forum_posts",
    "deadline_misses",
    "days_since_last_activity",
    "practice_tests_completed",
    "video_completion_rate",
    "attendance_rate",
]

CATEGORICAL_FEATURES = ["course_level"]


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
}


def ensure_directories() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def save_metrics(metrics: list[dict]) -> None:
    metrics_df = pd.DataFrame(metrics).sort_values(by="f1_score", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    lines = ["Model comparison", "================", ""]
    for item in metrics_df.to_dict("records"):
        lines.append(f"{item['model']}:")
        lines.append(f"  Accuracy:  {item['accuracy']}")
        lines.append(f"  Precision: {item['precision']}")
        lines.append(f"  Recall:    {item['recall']}")
        lines.append(f"  F1-score:  {item['f1_score']}")
        lines.append(f"  ROC-AUC:   {item['roc_auc']}")
        lines.append("")
    (REPORTS_DIR / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def save_classification_report(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["failed", "passed"],
        digits=4,
    )
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")


def save_confusion_matrix(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["failed", "passed"])
    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def save_roc_curve(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()


def get_feature_names(best_pipeline: Pipeline) -> list[str]:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_names = categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + categorical_names


def save_feature_importance(best_pipeline: Pipeline) -> None:
    classifier = best_pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return

    feature_names = get_feature_names(best_pipeline)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    top_features = importance_df.head(10).sort_values(by="importance")
    plt.figure(figsize=(9, 5))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=200)
    plt.close()


def train_models() -> None:
    ensure_directories()
    df = load_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics = []
    trained_pipelines: dict[str, Pipeline] = {}

    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        metrics.append(evaluate_model(name, pipeline, X_test, y_test))

    save_metrics(metrics)

    best_metric = max(metrics, key=lambda item: item["roc_auc"])
    best_name = best_metric["model"]
    best_pipeline = trained_pipelines[best_name]

    joblib.dump(best_pipeline, MODELS_DIR / "student_performance_model.pkl")
    joblib.dump(NUMERIC_FEATURES + CATEGORICAL_FEATURES, MODELS_DIR / "feature_columns.pkl")

    save_classification_report(best_pipeline, X_test, y_test)
    save_confusion_matrix(best_pipeline, X_test, y_test)
    save_roc_curve(best_pipeline, X_test, y_test)
    save_feature_importance(best_pipeline)

    summary = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "best_model": best_name,
        "best_model_metrics": best_metric,
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("Training completed")
    print(f"Best model: {best_name}")
    print(best_metric)


if __name__ == "__main__":
    train_models()

"""Predict whether a student will complete an online course."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/student_performance_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict online course completion")
    parser.add_argument("--age", type=int, default=22)
    parser.add_argument("--course_level", type=str, default="intermediate", choices=["beginner", "intermediate", "advanced"])
    parser.add_argument("--previous_experience", type=int, default=1, choices=[0, 1])
    parser.add_argument("--study_hours_per_week", type=float, default=8.5)
    parser.add_argument("--lectures_viewed", type=int, default=45)
    parser.add_argument("--assignments_submitted", type=int, default=15)
    parser.add_argument("--avg_quiz_score", type=float, default=78.0)
    parser.add_argument("--forum_posts", type=int, default=5)
    parser.add_argument("--deadline_misses", type=int, default=1)
    parser.add_argument("--days_since_last_activity", type=int, default=4)
    parser.add_argument("--practice_tests_completed", type=int, default=6)
    parser.add_argument("--video_completion_rate", type=float, default=82.0)
    parser.add_argument("--attendance_rate", type=float, default=88.0)
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run code.py first.")

    args = parse_args()
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([vars(args)])
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    label = "passed" if prediction == 1 else "failed"
    print(f"Prediction: {label}")
    print(f"Course completion probability: {probability:.3f}")


if __name__ == "__main__":
    main()

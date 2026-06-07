"""Generate a synthetic dataset for student performance prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_SAMPLES = 1200


def generate_dataset(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 56, n_samples)
    course_level = rng.choice(
        ["beginner", "intermediate", "advanced"],
        size=n_samples,
        p=[0.42, 0.38, 0.20],
    )
    previous_experience = rng.binomial(1, 0.48, n_samples)

    study_hours_per_week = np.clip(rng.gamma(shape=2.6, scale=2.4, size=n_samples), 0.5, 20)
    lectures_viewed = np.clip((study_hours_per_week * 5 + rng.normal(12, 10, n_samples)).round(), 0, 80).astype(int)
    assignments_submitted = np.clip((study_hours_per_week * 1.3 + rng.normal(4, 4, n_samples)).round(), 0, 30).astype(int)
    avg_quiz_score = np.clip(
        45 + study_hours_per_week * 3.0 + previous_experience * 8 + rng.normal(0, 13, n_samples),
        0,
        100,
    ).round(1)
    forum_posts = np.clip(rng.poisson(lam=1.5 + study_hours_per_week / 4), 0, 25).astype(int)
    deadline_misses = np.clip((rng.poisson(lam=2.2, size=n_samples) - previous_experience + rng.normal(0, 1, n_samples)).round(), 0, 12).astype(int)
    days_since_last_activity = np.clip((42 - study_hours_per_week * 2.3 + rng.normal(0, 10, n_samples)).round(), 0, 60).astype(int)
    practice_tests_completed = np.clip((study_hours_per_week * 0.8 + rng.normal(1, 2, n_samples)).round(), 0, 15).astype(int)
    video_completion_rate = np.clip(35 + lectures_viewed * 0.7 + rng.normal(0, 18, n_samples), 0, 100).round(1)
    attendance_rate = np.clip(40 + study_hours_per_week * 3.2 - deadline_misses * 3 + rng.normal(0, 15, n_samples), 0, 100).round(1)

    level_penalty = np.select(
        [course_level == "beginner", course_level == "intermediate", course_level == "advanced"],
        [1.5, -0.5, -3.0],
    )

    latent_score = (
        0.05 * lectures_viewed
        + 0.22 * assignments_submitted
        + 0.07 * avg_quiz_score
        + 0.35 * study_hours_per_week
        + 0.04 * video_completion_rate
        + 0.03 * attendance_rate
        + 0.18 * practice_tests_completed
        + 0.12 * forum_posts
        + 1.7 * previous_experience
        - 0.55 * deadline_misses
        - 0.10 * days_since_last_activity
        + level_penalty
        + rng.normal(0, 4.2, n_samples)
    )

    threshold = np.quantile(latent_score, 0.43)
    passed = (latent_score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "course_level": course_level,
            "previous_experience": previous_experience,
            "study_hours_per_week": study_hours_per_week.round(1),
            "lectures_viewed": lectures_viewed,
            "assignments_submitted": assignments_submitted,
            "avg_quiz_score": avg_quiz_score,
            "forum_posts": forum_posts,
            "deadline_misses": deadline_misses,
            "days_since_last_activity": days_since_last_activity,
            "practice_tests_completed": practice_tests_completed,
            "video_completion_rate": video_completion_rate,
            "attendance_rate": attendance_rate,
            "passed": passed,
        }
    )
    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("data.csv", index=False)
    print(f"Saved data.csv with {len(dataset)} rows")
    print(dataset["passed"].value_counts(normalize=True).rename("share"))

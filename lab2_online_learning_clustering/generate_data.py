from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'data' / 'students_activity.csv'
RNG = np.random.default_rng(42)

# Five behavioral archetypes for an online course.
SEGMENTS = {
    'video_focused_students': {
        'p': 0.22, 'lectures': (91, 5), 'hours': (6.8, 0.8), 'assignments': (68, 6),
        'quiz': (72, 6), 'deadlines': (2.2, 0.8), 'forum': (2, 1), 'logins': (16, 3),
        'last': (3.5, 1.6), 'rewatch': (13, 3), 'messages': (1, 1), 'mobile': (62, 8), 'exam': (73, 7)
    },
    'practice_oriented_students': {
        'p': 0.24, 'lectures': (66, 5), 'hours': (5.9, 0.9), 'assignments': (93, 4),
        'quiz': (88, 5), 'deadlines': (0.8, 0.6), 'forum': (3, 1), 'logins': (18, 3),
        'last': (2.2, 1.1), 'rewatch': (4, 2), 'messages': (1, 1), 'mobile': (36, 8), 'exam': (87, 6)
    },
    'social_learners': {
        'p': 0.20, 'lectures': (73, 6), 'hours': (6.3, 0.9), 'assignments': (74, 6),
        'quiz': (76, 6), 'deadlines': (1.7, 0.8), 'forum': (17, 3), 'logins': (23, 3),
        'last': (1.8, 0.9), 'rewatch': (5, 2), 'messages': (8, 2), 'mobile': (47, 8), 'exam': (77, 6)
    },
    'at_risk_students': {
        'p': 0.22, 'lectures': (27, 7), 'hours': (1.7, 0.5), 'assignments': (25, 8),
        'quiz': (39, 9), 'deadlines': (8.6, 1.3), 'forum': (1, 1), 'logins': (5, 2),
        'last': (16.0, 3.5), 'rewatch': (2, 1), 'messages': (1, 1), 'mobile': (71, 10), 'exam': (38, 10)
    },
    'catch_up_students': {
        'p': 0.12, 'lectures': (58, 7), 'hours': (9.5, 1.0), 'assignments': (61, 8),
        'quiz': (63, 8), 'deadlines': (5.4, 1.0), 'forum': (6, 2), 'logins': (28, 2),
        'last': (1.2, 0.7), 'rewatch': (10, 2), 'messages': (5, 2), 'mobile': (54, 8), 'exam': (65, 8)
    },
}


def clipped_normal(mean, sd, n, lo, hi):
    return np.clip(RNG.normal(mean, sd, n), lo, hi)


def make_dataset(n=1000):
    labels = RNG.choice(list(SEGMENTS), size=n, p=[v['p'] for v in SEGMENTS.values()])
    rows = []
    for label in labels:
        cfg = SEGMENTS[label]
        lectures_watched_pct = clipped_normal(*cfg['lectures'], 1, 0, 100)[0]
        weekly_study_hours = clipped_normal(*cfg['hours'], 1, 0.2, 14)[0]
        assignments_completed_pct = clipped_normal(*cfg['assignments'], 1, 0, 100)[0]
        avg_quiz_score = clipped_normal(*cfg['quiz'], 1, 0, 100)[0]
        missed_deadlines = int(round(clipped_normal(*cfg['deadlines'], 1, 0, 14)[0]))
        forum_posts = int(round(clipped_normal(*cfg['forum'], 1, 0, 30)[0]))
        logins_per_month = int(round(clipped_normal(*cfg['logins'], 1, 1, 31)[0]))
        days_since_last_login = clipped_normal(*cfg['last'], 1, 0, 30)[0]
        video_rewatch_count = int(round(clipped_normal(*cfg['rewatch'], 1, 0, 25)[0]))
        messages_to_teacher = int(round(clipped_normal(*cfg['messages'], 1, 0, 15)[0]))
        mobile_activity_pct = clipped_normal(*cfg['mobile'], 1, 5, 95)[0]
        final_exam_score = clipped_normal(*cfg['exam'], 1, 0, 100)[0]
        progress_consistency = np.clip(
            0.32 * assignments_completed_pct + 0.22 * avg_quiz_score + 0.18 * lectures_watched_pct +
            0.9 * logins_per_month - 2.5 * missed_deadlines - 0.7 * days_since_last_login + RNG.normal(0, 3),
            0, 100,
        )
        passed = int((0.20 * lectures_watched_pct +
                      0.26 * assignments_completed_pct +
                      0.24 * avg_quiz_score +
                      0.20 * final_exam_score +
                      0.9 * weekly_study_hours +
                      0.4 * logins_per_month -
                      2.5 * missed_deadlines -
                      0.7 * days_since_last_login + RNG.normal(0, 5)) >= 65)
        risk_score = np.clip(
            100 - (0.25 * assignments_completed_pct + 0.18 * avg_quiz_score + 0.15 * lectures_watched_pct +
                   0.12 * final_exam_score + 1.0 * weekly_study_hours + 0.35 * logins_per_month -
                   2.6 * missed_deadlines - 0.9 * days_since_last_login),
            0, 100)
        rows.append({
            'lectures_watched_pct': round(lectures_watched_pct, 1),
            'weekly_study_hours': round(weekly_study_hours, 1),
            'assignments_completed_pct': round(assignments_completed_pct, 1),
            'avg_quiz_score': round(avg_quiz_score, 1),
            'missed_deadlines': missed_deadlines,
            'forum_posts': forum_posts,
            'logins_per_month': logins_per_month,
            'days_since_last_login': round(days_since_last_login, 1),
            'video_rewatch_count': video_rewatch_count,
            'messages_to_teacher': messages_to_teacher,
            'mobile_activity_pct': round(mobile_activity_pct, 1),
            'progress_consistency': round(progress_consistency, 1),
            'final_exam_score': round(final_exam_score, 1),
            'learning_style': label,
            'risk_score': round(risk_score, 1),
            'passed': passed,
        })
    df = pd.DataFrame(rows)
    # A small amount of label noise makes the task closer to real educational data.
    labels_list = list(SEGMENTS)
    noise_idx = RNG.choice(df.index, size=max(1, int(len(df) * 0.04)), replace=False)
    for idx in noise_idx:
        current = df.at[idx, 'learning_style']
        alternatives = [x for x in labels_list if x != current]
        df.at[idx, 'learning_style'] = RNG.choice(alternatives)
    return df


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(OUT, index=False)
    print(f'Saved {len(df)} rows to {OUT}')
    print(df.head())

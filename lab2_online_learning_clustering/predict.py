from pathlib import Path
import json

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'


def load_artifacts():
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    kmeans = joblib.load(MODELS_DIR / 'kmeans_model.pkl')
    classifier = joblib.load(MODELS_DIR / 'learning_style_classifier.pkl')
    encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')
    feature_columns = joblib.load(MODELS_DIR / 'feature_columns.pkl')
    cluster_names_path = REPORTS_DIR / 'cluster_names.json'
    cluster_names = {}
    if cluster_names_path.exists():
        cluster_names = json.loads(cluster_names_path.read_text(encoding='utf-8'))
    return scaler, kmeans, classifier, encoder, feature_columns, cluster_names


def predict_student_profile(student_features: dict):
    scaler, kmeans, classifier, encoder, feature_columns, cluster_names = load_artifacts()
    row = pd.DataFrame([student_features], columns=feature_columns)
    scaled = scaler.transform(row)
    cluster = int(kmeans.predict(scaled)[0])
    style_code = int(classifier.predict(row)[0])
    style = encoder.inverse_transform([style_code])[0]
    probabilities = classifier.predict_proba(row)[0]
    confidence = float(probabilities.max())
    return {
        'cluster_label': cluster,
        'cluster_interpretation': cluster_names.get(str(cluster), 'unknown_cluster'),
        'predicted_learning_style': style,
        'confidence': round(confidence, 4),
    }


if __name__ == '__main__':
    example_student = {
        'lectures_watched_pct': 64.0,
        'weekly_study_hours': 9.2,
        'assignments_completed_pct': 58.0,
        'avg_quiz_score': 61.0,
        'missed_deadlines': 5,
        'forum_posts': 6,
        'logins_per_month': 27,
        'days_since_last_login': 1.0,
        'video_rewatch_count': 9,
        'messages_to_teacher': 5,
        'mobile_activity_pct': 55.0,
        'progress_consistency': 53.0,
        'final_exam_score': 65.0,
    }
    result = predict_student_profile(example_student)
    print('Результат анализа студента:')
    for key, value in result.items():
        print(f'{key}: {value}')

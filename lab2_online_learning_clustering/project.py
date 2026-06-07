from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'students_activity.csv'
REPORTS_DIR = BASE_DIR / 'reports'
MODELS_DIR = BASE_DIR / 'models'

FEATURE_COLUMNS = [
    'lectures_watched_pct',
    'weekly_study_hours',
    'assignments_completed_pct',
    'avg_quiz_score',
    'missed_deadlines',
    'forum_posts',
    'logins_per_month',
    'days_since_last_login',
    'video_rewatch_count',
    'messages_to_teacher',
    'mobile_activity_pct',
    'progress_consistency',
    'final_exam_score',
]

TARGET_COLUMN = 'learning_style'


def create_project_structure() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    return df.dropna().reset_index(drop=True)


def prepare_features(df: pd.DataFrame):
    features_df = df[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / 'feature_columns.pkl')
    return features_df, x_scaled, scaler


def plot_correlation_matrix(features_df: pd.DataFrame) -> None:
    corr = features_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Correlation matrix of student activity features')
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'correlation_matrix.png', dpi=200)
    plt.close(fig)


def perform_clustering(df: pd.DataFrame, features_df: pd.DataFrame, x_scaled: np.ndarray):
    scores = []
    best_score = -1
    best_k = 2
    selected_k = 5
    selected_score = None
    selected_labels = None
    selected_model = None

    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = model.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels, sample_size=min(500, len(x_scaled)), random_state=42)
        scores.append({'method': 'KMeans', 'parameter': f'k={k}', 'silhouette_score': round(float(score), 4)})
        if score > best_score:
            best_score = score
            best_k = k
        if k == selected_k:
            selected_score = score
            selected_labels = labels
            selected_model = model

    dbscan_params = [(1.0, 8), (1.2, 10)]
    for eps, min_samples in dbscan_params:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)
        unique = set(labels)
        clusters_without_noise = [x for x in unique if x != -1]
        if len(clusters_without_noise) >= 2:
            mask = labels != -1
            score = silhouette_score(x_scaled[mask], labels[mask])
            score_value = round(float(score), 4)
        else:
            score_value = None
        scores.append({'method': 'DBSCAN', 'parameter': f'eps={eps}, min_samples={min_samples}', 'silhouette_score': score_value})

    for n_clusters in range(2, 5):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels, sample_size=min(500, len(x_scaled)), random_state=42)
        scores.append({'method': 'Agglomerative', 'parameter': f'n_clusters={n_clusters}', 'silhouette_score': round(float(score), 4)})

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(REPORTS_DIR / 'silhouette_scores.csv', index=False)
    joblib.dump(selected_model, MODELS_DIR / 'kmeans_model.pkl')

    clustered = df.copy()
    clustered['cluster_label'] = selected_labels
    clustered.to_csv(REPORTS_DIR / 'clustered_students.csv', index=False)

    summary = clustered.groupby('cluster_label')[FEATURE_COLUMNS + ['risk_score', 'passed']].mean().round(2)
    summary['student_count'] = clustered.groupby('cluster_label').size()
    summary = summary.reset_index()
    summary.to_csv(REPORTS_DIR / 'cluster_summary.csv', index=False)

    cluster_names = {}
    for _, row in summary.iterrows():
        label = int(row['cluster_label'])
        risk = row['risk_score']
        lectures = row['lectures_watched_pct']
        deadlines = row['missed_deadlines']
        hours = row['weekly_study_hours']
        if risk >= 75:
            name = 'at_risk_students'
        elif lectures >= 80 and deadlines <= 2:
            name = 'active_students'
        elif hours >= 7 and deadlines >= 3:
            name = 'catch_up_students'
        elif lectures >= 60:
            name = 'stable_students'
        else:
            name = 'passive_students'
        cluster_names[label] = name

    with open(REPORTS_DIR / 'cluster_names.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_names, f, ensure_ascii=False, indent=2)

    plot_silhouette(scores_df)
    plot_pca(x_scaled, selected_labels)
    # t-SNE is useful, but PCA is sufficient for the current report and runs faster.

    return selected_labels, selected_k, float(selected_score), summary, scores_df, cluster_names


def plot_silhouette(scores_df: pd.DataFrame) -> None:
    kmeans = scores_df[scores_df['method'] == 'KMeans'].copy()
    kmeans['k'] = kmeans['parameter'].str.extract(r'k=(\d+)').astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kmeans['k'], kmeans['silhouette_score'], marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('KMeans clustering quality')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'silhouette_scores.png', dpi=200)
    plt.close(fig)


def plot_pca(x_scaled: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, alpha=0.75)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Student clusters, PCA projection')
    fig.colorbar(scatter, ax=ax, label='cluster_label')
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'pca_clusters.png', dpi=200)
    plt.close(fig)


def plot_tsne(x_scaled: np.ndarray, labels: np.ndarray) -> None:
    # For speed and stability t-SNE is calculated on a deterministic sample.
    sample_size = min(250, len(x_scaled))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x_scaled), size=sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=25, init='pca', learning_rate='auto', max_iter=350)
    coords = tsne.fit_transform(x_scaled[sample_idx])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels[sample_idx], s=18, alpha=0.75)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Student clusters, t-SNE projection')
    fig.colorbar(scatter, ax=ax, label='cluster_label')
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'tsne_clusters.png', dpi=200)
    plt.close(fig)


def perform_multiclass_classification(features_df: pd.DataFrame, labels: pd.Series):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        features_df,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results.append({
            'model': name,
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
            'recall_macro': round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
            'f1_macro': round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / 'classification_metrics.csv', index=False)
    best_name = results_df.sort_values('f1_macro', ascending=False).iloc[0]['model']
    best_model = fitted_models[best_name]
    y_pred_best = best_model.predict(x_test)

    with open(REPORTS_DIR / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(classification_report(y_test, y_pred_best, target_names=encoder.classes_, zero_division=0))

    plot_confusion_matrix(y_test, y_pred_best, encoder.classes_)

    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': best_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
        plot_feature_importance(importance.head(10))

    joblib.dump(best_model, MODELS_DIR / 'learning_style_classifier.pkl')
    joblib.dump(encoder, MODELS_DIR / 'label_encoder.pkl')

    return results_df, best_name


def plot_confusion_matrix(y_true, y_pred, class_names) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'confusion_matrix.png', dpi=200)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    ordered = importance.sort_values('importance')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered['feature'], ordered['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance for learning style classification')
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'feature_importance.png', dpi=200)
    plt.close(fig)


def create_final_report(df, best_k, best_score, cluster_summary, scores_df, classification_metrics, best_classifier, cluster_names):
    summary = {
        'dataset_rows': int(len(df)),
        'dataset_columns': int(len(df.columns)),
        'selected_kmeans_k': int(best_k),
        'selected_kmeans_silhouette': round(float(best_score), 4),
        'best_classifier': str(best_classifier),
        'class_count': int(df[TARGET_COLUMN].nunique()),
        'cluster_names': cluster_names,
    }
    with open(REPORTS_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        'Итоговый отчет по лабораторной работе',
        f'Количество записей в датасете: {len(df)}',
        f'Количество признаков для анализа: {len(FEATURE_COLUMNS)}',
        f'Выбранное число кластеров KMeans: {best_k}',
        f'Silhouette Score выбранной кластеризации: {best_score:.4f}',
        f'Лучшая модель классификации стиля обучения: {best_classifier}',
        '',
        'Сводка по кластерам:',
        cluster_summary.to_string(index=False),
        '',
        'Сравнение моделей классификации:',
        classification_metrics.to_string(index=False),
    ]
    (REPORTS_DIR / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    create_project_structure()
    df = load_data()
    features_df, x_scaled, scaler = prepare_features(df)
    plot_correlation_matrix(features_df)
    cluster_labels, best_k, best_score, cluster_summary, scores_df, cluster_names = perform_clustering(df, features_df, x_scaled)
    classification_metrics, best_classifier = perform_multiclass_classification(features_df, df[TARGET_COLUMN])
    create_final_report(df, best_k, best_score, cluster_summary, scores_df, classification_metrics, best_classifier, cluster_names)
    print('Проект успешно выполнен')
    print(f'Выбранное количество кластеров: {best_k}')
    print(f'Silhouette Score: {best_score:.4f}')
    print(f'Лучшая модель классификации: {best_classifier}')


if __name__ == '__main__':
    main()

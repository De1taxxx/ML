# Лабораторная работа 2. Кластеризация студентов онлайн-платформы по стилю обучения

Проект посвящен анализу учебной активности студентов онлайн-курса с применением методов машинного обучения. В работе выполняется кластеризация студентов по поведенческим признакам, визуализация групп и классификация стиля обучения.

## Структура проекта

```text
lab2_online_learning_clustering/
├── data/
│   └── students_activity.csv
├── models/
├── reports/
├── generate_data.py
├── project.py
├── predict.py
├── requirements.txt
└── README.md
```

## Используемые признаки

- lectures_watched_pct;
- weekly_study_hours;
- assignments_completed_pct;
- avg_quiz_score;
- missed_deadlines;
- forum_posts;
- logins_per_month;
- days_since_last_login;
- video_rewatch_count;
- messages_to_teacher;
- mobile_activity_pct;
- progress_consistency;
- final_exam_score.

## Целевые группы

- video_focused_students;
- practice_oriented_students;
- social_learners;
- at_risk_students;
- catch_up_students.

## Используемые методы

- KMeans;
- DBSCAN;
- Agglomerative Clustering;
- PCA;
- Random Forest;
- Logistic Regression;
- K-Nearest Neighbors.

## Запуск проекта

```bash
pip install -r requirements.txt
python generate_data.py
python project.py
python predict.py
```

## Результаты

После выполнения `project.py` автоматически создаются:

- `reports/silhouette_scores.csv` — сравнение качества кластеризации;
- `reports/cluster_summary.csv` — сводка по кластерам;
- `reports/classification_metrics.csv` — метрики классификации;
- `reports/report.txt` — итоговый текстовый отчет;
- `reports/correlation_matrix.png`;
- `reports/silhouette_scores.png`;
- `reports/pca_clusters.png`;
- `reports/confusion_matrix.png`;
- `reports/feature_importance.png`;
- `models/scaler.pkl`;
- `models/kmeans_model.pkl`;
- `models/learning_style_classifier.pkl`;
- `models/label_encoder.pkl`.

Выбранное число кластеров KMeans — 5. Такой вариант позволяет интерпретировать группы студентов не только как успешных и неуспешных, но и как разные стили учебного поведения.

# Лабораторная работа №1

## Прогнозирование успеваемости студентов онлайн-курса

Проект посвящен задаче бинарной классификации студентов онлайн-курса. На основе активности пользователя система прогнозирует, завершит ли студент курс успешно.

## Цель

Разработать модель машинного обучения, способную по учебной активности студента определить вероятность успешного завершения онлайн-курса.

## Структура проекта

```text
lab1_student_performance_prediction/
├── data.csv
├── generate_data.py
├── code.py
├── predict.py
├── requirements.txt
├── models/
│   ├── student_performance_model.pkl
│   └── feature_columns.pkl
└── reports/
    ├── metrics.txt
    ├── metrics.csv
    ├── metrics.json
    ├── summary.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.csv
    └── feature_importance.png
```

## Используемые признаки

- возраст студента;
- уровень курса;
- наличие предыдущего опыта;
- количество часов обучения в неделю;
- количество просмотренных лекций;
- количество выполненных заданий;
- средний балл за тесты;
- активность на форуме;
- количество пропущенных дедлайнов;
- количество дней с последней активности;
- количество выполненных тренировочных тестов;
- процент завершения видеоматериалов;
- посещаемость онлайн-занятий.

Целевая переменная: `passed`.

## Используемые модели

- Logistic Regression;
- Random Forest;
- K-Nearest Neighbors.

## Метрики качества

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

## Запуск

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Генерация датасета:

```bash
python generate_data.py
```

Обучение моделей:

```bash
python code.py
```

Пример предсказания:

```bash
python predict.py --study_hours_per_week 8.5 --lectures_viewed 45 --assignments_submitted 15 --avg_quiz_score 78
```

## Результат

В результате работы скрипта формируются обученная модель, отчет с метриками, матрица ошибок, ROC-кривая и график важности признаков. Лучшая модель выбирается по значению ROC-AUC.

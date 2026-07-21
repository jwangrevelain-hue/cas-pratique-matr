# Pipeline complet : nettoyage -> dataset analytique -> métriques ->
# entraînement -> scoring.
#
#   docker build -t churn-pipeline .
#   docker run --rm -v "$PWD/outputs:/app/outputs" churn-pipeline
#
# Tous les livrables (analytics_subscriptions.csv, metrics_*.csv,
# churn_model.pkl, churn_scores.csv, evaluation.md) sont écrits dans
# /app/outputs, monté sur l'hôte via -v.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY users_raw.csv subscriptions_raw.csv events_raw.csv ./
COPY cleaning.py build_dataset.py features.py metrics.py train.py predict.py run_all.py ./

CMD ["python", "run_all.py"]

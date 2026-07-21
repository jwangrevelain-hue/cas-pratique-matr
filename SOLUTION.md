# Solution — Subscription Analytics & Churn

**`analyse_churn.ipynb`** déroule toute l'analyse (exploration → nettoyage →
métriques → modèle → note métier) avec graphiques et narration : c'est le
document à ouvrir en premier. Il est une couche de présentation au-dessus des
modules de production ci-dessous — mêmes fonctions, mêmes chiffres.

## Exécution

```bash
# Local (Python 3.11+)
pip install -r requirements.txt
python run_all.py            # pipeline complet : ~30 s

# Docker (reproduit tous les livrables sans dépendance locale)
docker build -t churn-pipeline .
docker run --rm -v "$PWD/outputs:/app/outputs" churn-pipeline
```

Chaque étape est aussi exécutable seule, dans l'ordre :
`cleaning.py` → `build_dataset.py` → `metrics.py` → `train.py` → `predict.py`.

```bash
python predict.py u_00155        # probabilité de churn d'un utilisateur actif
python predict.py --all          # score toute la base active
```

## Livrables

| Livrable (README) | Fichier |
|---|---|
| Tables nettoyées + note des choix | `data_clean/*.csv`, `NOTES_CHOIX.md` |
| `analytics_subscriptions.csv` | `outputs/analytics_subscriptions.csv` |
| `metrics.py` + tables agrégées | `metrics.py`, `outputs/metrics_*.csv`, lecture dans `outputs/metrics_summary.md` |
| `train.py`, modèle sérialisé, `predict.py` | `train.py`, `outputs/churn_model.pkl`, `predict.py` (+ rapport `outputs/evaluation.md`) |
| Note métier (1 page) | `NOTE_METIER.md` |
| `Dockerfile` | `Dockerfile` (+ `run_all.py`) |
| Analyse commentée (bonus) | `analyse_churn.ipynb` |

## Points de conception clés

- **Cible** : T = 2026-04-06 (dernière date où la fenêtre ]T, T+90j] est
  entièrement observable). Churn = résiliation dans la fenêtre **ou** arrêt de
  paiement d'un cycle mensuel ; l'absence de paiement annuel n'est pas un
  signal (échéance hors fenêtre). Détail : `NOTES_CHOIX.md`.
- **Anti-fuite** : features construites uniquement sur les données ≤ T dans
  `features.py` (module partagé train/serving) ; `status`, `end_date`,
  `total_revenue` **et `last_payment_date`** (colonne snapshot, max 2026-07-06
  > T) exclues — la récence de paiement est recalculée depuis les événements.
- **Validation temporelle** : train sur cohortes anciennes, test sur les 25 %
  les plus récentes ; sélection de modèle sur un split temporel interne au
  train. AUC test **0,75** (gradient boosting).
- **Seuil** : dérivé des coûts, p* = coût action / (conversion × 6 mois de
  revenu) ≈ 0,32, et non du F1 — justification économique dans
  `NOTE_METIER.md`.

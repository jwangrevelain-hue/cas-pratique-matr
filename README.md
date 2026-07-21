# Mini-Challenge — Subscription Analytics & Churn

**Objectif global**
À partir de tables applicatives brutes, produire un dataset analytique propre, un modèle de détection de churn à 90 jours, et une note d'interprétation métier reliant le modèle à une décision business chiffrée.

**Format** : 100% local (Python / notebook). Aucune plateforme externe requise.
**Durée cible** : ~2h30.
**Ce qu'on évalue** : la rigueur du raisonnement (surtout sur la fuite de données et la validation temporelle) et la capacité à traduire un résultat ML en décision, bien plus que la performance brute du modèle.

---

## Datasets fournis (bruts)

### `users_raw.csv`
`user_id`, `email`, `created_at` (formats de date mélangés volontairement), `country`, `signup_source` (web / mobile / referral / partner), `age`, `device_type` (desktop / mobile / tablet).

Problèmes intentionnels : doublons sur `user_id`, formats de date hétérogènes.

### `subscriptions_raw.csv`
`subscription_id`, `user_id`, `plan_name` (Basic / Pro / Premium, avec typos et variantes), `start_date`, `end_date` (vide si actif), `status`, `monthly_price`, `auto_renew`, `billing_cycle`, `last_payment_date`, `total_revenue`.

Problèmes intentionnels : typos/variations dans `plan_name` (ex. `premium `, `BAsic`, `Proo`, `pr0`), doublons sur `subscription_id`.

### `events_raw.csv`
`user_id`, `event_date`, `event_type` (login / payment), une ligne par événement horodaté.
C'est la table comportementale : c'est d'elle que doivent venir les features prédictives (récence, fréquence, tendance d'usage). Elle contient du bruit (événements manquants, doublons).

---

## Partie 1 — Nettoyage

- Dédupliquer sur `user_id` et `subscription_id` (choisir et justifier la règle de conservation).
- Normaliser `created_at` vers un format date unique.
- Standardiser `plan_name` vers exactement : `Basic`, `Pro`, `Premium`.
- Documenter en 3-4 lignes les décisions non triviales (ex. comment on tranche un doublon en conflit).

**Livrable** : les tables nettoyées + une courte note des choix.

---

## Partie 2 — Dataset analytique

Construire `analytics_subscriptions` (une ligne par abonnement) avec au minimum :

`user_id`, `plan_name`, `subscription_start_date`, `subscription_end_date`, `is_active` (1/0), `tenure_days`, `total_revenue`, `monthly_price`, `auto_renew`, `billing_cycle`, `last_payment_date`, `country`, `signup_source`, `device_type`, `age`, `cohort_month` (dérivé de la date de début).

**Livrable** : `analytics_subscriptions.csv` (ou `.parquet`).

---

## Partie 3 — Métriques analytiques

Calculer, dans un script, les métriques répondant à l'objectif :
*mesurer la performance des abonnements dans le temps et donner une vue claire de la base active et du revenu.*

Au minimum : abonnés actifs par mois, nouvelles souscriptions vs annulations par mois, MRR par mois, revenu cumulé, répartition par plan et par canal d'acquisition, table de cohortes mensuelles.

**Livrable** : `metrics.py` produisant une ou plusieurs tables agrégées (`metrics_*.csv`), plus 3-4 lignes de lecture des chiffres clés. Aucun notebook attendu.

---

## Partie 4 — Modèle de churn (le cœur du test)

Construire un modèle de classification binaire prédisant le **churn à 90 jours**, entièrement scripté et reproductible.

### 4.1 — Définition de la cible (à écrire noir sur blanc)
- Choisir une **date d'observation** `T`.
- `churn_90d = 1` si l'utilisateur, actif à `T`, résilie ou cesse de payer dans la fenêtre `]T, T+90j]` ; `0` sinon.
- Ne conserver dans l'échantillon que les utilisateurs **actifs à `T`** (on ne prédit pas le churn de gens déjà partis).

### 4.2 — Anti-fuite de données (critère de notation principal)
Interdiction stricte d'utiliser toute variable qui n'est pas connue **à la date `T`** ou qui encode la cible :
- `status`, `end_date`, `total_revenue` (agrégé a posteriori) → **exclus** comme features.
- Toute feature construite depuis `events_raw` doit être calculée **uniquement sur la période antérieure à `T`**.
- Le candidat doit lister explicitement les features retenues et justifier leur légitimité temporelle.

### 4.3 — Validation temporelle
- **Split temporel**, pas aléatoire : entraîner sur les cohortes anciennes, tester sur les récentes.
- Justifier pourquoi un `train_test_split` aléatoire serait ici trompeur.

### 4.4 — Features attendues (depuis `events_raw`, avant `T`)
Au moins : récence du dernier login, nombre de logins sur 30/60/90 jours, tendance d'usage (usage récent vs antérieur), récence du dernier paiement, régularité des paiements. Plus les attributs statiques légitimes (plan, prix, cycle de facturation, canal, device, âge, ancienneté à `T`).

### 4.5 — Évaluation
- Métriques : **AUC** + **precision/recall à un seuil choisi et justifié** (pas seulement l'accuracy, trompeuse sur classes déséquilibrées).
- Matrice de confusion au seuil retenu.
- Importance des features (et commentaire : est-ce cohérent avec l'intuition métier ?).

### 4.6 — Livrables techniques
- `train.py` reproductible : de `analytics_subscriptions` + `events_raw` au modèle sérialisé (`model.pkl` ou équivalent), incluant construction de la cible, split temporel, entraînement, évaluation.
- Le modèle sérialisé.
- Un `predict.py` (ou fonction) qui prend un utilisateur actif et renvoie une probabilité de churn.

---

## Partie 5 — Interprétation métier (fortement pondérée)

Une note courte (1 page max) qui relie le modèle à une décision :

1. **Chiffrer l'enjeu.** À partir du MRR et du taux de churn observé, estimer la perte de revenu annualisée liée au churn.
2. **Traduire le modèle en action.** Si on cible les *X %* d'utilisateurs au risque le plus élevé avec une action de rétention coûtant *Y* par utilisateur et convertissant à *Z %*, l'opération est-elle rentable ? Poser le mini-calcul de ROI.
3. **Justifier le seuil de décision par le coût, pas par le F1.** Rappeler un client à tort coûte peu ; rater un churner coûte son revenu futur. Le seuil doit refléter cette asymétrie.
4. **Limites et prochaine donnée utile.** Ce que le modèle ne voit pas, et quelle donnée supplémentaire améliorerait le plus la prédiction.

---

## Récapitulatif des livrables
- `analytics_subscriptions.csv`
- `metrics.py` + tables agrégées (`metrics_*.csv`)
- `train.py`, modèle sérialisé, `predict.py`
- Note métier (1 page)
- `Dockerfile` : conteneurise le pipeline complet (nettoyage → dataset analytique → entraînement → prédiction). `docker build` puis `docker run` doit reproduire l'ensemble des livrables sans dépendance à l'environnement local.

## Fichiers
- `users_raw.csv`
- `subscriptions_raw.csv`
- `events_raw.csv`

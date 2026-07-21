# Notes des choix

## Partie 1 — Nettoyage (décisions non triviales)

- **Doublons** (60 lignes users, 40 subscriptions, 12 326 events) : tous sont
  des **copies strictement identiques** → suppression directe. La règle
  implémentée pour un éventuel conflit (`cleaning.dedupe`) : garder la ligne
  la plus complète (moins de nulls), départagée par la date la plus récente —
  l'enregistrement le plus récent est l'état le plus fiable du système source.
- **Dates `created_at`** : 6 formats explicitement listés (pas de parsing
  "magique"). Pour les motifs ambigus `DD/MM` et `DD-MM`, l'ordre jour/mois est
  prouvé par les 65 % de valeurs dont le premier composant dépasse 12, et
  supposé homogène au sein d'un même motif.
- **`plan_name`** : normalisation casse/espaces + table de correspondance des
  typos (`Bacic`, `pr0`, `Proo`, `Premuim`, `Premiun`…), puis **contrôle croisé
  avec `monthly_price`** (9,99 ↔ Basic, 19,99 ↔ Pro, 39,99 ↔ Premium) : 100 %
  de cohérence, aucune réassignation arbitraire.
- **Events** : granularité journalière → les lignes identiques sont
  indistinguables d'un doublon technique ; on conserve 1 événement par
  (user, jour, type).

## Partie 4.2 — Features retenues et légitimité temporelle

Date d'observation **T = 2026-04-06** (= dernière date des données − 90 j,
seule date où la fenêtre de churn est entièrement observable). Toutes les
features sont calculées **exclusivement sur ]-∞, T]** (`features.py`, testé
par assertion) :

| Feature | Source | Légitimité à T |
|---|---|---|
| `days_since_last_login`, `logins_30/60/90d` | events ≤ T | fenêtres bornées par T |
| `usage_trend` (logins 30 j vs 30 j précédents) | events ≤ T | idem |
| `logins_per_week_lifetime` | events ≤ T | idem |
| `days_since_last_payment`, `payment_gap_std` | events ≤ T | recalculées depuis les événements, pas depuis la colonne snapshot |
| `payment_overdue_ratio` (récence / cycle) | events ≤ T + billing_cycle | idem |
| `tenure_days_at_T` | start_date | ancienneté **à T**, pas la tenure finale |
| `plan_name`, `monthly_price`, `billing_cycle`, `auto_renew` | contrat | attributs contractuels connus dès la souscription |
| `country`, `signup_source`, `device_type`, `age` | profil | connus à l'inscription |

**Exclusions (fuite de données)** :

- `status`, `end_date` — encodent directement la cible ;
- `total_revenue` — agrégé jusqu'à l'extraction, donc postérieur à T ;
- `last_payment_date` (colonne de la table subscriptions) — c'est un champ
  **au moment de l'extraction** (max = 2026-07-06 > T) ; la récence de paiement
  est recalculée depuis `events_raw` filtré à T ;
- `is_active`, `tenure_days` du dataset analytique — dérivés du snapshot.

## Cible (Partie 4.1, noir sur blanc)

`churn_90d = 1` si, pour un abonnement **actif à T** (commencé ≤ T, non résilié
à T) : résiliation dans `]T, T+90j]`, **ou** cycle mensuel sans aucun paiement
dans la fenêtre (≈ 3 échéances manquées). L'absence de paiement d'un cycle
**annuel** n'est pas un signal de churn : l'échéance peut légitimement tomber
hors fenêtre — vérifié dans les données, les 114 annuels sans paiement dans la
fenêtre se connectent encore tous début juillet. Résultat : 799 actifs à T,
97 churners (12,1 %), tous par résiliation explicite.

## Validation temporelle (Partie 4.3)

Une seule fenêtre de churn est observable (les résiliations vont d'avril à
juin 2026) : impossible de tester sur un T ultérieur. La dimension temporelle
restante est la **cohorte d'inscription** : entraînement sur les cohortes
< 2025-10-20 (599 lignes), test sur les 25 % les plus récentes (200 lignes).
Un `train_test_split` aléatoire serait trompeur : il mélangerait les périodes
entre train et test, laissant le modèle exploiter des régularités propres à
chaque cohorte (mix de plans, intensité d'usage) — alors qu'en production on
score toujours des cohortes plus récentes que celles de l'entraînement. La
sélection de modèle (régression logistique vs gradient boosting) se fait sur
un second split temporel **interne au train**, jamais sur le test.

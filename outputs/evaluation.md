# Évaluation du modèle de churn 90 jours

- **Cible** : actifs à T=2026-04-06, churn si résiliation ou arrêt de
  paiement (cycle mensuel) dans ]T, T+90j]. Train : 599 lignes
  (12.5% churn), test : 200 lignes (11.0%).
- **Split temporel** : cohortes < 2025-10-20 en train, plus récentes en test.
  Un split aléatoire mélangerait les périodes et surestimerait la
  généralisation aux cohortes futures, seul cas d'usage réel en production.
- **Features** : exclusivement calculées sur ]-inf, T] ; `status`, `end_date`,
  `total_revenue`, `last_payment_date` exclues (connues après T / encodent la cible).

| Modèle | AUC validation interne |
|---|---|
| logistic_regression | 0.702 |
| hist_gradient_boosting | 0.800 |

**Champion : hist_gradient_boosting** — **AUC test = 0.751**

## Seuil de décision (par le coût, pas le F1)

p* = coût action / (conversion x revenu préservé) = 10 / (0.3 x 6 x prix mensuel moyen) = **0.32**

Au seuil 0.32 : precision **0.37**, rappel **0.32**,
19 utilisateurs ciblés sur 200, gain espéré de la campagne
**+8 €** sur la fenêtre de 90 j (test). Le seuil est bas car
l'asymétrie des coûts le veut : contacter à tort coûte 10 €, rater un churner coûte ~6 mois de revenu.

Matrice de confusion (test) : TN=166, FP=12, FN=15, TP=7

## Importance des features (permutation, AUC, jeu de test)

| Feature | ΔAUC |
|---|---|
| logins_30d | 0.1614 |
| days_since_last_login | 0.0290 |
| logins_per_week_lifetime | 0.0257 |
| auto_renew | 0.0163 |
| logins_90d | 0.0126 |
| usage_trend | 0.0121 |
| device_type | 0.0055 |
| logins_60d | 0.0041 |
| plan_name | 0.0030 |
| signup_source | 0.0024 |

Lecture métier : le signal est presque entièrement comportemental —
fréquence de login récente (logins_30d domine largement), récence du
dernier login et tendance d'usage. Cohérent avec l'intuition : un client
qui churne se désengage avant de résilier. `auto_renew` contribue aussi
(renouvellement manuel = friction). Les features de paiement pèsent peu
ici : dans ces données, les mensuels paient jusqu'à la résiliation, la
récence de paiement est donc redondante avec l'activité. Les attributs
statiques (pays, device, âge) sont marginaux.

## Notes
- Pas de repondération de classes : le seuil s'applique à des probabilités
  approximativement calibrées, ce que la repondération casserait.
- Test = 200 lignes (~22 churners) : les métriques
  ont une variance non négligeable ; à re-estimer sur plus d'historique.

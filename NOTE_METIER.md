# Note métier — Churn à 90 jours : de la prédiction à la décision

**Situation.** Fin juin 2026 : **702 abonnés actifs, MRR 12 353 €**. Sur le
dernier trimestre, **98 clients (12,2 % de la base) ont résilié**, emportant
~1 500 € de MRR.

## 1. L'enjeu chiffré

Un churn de 12,2 % par trimestre correspond à **≈ 41 % de la base sur un an**.
Au MRR actuel, cela détruit ≈ 5 600 € de MRR en un an, soit **≈ 68 000 € de
revenu annualisé** — sans compter le coût d'acquisition pour reconstituer la
base. Nuance importante : le churn est concentré sur le plan Basic (prix moyen
du churner : 15,3 €/mois contre 17,6 € pour la base) ; on perd du volume plus
que de la valeur, ce qui borne ce qu'on peut dépenser pour retenir.

## 2. Traduire le modèle en action

Le modèle (AUC test 0,75) concentre bien le risque : dans les **5 % les plus
risqués, 40 % churnent réellement** (3,6× le taux de base). Rentabilité d'une
campagne simulée sur le jeu de test (200 clients, 22 churners), en supposant
qu'une action coûtant *Y* retient un churner avec probabilité *Z* et préserve
6 mois de revenu :

| Action | Y | Z | Seuil p* | Ciblés | Churners touchés | Gain / 90 j |
|---|---|---|---|---|---|---|
| Appel + geste commercial | 10 € | 30 % | 0,32 | 19 | 7/22 | **+8 €** |
| Offre ciblée | 5 € | 30 % | 0,16 | 37 | 10/22 | **+67 €** |
| Email/in-app automatisé | 2 € | 20 % | 0,10 | 45 | 12/22 | **+114 €** |

**Lecture** : avec un churner moyen à 15 €/mois, une action coûteuse est à
peine rentable ; une action automatisée bon marché dégage ≈ +114 € par
tranche de 200 clients, soit **≈ +400 €/trimestre (~1 600 €/an) étendue aux
702 actifs** — et le geste coûteux doit être réservé aux profils Premium,
dont la valeur préservée (240 €) justifie un seuil de ciblage plus bas.
Aujourd'hui, 39 clients actifs dépassent le seuil d'action
(`outputs/churn_scores.csv`).

## 3. Le seuil de décision : par le coût, pas par le F1

On cible un client dès que le gain espéré est positif :
`p × Z × (6 × prix mensuel) > Y`, d'où **p\* = Y / (Z × valeur préservée)**.
Rappeler un client à tort coûte Y (2–10 €) ; rater un churner coûte ~6 mois de
revenu (60–240 €). Cette asymétrie pousse volontairement le seuil **bien en
dessous de 0,5** (0,10–0,32 selon l'action) : on accepte une majorité de faux
positifs parce qu'ils sont bon marché. Un seuil optimisé au F1 traiterait les
deux erreurs comme équivalentes, ce qui est économiquement faux ici.

## 4. Limites et prochaine donnée utile

- Le modèle ne voit que logins et paiements : il ne distingue pas churn
  volontaire et **échec de paiement**, et ne voit ni l'insatisfaction
  (tickets support, NPS) ni les changements de plan.
- Une seule fenêtre de churn est observable (avr.–juin 2026) : la validation
  est faite sur les cohortes récentes, pas sur une période future ; à
  re-valider dès qu'un deuxième trimestre d'historique existe. Le test
  (200 clients, 22 churners) donne des métriques encore volatiles.
- **Donnée à collecter en priorité** : les échecs de prélèvement et le motif
  de résiliation (sépare le churn involontaire, actionnable par du dunning,
  du churn d'usage) ; ensuite, l'usage produit fin (features utilisées,
  durée de session) pour anticiper le désengagement plus tôt que la simple
  fréquence de login.

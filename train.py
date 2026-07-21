"""Partie 4 — Modèle de churn à 90 jours, reproductible de bout en bout.

Définition de la cible (cf. features.build_target) :
  - Date d'observation T = dernière date telle que la fenêtre ]T, T+90j]
    soit entièrement observée dans les données (T = snapshot - 90j).
  - Population : abonnements actifs à T uniquement.
  - churn_90d = 1 si résiliation dans ]T, T+90j], ou arrêt des paiements
    d'un cycle mensuel sur la fenêtre.

Validation temporelle :
  Les résiliations n'étant observables que sur une seule fenêtre de 90j en
  fin d'historique, on ne peut pas tester sur une date T ultérieure ; la
  dimension temporelle disponible est la cohorte d'inscription. On entraîne
  donc sur les cohortes anciennes et on teste sur les 25 % les plus récentes.
  Un split aléatoire serait trompeur : il mélangerait les cohortes entre
  train et test, laissant le modèle profiter de régularités propres à chaque
  période (mix de plans, saisonnalité d'usage) qu'il ne connaîtra jamais en
  production, où l'on score toujours des cohortes plus récentes que celles
  de l'entraînement. La sélection de modèle utilise un second split temporel
  interne au train, pour ne jamais choisir sur le test.

Seuil de décision : dérivé des coûts (voir NOTE_METIER.md), pas du F1.
  Cibler un utilisateur coûte RETENTION_COST ; s'il allait churner, l'action
  le retient avec probabilité CONVERSION et préserve SAVED_MONTHS de revenu.
  L'espérance de gain est positive ssi p > COST / (CONVERSION x valeur),
  ce qui donne directement le seuil p*.

Sortie : outputs/churn_model.pkl, outputs/evaluation.md
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    build_target,
    load_clean_tables,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

HORIZON_DAYS = 90
TEST_COHORT_SHARE = 0.25
RANDOM_STATE = 42

# Hypothèses économiques du seuil (reprises dans NOTE_METIER.md)
RETENTION_COST = 10.0   # coût de l'action de rétention par utilisateur ciblé (€)
CONVERSION = 0.30       # probabilité que l'action retienne un churner
SAVED_MONTHS = 6        # revenu préservé si le churner est retenu (mois de MRR)


def make_candidates() -> dict[str, Pipeline]:
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return {
        "logistic_regression": Pipeline([
            ("prep", encode),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", encode),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def temporal_split(X: pd.DataFrame, share: float):
    """Cohortes anciennes -> train, cohortes les plus récentes -> éval."""
    cutoff = X["start_date"].quantile(1 - share)
    return X["start_date"] < cutoff, cutoff


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    users, subs, events = load_clean_tables()

    snapshot = events["event_date"].max()
    T = snapshot - pd.Timedelta(days=HORIZON_DAYS)
    print(f"Snapshot {snapshot.date()} -> date d'observation T = {T.date()}")

    X = build_features(subs, users, events, T)
    y = build_target(subs, events, T, HORIZON_DAYS).reindex(X.index)

    train_mask, cutoff = temporal_split(X, TEST_COHORT_SHARE)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"  split temporel sur start_date < {cutoff.date()} : "
          f"train {len(X_train)} (churn {y_train.mean():.1%}, cohortes "
          f"{X_train['start_date'].min().date()} -> {X_train['start_date'].max().date()}), "
          f"test {len(X_test)} (churn {y_test.mean():.1%}, cohortes "
          f"{X_test['start_date'].min().date()} -> {X_test['start_date'].max().date()})")

    # --- Sélection de modèle sur un split temporel interne au train ---------
    inner_mask, inner_cutoff = temporal_split(X_train, TEST_COHORT_SHARE)
    val_auc = {}
    for name, pipe in make_candidates().items():
        pipe.fit(X_train[inner_mask][FEATURES], y_train[inner_mask])
        p = pipe.predict_proba(X_train[~inner_mask][FEATURES])[:, 1]
        val_auc[name] = roc_auc_score(y_train[~inner_mask], p)
        print(f"  validation interne (cohortes >= {inner_cutoff.date()}) — "
              f"{name}: AUC {val_auc[name]:.3f}")
    champion = max(val_auc, key=val_auc.get)

    model = make_candidates()[champion]
    model.fit(X_train[FEATURES], y_train)

    # --- Évaluation sur les cohortes de test --------------------------------
    proba = model.predict_proba(X_test[FEATURES])[:, 1]
    auc = roc_auc_score(y_test, proba)

    avg_value = SAVED_MONTHS * X["monthly_price"].mean()
    threshold = RETENTION_COST / (CONVERSION * avg_value)
    pred = (proba >= threshold).astype(int)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    # Politique de ciblage évaluée économiquement sur le test
    targeted = pred == 1
    expected_gain = (
        CONVERSION * SAVED_MONTHS * X_test.loc[targeted, "monthly_price"] * y_test[targeted]
    ).sum() - RETENTION_COST * targeted.sum()

    print(f"\nChampion : {champion} — AUC test {auc:.3f}")
    print(f"Seuil de coût p* = {RETENTION_COST}€ / ({CONVERSION} x {avg_value:.0f}€) "
          f"= {threshold:.2f}")
    print(f"  precision {precision:.2f}, recall {recall:.2f} au seuil {threshold:.2f}")
    print(f"  matrice de confusion [[TN FP][FN TP]] : {cm.tolist()}")
    print(f"  gain espéré de la campagne sur le test : {expected_gain:+.0f} € / 90j")

    # --- Importance des features (permutation, sur le test) -----------------
    imp = permutation_importance(
        model, X_test[FEATURES], y_test, scoring="roc_auc",
        n_repeats=20, random_state=RANDOM_STATE,
    )
    importance = (
        pd.Series(imp.importances_mean, index=FEATURES)
        .sort_values(ascending=False)
    )
    print("\nTop features (perte d'AUC si permutée) :")
    print(importance.head(8).round(4).to_string())

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "champion": champion,
            "T": str(T.date()),
            "horizon_days": HORIZON_DAYS,
            "threshold": threshold,
            "test_auc": auc,
        },
        OUT_DIR / "churn_model.pkl",
    )

    write_report(
        champion=champion, T=T, cutoff=cutoff, val_auc=val_auc, auc=auc,
        threshold=threshold, precision=precision, recall=recall, cm=cm,
        importance=importance, y_train=y_train, y_test=y_test,
        expected_gain=expected_gain, n_targeted=int(targeted.sum()),
    )
    print(f"\nModèle -> {OUT_DIR / 'churn_model.pkl'}, "
          f"rapport -> {OUT_DIR / 'evaluation.md'}")


def write_report(*, champion, T, cutoff, val_auc, auc, threshold, precision,
                 recall, cm, importance, y_train, y_test, expected_gain,
                 n_targeted) -> None:
    (tn, fp), (fn, tp) = cm
    lines = [
        "# Évaluation du modèle de churn 90 jours",
        "",
        f"- **Cible** : actifs à T={T.date()}, churn si résiliation ou arrêt de",
        f"  paiement (cycle mensuel) dans ]T, T+90j]. Train : {len(y_train)} lignes",
        f"  ({y_train.mean():.1%} churn), test : {len(y_test)} lignes ({y_test.mean():.1%}).",
        f"- **Split temporel** : cohortes < {cutoff.date()} en train, plus récentes en test.",
        "  Un split aléatoire mélangerait les périodes et surestimerait la",
        "  généralisation aux cohortes futures, seul cas d'usage réel en production.",
        f"- **Features** : exclusivement calculées sur ]-inf, T] ; `status`, `end_date`,",
        "  `total_revenue`, `last_payment_date` exclues (connues après T / encodent la cible).",
        "",
        f"| Modèle | AUC validation interne |",
        f"|---|---|",
        *[f"| {n} | {a:.3f} |" for n, a in val_auc.items()],
        "",
        f"**Champion : {champion}** — **AUC test = {auc:.3f}**",
        "",
        f"## Seuil de décision (par le coût, pas le F1)",
        "",
        f"p* = coût action / (conversion x revenu préservé) = "
        f"{RETENTION_COST:.0f} / ({CONVERSION} x {SAVED_MONTHS} x prix mensuel moyen) "
        f"= **{threshold:.2f}**",
        "",
        f"Au seuil {threshold:.2f} : precision **{precision:.2f}**, rappel **{recall:.2f}**,",
        f"{n_targeted} utilisateurs ciblés sur {len(y_test)}, gain espéré de la campagne",
        f"**{expected_gain:+.0f} €** sur la fenêtre de 90 j (test). Le seuil est bas car",
        "l'asymétrie des coûts le veut : contacter à tort coûte "
        f"{RETENTION_COST:.0f} €, rater un churner coûte ~{SAVED_MONTHS} mois de revenu.",
        "",
        f"Matrice de confusion (test) : TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "",
        "## Importance des features (permutation, AUC, jeu de test)",
        "",
        "| Feature | ΔAUC |",
        "|---|---|",
        *[f"| {f} | {v:.4f} |" for f, v in importance.head(10).items()],
        "",
        "Lecture métier : le signal est presque entièrement comportemental —",
        "fréquence de login récente (logins_30d domine largement), récence du",
        "dernier login et tendance d'usage. Cohérent avec l'intuition : un client",
        "qui churne se désengage avant de résilier. `auto_renew` contribue aussi",
        "(renouvellement manuel = friction). Les features de paiement pèsent peu",
        "ici : dans ces données, les mensuels paient jusqu'à la résiliation, la",
        "récence de paiement est donc redondante avec l'activité. Les attributs",
        "statiques (pays, device, âge) sont marginaux.",
        "",
        "## Notes",
        "- Pas de repondération de classes : le seuil s'applique à des probabilités",
        "  approximativement calibrées, ce que la repondération casserait.",
        f"- Test = {len(y_test)} lignes (~{int(y_test.sum())} churners) : les métriques",
        "  ont une variance non négligeable ; à re-estimer sur plus d'historique.",
    ]
    (OUT_DIR / "evaluation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

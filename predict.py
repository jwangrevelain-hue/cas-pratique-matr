"""Partie 4 — Scoring : probabilité de churn 90j d'un utilisateur actif.

Usage :
  python predict.py u_00042            # score un utilisateur
  python predict.py --all              # score tous les actifs -> churn_scores.csv
  python predict.py u_00042 --date 2026-04-06   # score à une date passée

Les features sont recalculées à la date de scoring avec exactement le même
code que l'entraînement (features.build_features) : aucune divergence
train/serving possible.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURES, build_features, load_clean_tables

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def score_actives(at_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Probabilité de churn de chaque utilisateur actif à la date donnée."""
    bundle = joblib.load(OUT_DIR / "churn_model.pkl")
    users, subs, events = load_clean_tables()

    at_date = at_date or events["event_date"].max()
    X = build_features(subs, users, events, at_date)

    scores = pd.DataFrame({
        "user_id": X.index,
        "churn_probability": bundle["model"].predict_proba(X[FEATURES])[:, 1].round(4),
        "monthly_price": X["monthly_price"].values,
    })
    scores["target_for_retention"] = scores["churn_probability"] >= bundle["threshold"]
    return scores.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", help="ex. u_00042")
    parser.add_argument("--all", action="store_true", help="score tous les actifs")
    parser.add_argument("--date", type=pd.Timestamp, default=None,
                        help="date de scoring (défaut : dernière date des données)")
    args = parser.parse_args()
    if not args.all and not args.user_id:
        parser.error("donner un user_id ou --all")

    scores = score_actives(args.date)

    if args.all:
        path = OUT_DIR / "churn_scores.csv"
        scores.to_csv(path, index=False)
        print(f"{len(scores)} utilisateurs actifs scorés -> {path}")
        print(scores.head(10).to_string(index=False))
        return

    row = scores[scores["user_id"] == args.user_id]
    if row.empty:
        raise SystemExit(f"{args.user_id} : inconnu ou non actif à la date de scoring "
                         "(le churn ne se prédit que pour les actifs)")
    r = row.iloc[0]
    rank = row.index[0] + 1
    print(f"{r['user_id']} : probabilité de churn à 90 j = {r['churn_probability']:.1%} "
          f"(rang {rank}/{len(scores)}) — "
          f"{'À CIBLER par la rétention' if r['target_for_retention'] else 'sous le seuil d’action'}")


if __name__ == "__main__":
    main()

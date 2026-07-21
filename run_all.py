"""Pipeline complet : nettoyage -> dataset analytique -> métriques ->
entraînement -> scoring. Reproduit tous les livrables dans outputs/.
"""

import build_dataset
import cleaning
import metrics
import predict
import train


def main() -> None:
    for step in (cleaning, build_dataset, metrics, train):
        print(f"\n{'=' * 60}\n>>> {step.__name__}\n{'=' * 60}")
        step.main()

    print(f"\n{'=' * 60}\n>>> predict (scoring de la base active)\n{'=' * 60}")
    scores = predict.score_actives()
    scores.to_csv(predict.OUT_DIR / "churn_scores.csv", index=False)
    print(f"{len(scores)} utilisateurs actifs scorés -> outputs/churn_scores.csv")
    print(scores.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

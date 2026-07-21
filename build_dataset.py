"""Partie 2 — Dataset analytique : une ligne par abonnement.

Produit outputs/analytics_subscriptions.csv à partir des tables nettoyées.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"

COLUMNS = [
    "subscription_id", "user_id", "plan_name", "subscription_start_date",
    "subscription_end_date", "is_active", "tenure_days", "total_revenue",
    "monthly_price", "auto_renew", "billing_cycle", "last_payment_date",
    "country", "signup_source", "device_type", "age", "cohort_month",
]


def snapshot_date(events: pd.DataFrame) -> pd.Timestamp:
    """Date de l'extraction des données = dernier événement observé."""
    return events["event_date"].max()


def build(subs: pd.DataFrame, users: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    df = subs.merge(users, on="user_id", how="left", validate="one_to_one")

    df = df.rename(columns={
        "start_date": "subscription_start_date",
        "end_date": "subscription_end_date",
    })
    df["is_active"] = (df["status"] == "active").astype(int)
    # Ancienneté : jusqu'à la résiliation pour les churns, jusqu'à la date
    # d'extraction pour les abonnements encore actifs.
    tenure_end = df["subscription_end_date"].fillna(snapshot)
    df["tenure_days"] = (tenure_end - df["subscription_start_date"]).dt.days
    df["cohort_month"] = df["subscription_start_date"].dt.to_period("M").astype(str)

    return df[COLUMNS]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    users = pd.read_csv(CLEAN_DIR / "users_clean.csv", parse_dates=["created_at"])
    subs = pd.read_csv(
        CLEAN_DIR / "subscriptions_clean.csv",
        parse_dates=["start_date", "end_date", "last_payment_date"],
    )
    events = pd.read_csv(CLEAN_DIR / "events_clean.csv", parse_dates=["event_date"])

    snapshot = snapshot_date(events)
    df = build(subs, users, snapshot)

    assert df["tenure_days"].ge(0).all()
    assert df.drop(columns=["subscription_end_date"]).notna().all().all()

    df.to_csv(OUT_DIR / "analytics_subscriptions.csv", index=False)
    print(f"analytics_subscriptions.csv : {len(df)} lignes "
          f"(snapshot {snapshot.date()}, {df['is_active'].sum()} actifs, "
          f"{(1 - df['is_active']).sum()} résiliés)")


if __name__ == "__main__":
    main()

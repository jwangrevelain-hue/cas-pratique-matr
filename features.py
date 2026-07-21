"""Construction des features de churn, partagée par train.py et predict.py.

Règle anti-fuite : toutes les features sont calculables à la date
d'observation T. Les événements sont filtrés sur event_date <= T, et les
colonnes qui encodent l'avenir ou l'état au moment de l'extraction
(status, end_date, total_revenue, last_payment_date, tenure "finale") sont
exclues — voir LEAKY_COLUMNS.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data_clean"

# Colonnes interdites comme features : connues seulement après T ou
# encodant directement la cible.
LEAKY_COLUMNS = ["status", "end_date", "total_revenue", "last_payment_date"]

NUMERIC_FEATURES = [
    "days_since_last_login",
    "logins_30d",
    "logins_60d",
    "logins_90d",
    "usage_trend",
    "logins_per_week_lifetime",
    "days_since_last_payment",
    "payment_overdue_ratio",
    "payment_gap_std",
    "tenure_days_at_T",
    "monthly_price",
    "age",
    "auto_renew",
]
CATEGORICAL_FEATURES = [
    "plan_name",
    "billing_cycle",
    "country",
    "signup_source",
    "device_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CYCLE_DAYS = {"monthly": 30, "yearly": 365}


def load_clean_tables():
    users = pd.read_csv(CLEAN_DIR / "users_clean.csv", parse_dates=["created_at"])
    subs = pd.read_csv(
        CLEAN_DIR / "subscriptions_clean.csv",
        parse_dates=["start_date", "end_date", "last_payment_date"],
    )
    events = pd.read_csv(CLEAN_DIR / "events_clean.csv", parse_dates=["event_date"])
    return users, subs, events


def active_at(subs: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Abonnements actifs à T : commencés avant T et pas encore résiliés à T."""
    mask = (subs["start_date"] <= T) & (subs["end_date"].isna() | (subs["end_date"] > T))
    return subs[mask]


def _count_between(ev: pd.DataFrame, T: pd.Timestamp, days: int) -> pd.Series:
    window = ev[ev["event_date"] > T - pd.Timedelta(days=days)]
    return window.groupby("user_id").size()


def build_features(
    subs: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    T: pd.Timestamp,
) -> pd.DataFrame:
    """Features à la date T pour les abonnements actifs à T (1 ligne / user)."""
    pop = active_at(subs, T).drop(columns=LEAKY_COLUMNS)
    df = pop.merge(users, on="user_id", how="left", validate="one_to_one")
    df = df.set_index("user_id")

    df["tenure_days_at_T"] = (T - df["start_date"]).dt.days
    df["auto_renew"] = df["auto_renew"].astype(int)

    ev = events[events["event_date"] <= T]  # ANTI-FUITE : rien après T
    logins = ev[ev["event_type"] == "login"]
    payments = ev[ev["event_type"] == "payment"]

    # --- Usage : récence, fréquence, tendance -------------------------------
    df["days_since_last_login"] = (T - logins.groupby("user_id")["event_date"].max()).dt.days
    for d in (30, 60, 90):
        df[f"logins_{d}d"] = _count_between(logins, T, d)
    df[["logins_30d", "logins_60d", "logins_90d"]] = (
        df[["logins_30d", "logins_60d", "logins_90d"]].fillna(0)
    )
    # Tendance : activité des 30 derniers jours vs les 30 précédents
    prior_30 = df["logins_60d"] - df["logins_30d"]
    df["usage_trend"] = (df["logins_30d"] + 1) / (prior_30 + 1)
    total_logins = logins.groupby("user_id").size().reindex(df.index).fillna(0)
    df["logins_per_week_lifetime"] = total_logins / (df["tenure_days_at_T"] / 7).clip(lower=1)

    # --- Paiements : récence et régularité -----------------------------------
    df["days_since_last_payment"] = (T - payments.groupby("user_id")["event_date"].max()).dt.days
    # Récence normalisée par le cycle : 1.0 = un cycle entier sans payer
    cycle = df["billing_cycle"].map(CYCLE_DAYS)
    df["payment_overdue_ratio"] = df["days_since_last_payment"] / cycle
    gaps = payments.sort_values("event_date").groupby("user_id")["event_date"].diff().dt.days
    df["payment_gap_std"] = gaps.groupby(payments["user_id"]).std()

    # Jamais connecté / jamais payé depuis la souscription : récence = ancienneté
    for col in ("days_since_last_login", "days_since_last_payment"):
        df[col] = df[col].fillna(df["tenure_days_at_T"])
    df["payment_overdue_ratio"] = df["payment_overdue_ratio"].fillna(
        df["days_since_last_payment"] / cycle
    )
    df["payment_gap_std"] = df["payment_gap_std"].fillna(0)

    out = df[FEATURES + ["start_date"]].copy()  # start_date : sert au split temporel
    assert not out[FEATURES].isna().any().any(), "features avec valeurs manquantes"
    return out


def build_target(
    subs: pd.DataFrame,
    events: pd.DataFrame,
    T: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.Series:
    """churn_90d pour les actifs à T.

    churn = 1 si, dans la fenêtre ]T, T+90j] :
      - l'abonnement est résilié (end_date dans la fenêtre), OU
      - l'utilisateur en cycle mensuel cesse de payer (aucun événement
        payment dans la fenêtre, alors que ~3 échéances y tombent).
    L'absence de paiement d'un cycle annuel n'est pas un signal : son
    échéance peut légitimement tomber hors fenêtre.
    """
    H = T + pd.Timedelta(days=horizon_days)
    pop = active_at(subs, T).set_index("user_id")

    cancelled = pop["end_date"].notna() & (pop["end_date"] <= H)

    pay_window = events[
        (events["event_type"] == "payment")
        & (events["event_date"] > T)
        & (events["event_date"] <= H)
    ]
    paid = pop.index.isin(pay_window["user_id"])
    stopped_paying = (pop["billing_cycle"] == "monthly") & ~paid & ~cancelled

    target = (cancelled | stopped_paying).astype(int).rename("churn_90d")
    print(f"  cible à T={T.date()} : {len(target)} actifs, "
          f"{int(cancelled.sum())} résiliations + {int(stopped_paying.sum())} arrêts "
          f"de paiement silencieux = {target.mean():.1%} de churn")
    return target

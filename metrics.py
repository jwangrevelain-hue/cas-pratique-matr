"""Partie 3 — Métriques analytiques.

Produit, depuis outputs/analytics_subscriptions.csv :
  - outputs/metrics_monthly.csv   (actifs, flux, MRR, revenu cumulé par mois)
  - outputs/metrics_by_plan.csv
  - outputs/metrics_by_channel.csv
  - outputs/metrics_cohorts.csv   (rétention par cohorte mensuelle)
  - outputs/metrics_summary.md    (lecture des chiffres clés)

Conventions : un abonnement est compté actif un mois donné s'il est actif au
dernier jour du mois. Le MRR normalise les cycles annuels en équivalent
mensuel (monthly_price). Le revenu mensuel est approximé par le MRR du mois
(granularité mensuelle, prix constants) ; le revenu cumulé est sa somme.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def load() -> pd.DataFrame:
    return pd.read_csv(
        OUT_DIR / "analytics_subscriptions.csv",
        parse_dates=["subscription_start_date", "subscription_end_date",
                     "last_payment_date"],
    )


def last_full_month(df: pd.DataFrame) -> pd.Period:
    """Dernier mois entièrement observé, déduit des dates les plus récentes."""
    snapshot = max(df["subscription_end_date"].max(), df["last_payment_date"].max())
    period = snapshot.to_period("M")
    return period if snapshot == period.to_timestamp(how="end").normalize() else period - 1


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    start = df["subscription_start_date"]
    end = df["subscription_end_date"]
    # On s'arrête au dernier mois complet (le snapshot tombe en cours de mois)
    months = pd.period_range(start.min().to_period("M"), last_full_month(df), freq="M")

    rows = []
    for m in months:
        month_end = m.to_timestamp(how="end").normalize()
        active_mask = (start <= month_end) & (end.isna() | (end > month_end))
        rows.append({
            "month": str(m),
            "active_subscribers": int(active_mask.sum()),
            "new_subscriptions": int((start.dt.to_period("M") == m).sum()),
            "cancellations": int((end.dt.to_period("M") == m).sum()),
            "mrr": round(df.loc[active_mask, "monthly_price"].sum(), 2),
        })
    out = pd.DataFrame(rows)
    out["net_change"] = out["new_subscriptions"] - out["cancellations"]
    out["cumulative_revenue"] = out["mrr"].cumsum().round(2)
    return out


def by_plan(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("plan_name").agg(
        subscriptions=("subscription_id", "count"),
        active=("is_active", "sum"),
        cancelled=("is_active", lambda s: int((1 - s).sum())),
        mrr_active=("monthly_price", lambda s: round(s[df.loc[s.index, "is_active"] == 1].sum(), 2)),
        avg_tenure_days=("tenure_days", "mean"),
        total_revenue=("total_revenue", "sum"),
    ).reset_index()
    g["churn_rate"] = (g["cancelled"] / g["subscriptions"]).round(3)
    g["mrr_share"] = (g["mrr_active"] / g["mrr_active"].sum()).round(3)
    return g.round(2)


def by_channel(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("signup_source").agg(
        subscriptions=("subscription_id", "count"),
        active=("is_active", "sum"),
        mrr_active=("monthly_price", lambda s: round(s[df.loc[s.index, "is_active"] == 1].sum(), 2)),
        avg_revenue_per_sub=("total_revenue", "mean"),
    ).reset_index()
    g["churn_rate"] = (1 - g["active"] / g["subscriptions"]).round(3)
    return g.round(2)


def cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    """% de chaque cohorte mensuelle encore actif N mois après la souscription."""
    start = df["subscription_start_date"]
    end_period = df["subscription_end_date"].dt.to_period("M")
    start_period = start.dt.to_period("M")
    # Durée de vie en mois : jusqu'à la résiliation, sinon censurée au snapshot
    lifetime = (end_period - start_period).map(lambda d: d.n if pd.notna(d) else None)
    snapshot_period = last_full_month(df)
    max_offset = (snapshot_period - start_period.min()).n  # profondeur observable

    rows = {}
    for cohort, grp in df.groupby(start_period):
        observed = (snapshot_period - cohort).n  # offsets observables (censure)
        size = len(grp)
        life = lifetime.loc[grp.index]
        row = {"cohort_size": size}
        for k in range(0, max_offset + 1):
            if k > observed:
                row[f"m{k}"] = None  # cohorte trop récente : non observable
            else:
                still = ((life.isna()) | (life >= k)).sum()
                row[f"m{k}"] = round(still / size, 3)
        rows[str(cohort)] = row
    return pd.DataFrame(rows).T.rename_axis("cohort_month").reset_index()


def summary(monthly: pd.DataFrame, plans: pd.DataFrame, channels: pd.DataFrame) -> str:
    last = monthly.iloc[-1]
    last3 = monthly.tail(3)
    churn_q = last3["cancellations"].sum() / monthly.iloc[-4]["active_subscribers"]
    top_plan = plans.sort_values("mrr_active", ascending=False).iloc[0]
    lines = [
        "# Lecture des chiffres clés",
        "",
        f"- Base active : **{last['active_subscribers']:.0f} abonnés** fin "
        f"{last['month']}, pour un **MRR de {last['mrr']:,.0f} €**. La base a crû "
        f"régulièrement depuis {monthly.iloc[0]['month']} puis s'est stabilisée : "
        f"les nouvelles souscriptions se sont taries début 2026 tandis que les "
        f"résiliations sont toutes concentrées sur {monthly.tail(3)['month'].iloc[0]}"
        f"–{last['month']} ({last3['cancellations'].sum():.0f} départs, soit "
        f"~{churn_q:.1%} de la base sur le trimestre).",
        f"- **{top_plan['plan_name']}** porte {top_plan['mrr_share']:.0%} du MRR avec "
        f"seulement {top_plan['active']:.0f} actifs : la valeur est concentrée sur "
        f"le haut de gamme, alors que Basic domine en volume.",
        f"- Le churn est du même ordre sur tous les canaux d'acquisition "
        f"({channels['churn_rate'].min():.0%}–{channels['churn_rate'].max():.0%}) : "
        f"pas de canal 'toxique' évident, la rétention se joue ailleurs (usage).",
        f"- Revenu cumulé reconnu depuis l'origine : "
        f"**{last['cumulative_revenue']:,.0f} €**.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    monthly = monthly_metrics(df)
    plans = by_plan(df)
    channels = by_channel(df)
    cohorts = cohort_table(df)

    monthly.to_csv(OUT_DIR / "metrics_monthly.csv", index=False)
    plans.to_csv(OUT_DIR / "metrics_by_plan.csv", index=False)
    channels.to_csv(OUT_DIR / "metrics_by_channel.csv", index=False)
    cohorts.to_csv(OUT_DIR / "metrics_cohorts.csv", index=False)

    text = summary(monthly, plans, channels)
    (OUT_DIR / "metrics_summary.md").write_text(text)
    print(text)
    print(monthly.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()

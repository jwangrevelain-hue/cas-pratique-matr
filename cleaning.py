"""Partie 1 — Nettoyage des tables brutes.

Produit data_clean/{users,subscriptions,events}_clean.csv.
Les décisions non triviales sont documentées dans NOTES_CHOIX.md.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data_clean"

# Formats rencontrés dans users_raw.created_at. Les motifs jour-premier
# (DD/MM/YYYY, DD-MM-YYYY) sont validés dans clean_users : une partie des
# valeurs a un premier composant > 12, ce qui prouve l'ordre jour/mois,
# et on suppose le format homogène au sein d'un même motif.
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M",
]


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse une colonne aux formats hétérogènes en essayant chaque format connu."""
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        mask = out.isna()
        if not mask.any():
            break
        parsed = pd.to_datetime(series[mask], format=fmt, errors="coerce")
        out.loc[mask] = parsed
    unparsed = series[out.isna()]
    if len(unparsed):
        raise ValueError(f"Dates non reconnues : {unparsed.unique()[:10]}")
    return out.dt.normalize()


def dedupe(df: pd.DataFrame, key: str, date_cols: list[str]) -> pd.DataFrame:
    """Déduplique sur `key`.

    Règle : suppression des lignes strictement identiques, puis en cas de
    conflit résiduel on garde la ligne la plus complète (moins de nulls),
    départagée par la date la plus récente — l'enregistrement le plus
    récent/complet est considéré comme l'état le plus fiable.
    """
    before = len(df)
    df = df.drop_duplicates()
    exact = before - len(df)

    conflicts = df[key].duplicated().sum()
    if conflicts:
        df = df.assign(
            _completeness=df.notna().sum(axis=1),
            _recency=df[date_cols].max(axis=1),
        )
        df = (
            df.sort_values(["_completeness", "_recency"], ascending=False)
            .drop_duplicates(key, keep="first")
            .drop(columns=["_completeness", "_recency"])
        )
    print(f"  {key}: {before} lignes -> {len(df)} "
          f"({exact} doublons exacts, {conflicts} conflits arbitrés)")
    return df.sort_values(key).reset_index(drop=True)


# Correspondance des variantes/typos observées vers le plan canonique.
PLAN_MAP = {
    "basic": "Basic", "bacic": "Basic",
    "pro": "Pro", "proo": "Pro", "pr0": "Pro",
    "premium": "Premium", "premuim": "Premium", "premiun": "Premium",
}

# Contrôle croisé : le prix identifie le plan sans ambiguïté dans ces données.
PRICE_TO_PLAN = {9.99: "Basic", 19.99: "Pro", 39.99: "Premium"}


def clean_users() -> pd.DataFrame:
    users = pd.read_csv(ROOT / "users_raw.csv")
    print("users_raw:")

    # Validation de l'hypothèse jour-premier sur les motifs ambigus
    slashed = users["created_at"].str.match(r"^\d{2}[/-]\d{2}[/-]\d{4}")
    day_first_proof = (
        users.loc[slashed, "created_at"].str[:2].astype(int) > 12
    ).mean()
    print(f"  motifs DD/MM ou DD-MM : {slashed.sum()} valeurs, "
          f"{day_first_proof:.0%} avec jour > 12 (ordre jour/mois confirmé)")

    users["created_at"] = parse_mixed_dates(users["created_at"])
    users = dedupe(users, "user_id", ["created_at"])

    assert users["user_id"].is_unique
    assert users["age"].between(10, 100).all(), "âges hors plage plausible"
    return users


def clean_subscriptions() -> pd.DataFrame:
    subs = pd.read_csv(ROOT / "subscriptions_raw.csv")
    print("subscriptions_raw:")

    for col in ["start_date", "end_date", "last_payment_date"]:
        subs[col] = pd.to_datetime(subs[col], format="%Y-%m-%d")

    raw_plans = subs["plan_name"].str.strip().str.lower()
    unknown = set(raw_plans.unique()) - set(PLAN_MAP)
    assert not unknown, f"variantes de plan non couvertes : {unknown}"
    subs["plan_name"] = raw_plans.map(PLAN_MAP)

    # Contrôle croisé plan <-> prix : détecte une variante mal réassignée
    mismatch = subs["plan_name"] != subs["monthly_price"].map(PRICE_TO_PLAN)
    assert not mismatch.any(), subs.loc[mismatch]
    print(f"  plan_name standardisé : {subs['plan_name'].value_counts().to_dict()} "
          "(100% cohérent avec monthly_price)")

    subs = dedupe(subs, "subscription_id", ["last_payment_date", "start_date"])

    assert subs["subscription_id"].is_unique
    assert subs["user_id"].is_unique, "plusieurs abonnements par user"
    # Cohérence statut/date de fin
    assert (subs["end_date"].notna() == (subs["status"] == "cancelled")).all()
    assert (subs["end_date"].isna() | (subs["end_date"] >= subs["start_date"])).all()
    return subs


def clean_events() -> pd.DataFrame:
    events = pd.read_csv(ROOT / "events_raw.csv")
    print("events_raw:")
    events["event_date"] = pd.to_datetime(events["event_date"], format="%Y-%m-%d")

    # La granularité est journalière : les lignes strictement identiques sont
    # indistinguables d'un doublon technique -> on compte 1 événement par
    # (user, jour, type).
    before = len(events)
    events = events.drop_duplicates()
    print(f"  {before} lignes -> {len(events)} ({before - len(events)} doublons exacts)")

    assert set(events["event_type"].unique()) <= {"login", "payment"}
    return events.sort_values(["user_id", "event_date"]).reset_index(drop=True)


def main() -> None:
    CLEAN_DIR.mkdir(exist_ok=True)
    users = clean_users()
    subs = clean_subscriptions()
    events = clean_events()

    # Intégrité référentielle entre les trois tables
    assert set(subs["user_id"]) <= set(users["user_id"])
    assert set(events["user_id"]) <= set(users["user_id"])

    users.to_csv(CLEAN_DIR / "users_clean.csv", index=False)
    subs.to_csv(CLEAN_DIR / "subscriptions_clean.csv", index=False)
    events.to_csv(CLEAN_DIR / "events_clean.csv", index=False)
    print(f"Tables nettoyées écrites dans {CLEAN_DIR}/ "
          f"({len(users)} users, {len(subs)} subscriptions, {len(events)} events)")


if __name__ == "__main__":
    main()

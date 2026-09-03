from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR
from data import normalize_team

FD_BASE = "https://www.football-data.co.uk/mmz4281"
DIV = "SP1"


def _season_code(year: int) -> str:
    yy = year % 100
    nn = (year + 1) % 100
    return f"{yy:02d}{nn:02d}"


FD_TEAM_ALIASES = {
    "Ath Bilbao": "Bilbao",
    "Ath Madrid": "Atletico",
    "Alaves": "Alaves",
    "Barcelona": "Barcelona",
    "Betis": "Real Betis",
    "Cadiz": "Cadiz",
    "Celta": "Celta Vigo",
    "Cordoba": "Cordoba",
    "Eibar": "Eibar",
    "Elche": "Elche",
    "Espanol": "Espanyol",
    "Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Granada": "Granada",
    "Huesca": "Huesca",
    "La Coruna": "Deportivo De La Coruna",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "Levante": "Levante",
    "Malaga": "Malaga",
    "Mallorca": "Mallorca",
    "Osasuna": "Osasuna",
    "Oviedo": "Oviedo",
    "Real Madrid": "Real Madrid",
    "Sevilla": "Sevilla",
    "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting Gijon",
    "Valencia": "Valencia",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Vallecano",
    "Villarreal": "Villarreal",
    "Zaragoza": "Zaragoza",
    "Almeria": "Almeria",
}


def _norm_fd(name: str) -> str:
    if not isinstance(name, str):
        return name
    n = name.strip()
    if n in FD_TEAM_ALIASES:
        return FD_TEAM_ALIASES[n]
    return normalize_team(n)


def _fetch_season(year: int) -> pd.DataFrame:
    code = _season_code(year)
    cache = DATA_DIR / f"fd_SP1_{code}.csv"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        url = f"{FD_BASE}/{code}/{DIV}.csv"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        raw = r.content
        cache.write_bytes(raw)

    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    except Exception:
        return pd.DataFrame()

    cols_keep = ["Date", "HomeTeam", "AwayTeam"]
    for c in ["PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA", "PSH", "PSD", "PSA", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"]:
        if c in df.columns:
            cols_keep.append(c)
    df = df[[c for c in cols_keep if c in df.columns]].copy()
    df["season"] = year

    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    df["home"] = df["HomeTeam"].map(_norm_fd)
    df["away"] = df["AwayTeam"].map(_norm_fd)
    df = df.drop(columns=["Date", "HomeTeam", "AwayTeam"])
    return df


def fetch_odds(seasons: range) -> pd.DataFrame:
    frames = [_fetch_season(y) for y in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_market_odds(row: pd.Series) -> tuple[float, float, float] | None:
    """Prefer Pinnacle closing (PSCH/D/A) > market avg closing > Pinnacle live > avg live > B365."""
    for h, d, a in [
        ("PSCH", "PSCD", "PSCA"),
        ("AvgCH", "AvgCD", "AvgCA"),
        ("PSH", "PSD", "PSA"),
        ("AvgH", "AvgD", "AvgA"),
        ("B365H", "B365D", "B365A"),
    ]:
        if h in row and d in row and a in row:
            hv, dv, av = row.get(h), row.get(d), row.get(a)
            if all(isinstance(x, (int, float)) and pd.notna(x) and x > 1 for x in [hv, dv, av]):
                return float(hv), float(dv), float(av)
    return None


def devig(h: float, d: float, a: float) -> tuple[float, float, float]:
    ih, idr, ia = 1 / h, 1 / d, 1 / a
    s = ih + idr + ia
    return ih / s, idr / s, ia / s


def add_devigged_market_probs(played_with_odds: pd.DataFrame) -> pd.DataFrame:
    df = played_with_odds.copy()
    probs = df.apply(
        lambda r: devig(*best_market_odds(r)) if best_market_odds(r) else (None, None, None),
        axis=1,
        result_type="expand",
    )
    df["mkt_p_h"], df["mkt_p_d"], df["mkt_p_a"] = probs[0], probs[1], probs[2]
    return df


if __name__ == "__main__":
    df = fetch_odds(range(2005, 2026))
    print(f"rows: {len(df)}")
    print(f"cols: {list(df.columns)}")
    if not df.empty:
        with_probs = add_devigged_market_probs(df)
        matched = with_probs["mkt_p_h"].notna().sum()
        print(f"rows with usable market probs: {matched}")
        print(with_probs[["season", "date", "home", "away", "mkt_p_h", "mkt_p_d", "mkt_p_a"]].tail(5))

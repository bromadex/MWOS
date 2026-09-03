from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from config import (
    CURRENT_SEASON,
    DATA_DIR,
    FIRST_XG_SEASON,
    OPENFOOTBALL_REPO,
    UNDERSTAT_LEAGUE,
)

UA = {"User-Agent": "Mozilla/5.0 kalshi-football-model"}


def _season_slug(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _fetch_openfootball_season(year: int, division: str = "es.1") -> pd.DataFrame:
    slug = _season_slug(year)
    url = f"{OPENFOOTBALL_REPO}/{slug}/{division}.json"
    div_tag = division.replace(".", "")
    cache = DATA_DIR / f"of_{div_tag}_{slug}.json" if division != "es.1" else DATA_DIR / f"of_{slug}.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        raw = r.json()
        cache.write_text(json.dumps(raw), encoding="utf-8")

    rows = []
    for md in raw.get("matches", []):
        date = md.get("date")
        team1 = md.get("team1")
        team2 = md.get("team2")
        score_raw = md.get("score")
        home_g = away_g = None
        ft = None
        if isinstance(score_raw, dict):
            ft = score_raw.get("ft")
        elif isinstance(score_raw, list):
            ft = score_raw
        if ft and len(ft) == 2 and ft[0] is not None and ft[1] is not None:
            try:
                home_g, away_g = int(ft[0]), int(ft[1])
            except (TypeError, ValueError):
                pass
        rows.append(
            {
                "season": year,
                "date": date,
                "home": team1,
                "away": team2,
                "home_goals": home_g,
                "away_goals": away_g,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def fetch_openfootball(seasons: range) -> pd.DataFrame:
    frames = [_fetch_openfootball_season(y) for y in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_euro_fixtures(seasons: range) -> pd.DataFrame:
    """Fetch Champions League + Europa League fixtures from openfootball.
    Returns rows of (date, team, comp) — one row per team per fixture.
    Missing seasons are silently skipped."""
    rows = []
    for div, tag in (("cl", "cl"), ("el", "el")):
        for y in seasons:
            df = _fetch_openfootball_season(y, division=div)
            if df.empty:
                continue
            for _, r in df.iterrows():
                if pd.isna(r["date"]):
                    continue
                for team_col in ("home", "away"):
                    tname = normalize_team(r[team_col]) if isinstance(r[team_col], str) else None
                    if tname:
                        rows.append({"date": r["date"], "team": tname, "comp": tag})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "team", "comp"])


def all_spanish_team_names(seasons: range) -> set[str]:
    """
    Union of every team name that has appeared in La Liga Primera (es.1) or
    La Liga Segunda (es.2) in the given seasons, normalised. Used to widen the
    MWOS PDF parser so cup fixtures (Copa del Rey, Supercopa) involving Segunda
    opponents aren't silently dropped.
    """
    names: set[str] = set()
    for div in ("es.1", "es.2"):
        for y in seasons:
            df = _fetch_openfootball_season(y, division=div)
            if df.empty:
                continue
            names.update(df["home"].dropna().map(normalize_team).unique())
            names.update(df["away"].dropna().map(normalize_team).unique())
    return {n for n in names if isinstance(n, str) and n}


def _understat_season_labels(seasons: range) -> list[str]:
    return [f"{y}-{y + 1}" for y in seasons if y >= FIRST_XG_SEASON]


def fetch_understat(seasons: range) -> pd.DataFrame:
    """Pull xG via soccerdata's Understat reader (has its own caching + browser handling)."""
    labels = _understat_season_labels(seasons)
    if not labels:
        return pd.DataFrame()
    try:
        import soccerdata as sd
        us = sd.Understat(leagues="ESP-La Liga", seasons=labels)
        sched = us.read_schedule()
    except Exception as e:
        print(f"[understat] soccerdata fetch failed: {e}")
        return pd.DataFrame()

    if sched is None or sched.empty:
        return pd.DataFrame()

    df = sched.reset_index()
    df = df[df.get("is_result", True) == True].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["season"] = df["date"].dt.year.where(df["date"].dt.month >= 7, df["date"].dt.year - 1)

    out = df.rename(
        columns={
            "home_team": "home",
            "away_team": "away",
        }
    )[["season", "date", "home", "away", "home_goals", "away_goals", "home_xg", "away_xg"]]
    return out.reset_index(drop=True)


TEAM_ALIASES = {
    "Athletic Club": "Bilbao",
    "Athletic Bilbao": "Bilbao",
    "UD Almería": "Almeria",
    "Almería": "Almeria",
    "Cádiz CF": "Cadiz",
    "Cádiz": "Cadiz",
    "Cadiz CF": "Cadiz",
    "RCD Espanyol": "Espanyol",
    "Getafe CF": "Getafe",
    "Granada CF": "Granada",
    "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes",
    "RCD Mallorca": "Mallorca",
    "CA Osasuna": "Osasuna",
    "Real Madrid CF": "Real Madrid",
    "Valencia CF": "Valencia",
    "SD Eibar": "Eibar",
    "Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "Huesca": "Huesca",
    "Sporting Gijón": "Sporting Gijon",
    "Real Zaragoza": "Zaragoza",
    "Córdoba CF": "Cordoba",
    "Málaga CF": "Malaga",
    "Málaga": "Malaga",
    "Malaga CF": "Malaga",
    "Real Valladolid CF": "Real Valladolid",
    "Atletico Madrid": "Atletico",
    "Atlético Madrid": "Atletico",
    "Club Atlético de Madrid": "Atletico",
    "Real Sociedad de Futbol": "Real Sociedad",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Rayo Vallecano": "Vallecano",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Espanyol Barcelona": "RCD Espanyol",
    "Espanyol": "RCD Espanyol",
    "RCD Espanyol de Barcelona": "RCD Espanyol",
    "FC Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid CF",
    "Sevilla FC": "Sevilla",
    "Valencia": "Valencia CF",
    "Getafe": "Getafe CF",
    "Villarreal CF": "Villarreal",
    "Elche": "Elche CF",
    "Deportivo Alaves": "Alaves",
    "Deportivo Alavés": "Alaves",
    "CD Alavés": "Alaves",
    "CD Alaves": "Alaves",
    "Levante": "Levante UD",
    "Real Betis Balompie": "Real Betis",
    "Real Betis Balompié": "Real Betis",
    "Real Oviedo": "Oviedo",
    "Girona FC": "Girona",
    "Osasuna": "CA Osasuna",
    "Mallorca": "RCD Mallorca",
    "Racing Santander": "Santander",
    "Malaga": "Malaga CF",
    "Málaga CF": "Malaga CF",
    "Málaga": "Malaga CF",
    "Deportivo La Coruna": "Deportivo De La Coruna",
    "Deportivo La Coruña": "Deportivo De La Coruna",
    "RC Celta": "Celta Vigo",
    "RC Celta de Vigo": "Celta Vigo",
    "Real Valladolid CF": "Real Valladolid",
    "CD Leganés": "Leganes",
    "Levante UD": "Levante",
    "Elche CF": "Elche",
    "Villarreal CF": "Villarreal",
}


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip(), name.strip())


def build_matches(with_odds: bool = True) -> pd.DataFrame:
    seasons = range(2012, CURRENT_SEASON + 1)
    of = fetch_openfootball(seasons)
    us = fetch_understat(seasons)
    odds = None
    if with_odds:
        try:
            from odds_history import add_devigged_market_probs, fetch_odds
            raw = fetch_odds(seasons)
            if not raw.empty:
                odds = add_devigged_market_probs(raw)
        except Exception as e:
            print(f"[odds] fetch failed, continuing without: {e}")

    if of.empty:
        raise RuntimeError("openfootball fetch returned no data")

    of["home"] = of["home"].map(normalize_team)
    of["away"] = of["away"].map(normalize_team)
    if not us.empty:
        us["home"] = us["home"].map(normalize_team)
        us["away"] = us["away"].map(normalize_team)
        merged = of.merge(
            us[["season", "date", "home", "away", "home_xg", "away_xg"]],
            on=["season", "date", "home", "away"],
            how="left",
        )
    else:
        merged = of.copy()
        merged["home_xg"] = pd.NA
        merged["away_xg"] = pd.NA

    if odds is not None and not odds.empty:
        merged = merged.merge(
            odds[["season", "date", "home", "away", "mkt_p_h", "mkt_p_d", "mkt_p_a"]],
            on=["season", "date", "home", "away"],
            how="left",
        )
    else:
        merged["mkt_p_h"] = pd.NA
        merged["mkt_p_d"] = pd.NA
        merged["mkt_p_a"] = pd.NA
    return merged


def split_completed_upcoming(df: pd.DataFrame):
    played = df["home_goals"].notna() & df["away_goals"].notna()
    return df[played].copy(), df[~played].copy()


if __name__ == "__main__":
    df = build_matches()
    played, upcoming = split_completed_upcoming(df)
    print(f"total: {len(df)}   played: {len(played)}   upcoming: {len(upcoming)}")
    print(f"xG matched: {played['home_xg'].notna().sum()}")
    print(played.tail(5))

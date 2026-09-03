from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from data import normalize_team

FLOAT_TOKEN = re.compile(r"^\d+\.\d+$")
FIXTURE_HEAD = re.compile(r"^(\d+)\s+(\d{2}/\d{2})\s+(\d{2}:\d{2})\s+(.+)$")

MWOS_TEAM_ALIASES = {
    "REAL SOCIEDAD SAN SE..": "Real Sociedad",
    "REAL SOCIEDAD SAN SEBASTIAN": "Real Sociedad",
    "REAL SOCIEDAD DE FUTBOL": "Real Sociedad",
    "REAL BETIS SEVILLE": "Real Betis",
    "ATHLETIC BILBAO": "Bilbao",
    "ATLETICO MADRID": "Atletico",
    "CLUB ATLETICO DE MADRID": "Atletico",
    "RAYO VALLECANO": "Vallecano",
    "RAYO VALLECANO DE MADRID": "Vallecano",
    "VILLARREAL CF": "Villarreal",
    "RC DEPORTIVO DE LA C..": "Deportivo De La Coruna",
    "RC DEPORTIVO DE LA CORUNA": "Deportivo De La Coruna",
    "VALENCIA CF": "Valencia",
    "FC BARCELONA": "Barcelona",
    "REAL MADRID": "Real Madrid",
    "DEPORTIVO ALAVES": "Alaves",
    "CA OSASUNA": "Osasuna",
    "MALAGA CF": "Malaga",
    "LEVANTE UD": "Levante",
    "ESPANYOL BARCELONA": "Espanyol",
    "RCD ESPANYOL DE BARCELONA": "Espanyol",
    "SEVILLA FC": "Sevilla",
    "UD LAS PALMAS": "Las Palmas",
    "CD LEGANES": "Leganes",
    "GIRONA FC": "Girona",
    "ELCHE CF": "Elche",
    "REAL OVIEDO": "Oviedo",
    "RCD MALLORCA": "Mallorca",
    "RC CELTA DE VIGO": "Celta Vigo",
    "GETAFE CF": "Getafe",
    "REAL VALLADOLID CF": "Real Valladolid",
    "CD ELDENSE": "Eldense",
}


def _norm_mwos(name: str) -> str:
    key = name.strip().upper()
    if key in MWOS_TEAM_ALIASES:
        return MWOS_TEAM_ALIASES[key]
    titled = " ".join(w.capitalize() for w in key.split())
    return normalize_team(titled)


ODDS_COLS = [
    "home_odds", "draw_odds", "away_odds",
    "odds_x2", "odds_12", "odds_1x",
    "odds_under25", "odds_over25",
    "odds_gg", "odds_ng",
]


def _split_teams(tokens: list[str], known_teams: set[str]) -> tuple[str, str] | None:
    """Split token list into home/away by trying every cut point and matching known teams."""
    n = len(tokens)
    if n < 2:
        return None
    for cut in range(1, n):
        left = " ".join(tokens[:cut])
        right = " ".join(tokens[cut:])
        h = _norm_mwos(left)
        a = _norm_mwos(right)
        if h in known_teams and a in known_teams:
            return h, a
    return None


def parse_pdf(path: str | Path, known_teams: set[str], default_year: int | None = None) -> pd.DataFrame:
    reader = PdfReader(str(path))
    if default_year is None:
        default_year = datetime.now().year

    rows = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = FIXTURE_HEAD.match(line)
            if not m:
                continue

            match_id, date_str, time_str, tail = m.groups()
            tokens = tail.split()

            odds = []
            while tokens and FLOAT_TOKEN.match(tokens[-1]):
                odds.insert(0, float(tokens.pop()))

            teams = _split_teams(tokens, known_teams)
            if not teams:
                continue
            home, away = teams

            odds_padded = odds + [None] * (len(ODDS_COLS) - len(odds))
            odds_padded = odds_padded[: len(ODDS_COLS)]

            try:
                d, mo = date_str.split("/")
                dt = pd.Timestamp(year=default_year, month=int(mo), day=int(d))
            except Exception:
                continue

            rows.append(
                {
                    "match_id": match_id,
                    "date": dt,
                    "kickoff": time_str,
                    "home": home,
                    "away": away,
                    **dict(zip(ODDS_COLS, odds_padded)),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "home", "away"]).reset_index(drop=True)
    return df


def devig_1x2(h: float, d: float, a: float) -> tuple[float, float, float] | None:
    if not all(x and x > 1.0 for x in [h, d, a]):
        return None
    ih, idr, ia = 1 / h, 1 / d, 1 / a
    s = ih + idr + ia
    return ih / s, idr / s, ia / s


if __name__ == "__main__":
    import sys

    from data import build_matches, split_completed_upcoming

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\BRAVURA\Downloads\MWOS-DAILYFIXTURE (1).pdf"
    print(f"loading match data for team roster...")
    df_matches = build_matches()
    played, _ = split_completed_upcoming(df_matches)
    teams = set(played["home"].dropna().unique()) | set(played["away"].dropna().unique())
    print(f"parsing {pdf_path}...")
    fx = parse_pdf(pdf_path, teams)
    print(f"parsed {len(fx)} fixtures matching known teams")
    if not fx.empty:
        print(fx.head(20).to_string(index=False))

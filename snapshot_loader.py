from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import ROOT

SNAPSHOT_DIR = ROOT / "snapshot"


def load_snapshot() -> tuple[dict, set[str], set[str], dict, dict] | None:
    s_path = SNAPSHOT_DIR / "strengths.json"
    r_path = SNAPSHOT_DIR / "recent_teams.json"
    k_path = SNAPSHOT_DIR / "known_teams.json"
    c_path = SNAPSHOT_DIR / "fixture_context.json"
    m_path = SNAPSHOT_DIR / "meta.json"
    if not (s_path.exists() and r_path.exists()):
        return None

    strengths = json.loads(s_path.read_text(encoding="utf-8"))
    if "ref_date" in strengths and isinstance(strengths["ref_date"], str):
        try:
            strengths["ref_date"] = pd.Timestamp(strengths["ref_date"])
        except Exception:
            pass

    recent_teams = set(json.loads(r_path.read_text(encoding="utf-8")))
    known_teams = set(json.loads(k_path.read_text(encoding="utf-8"))) if k_path.exists() else set(recent_teams)
    fixture_context = json.loads(c_path.read_text(encoding="utf-8")) if c_path.exists() else {}
    meta = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else {}
    return strengths, recent_teams, known_teams, fixture_context, meta


def context_for(fixture_context: dict, home: str, away: str, fixture_date) -> dict:
    """Build the {home_rest_days, away_rest_days, home_euro_midweek, away_euro_midweek}
    dict from the fixture_context snapshot for a single fixture."""
    fx_ts = pd.Timestamp(fixture_date).normalize()
    out: dict = {}
    for role, team in (("home", home), ("away", away)):
        entries = fixture_context.get(team, [])
        dates = [pd.Timestamp(e["date"]).normalize() for e in entries]
        comps = [e["comp"] for e in entries]
        prior = [(d, c) for d, c in zip(dates, comps) if d < fx_ts]
        if prior:
            last_d, _ = max(prior, key=lambda x: x[0])
            out[f"{role}_rest_days"] = int((fx_ts - last_d).days)
        else:
            out[f"{role}_rest_days"] = None
        window_lo = fx_ts - pd.Timedelta(days=4)
        out[f"{role}_euro_midweek"] = any(
            (window_lo <= d < fx_ts) and c in ("cl", "el") for d, c in zip(dates, comps)
        )
        out[f"{role}_matches_14d"] = sum(
            1 for d in dates if fx_ts - pd.Timedelta(days=14) <= d < fx_ts
        )
    return out

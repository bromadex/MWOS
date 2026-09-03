from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    MIN_RECENT_MATCHES,
    RECENCY_HALF_LIFE_DAYS,
    RECENT_WINDOW_DAYS,
    SHRINKAGE_MATCHES,
    STRENGTH_CEILING,
    STRENGTH_FLOOR,
)


def recent_top_flight_teams(
    played: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    window_days: int = RECENT_WINDOW_DAYS,
    min_matches: int = MIN_RECENT_MATCHES,
) -> set[str]:
    """Teams with >= min_matches in the top-flight dataset within the last `window_days`."""
    if ref_date is None:
        ref_date = pd.Timestamp(datetime.now(timezone.utc).date())
    cutoff = ref_date - pd.Timedelta(days=window_days)
    recent = played[played["date"] >= cutoff]
    home_ct = recent["home"].value_counts()
    away_ct = recent["away"].value_counts()
    total = home_ct.add(away_ct, fill_value=0)
    return set(total[total >= min_matches].index)


def _weights(dates: pd.Series, ref: pd.Timestamp, half_life: float) -> np.ndarray:
    age = (ref - dates).dt.total_seconds().to_numpy() / 86400.0
    age = np.clip(age, 0, None)
    return np.exp(-np.log(2) * age / half_life)


def _shrunk(numer_sum: float, denom_sum: float, league_mean: float, k: float) -> float:
    return (numer_sum + k * league_mean) / (denom_sum + k)


def _cap(x: float) -> float:
    return float(np.clip(x, STRENGTH_FLOOR, STRENGTH_CEILING))


def compute_team_strengths(
    played: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    half_life: float = RECENCY_HALF_LIFE_DAYS,
) -> dict:
    if ref_date is None:
        ref_date = pd.Timestamp(datetime.now(timezone.utc).date())

    df = played.copy()
    df = df.dropna(subset=["home_goals", "away_goals", "date"])
    df["home_xg"] = pd.to_numeric(df.get("home_xg"), errors="coerce")
    df["away_xg"] = pd.to_numeric(df.get("away_xg"), errors="coerce")
    df["w"] = _weights(df["date"], ref_date, half_life)

    total_w = df["w"].sum()
    if total_w == 0:
        raise RuntimeError("no weighted matches")

    league_mean_goals_home = (df["w"] * df["home_goals"]).sum() / total_w
    league_mean_goals_away = (df["w"] * df["away_goals"]).sum() / total_w
    league_mean_goals = 0.5 * (league_mean_goals_home + league_mean_goals_away)
    home_adv_goals = league_mean_goals_home / max(league_mean_goals_away, 1e-6)

    xg_mask = df["home_xg"].notna() & df["away_xg"].notna()
    xg_df = df[xg_mask]
    if len(xg_df) > 0:
        w_xg = xg_df["w"].sum()
        league_mean_xg_home = (xg_df["w"] * xg_df["home_xg"]).sum() / w_xg
        league_mean_xg_away = (xg_df["w"] * xg_df["away_xg"]).sum() / w_xg
        league_mean_xg = 0.5 * (league_mean_xg_home + league_mean_xg_away)
        home_adv_xg = league_mean_xg_home / max(league_mean_xg_away, 1e-6)
    else:
        league_mean_xg = league_mean_goals
        home_adv_xg = home_adv_goals

    home_rows = df.rename(
        columns={
            "home": "team",
            "away": "opp",
            "home_goals": "gf",
            "away_goals": "ga",
            "home_xg": "xgf",
            "away_xg": "xga",
        }
    )[["team", "opp", "gf", "ga", "xgf", "xga", "w", "date"]]
    home_rows["venue"] = "H"

    away_rows = df.rename(
        columns={
            "away": "team",
            "home": "opp",
            "away_goals": "gf",
            "home_goals": "ga",
            "away_xg": "xgf",
            "home_xg": "xga",
        }
    )[["team", "opp", "gf", "ga", "xgf", "xga", "w", "date"]]
    away_rows["venue"] = "A"

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)

    teams = sorted(long_df["team"].dropna().unique())
    strengths = {}

    for team in teams:
        t = long_df[long_df["team"] == team]
        w = t["w"].to_numpy()
        wsum = w.sum()
        if wsum == 0:
            continue

        gf = (w * t["gf"].to_numpy()).sum()
        ga = (w * t["ga"].to_numpy()).sum()
        att_g_raw = _shrunk(gf, wsum, league_mean_goals, SHRINKAGE_MATCHES)
        def_g_raw = _shrunk(ga, wsum, league_mean_goals, SHRINKAGE_MATCHES)

        xgf = t["xgf"].to_numpy()
        xga = t["xga"].to_numpy()
        xg_mask_t = ~np.isnan(xgf) & ~np.isnan(xga)
        if xg_mask_t.sum() > 0:
            wx = w[xg_mask_t]
            wsum_x = wx.sum()
            xgf_num = (wx * xgf[xg_mask_t]).sum()
            xga_num = (wx * xga[xg_mask_t]).sum()
            att_xg_raw = _shrunk(xgf_num, wsum_x, league_mean_xg, SHRINKAGE_MATCHES)
            def_xg_raw = _shrunk(xga_num, wsum_x, league_mean_xg, SHRINKAGE_MATCHES)
        else:
            att_xg_raw = att_g_raw
            def_xg_raw = def_g_raw

        strengths[team] = {
            "att_g": _cap(att_g_raw / league_mean_goals),
            "def_g": _cap(def_g_raw / league_mean_goals),
            "att_xg": _cap(att_xg_raw / league_mean_xg),
            "def_xg": _cap(def_xg_raw / league_mean_xg),
            "matches_eff": float(wsum),
        }

    return {
        "teams": strengths,
        "league_mean_goals": float(league_mean_goals),
        "league_mean_xg": float(league_mean_xg),
        "home_adv_goals": float(home_adv_goals),
        "home_adv_xg": float(home_adv_xg),
        "ref_date": ref_date,
    }

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

    HOME_FACTOR_SHRINKAGE = 20.0  # matches; heavier shrinkage since home split is noisier

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

        # Per-team home lift factor: how much bigger THIS team's home boost is
        # than the league average. Ratio-of-ratios shrunk toward 1.0.
        home_only = t[t["venue"] == "H"]
        away_only = t[t["venue"] == "A"]
        wh = home_only["w"].to_numpy()
        wa = away_only["w"].to_numpy()
        wh_sum, wa_sum = wh.sum(), wa.sum()

        if wh_sum > 0 and wa_sum > 0:
            gf_h = (wh * home_only["gf"].to_numpy()).sum() / wh_sum
            gf_a = (wa * away_only["gf"].to_numpy()).sum() / wa_sum
            h_i_g = gf_h / max(gf_a, 1e-6)
            hf_g_raw = h_i_g / max(home_adv_goals, 1e-6)

            xgf_h = home_only["xgf"].to_numpy()
            xgf_a = away_only["xgf"].to_numpy()
            mh = ~np.isnan(xgf_h)
            ma = ~np.isnan(xgf_a)
            if mh.sum() > 0 and ma.sum() > 0:
                xgf_h_mean = (wh[mh] * xgf_h[mh]).sum() / max(wh[mh].sum(), 1e-6)
                xgf_a_mean = (wa[ma] * xgf_a[ma]).sum() / max(wa[ma].sum(), 1e-6)
                h_i_xg = xgf_h_mean / max(xgf_a_mean, 1e-6)
                hf_xg_raw = h_i_xg / max(home_adv_xg, 1e-6)
            else:
                hf_xg_raw = hf_g_raw

            eff = min(wh_sum, wa_sum)
            hf_g = (hf_g_raw * eff + 1.0 * HOME_FACTOR_SHRINKAGE) / (eff + HOME_FACTOR_SHRINKAGE)
            hf_xg = (hf_xg_raw * eff + 1.0 * HOME_FACTOR_SHRINKAGE) / (eff + HOME_FACTOR_SHRINKAGE)
            hf_g = float(np.clip(hf_g, 0.70, 1.40))
            hf_xg = float(np.clip(hf_xg, 0.70, 1.40))
        else:
            hf_g = 1.0
            hf_xg = 1.0

        strengths[team] = {
            "att_g": _cap(att_g_raw / league_mean_goals),
            "def_g": _cap(def_g_raw / league_mean_goals),
            "att_xg": _cap(att_xg_raw / league_mean_xg),
            "def_xg": _cap(def_xg_raw / league_mean_xg),
            "hf_g": hf_g,
            "hf_xg": hf_xg,
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

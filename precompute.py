"""
Run this locally (never on Streamlit Cloud) whenever you want to refresh the
model's team ratings. It fetches openfootball + Understat xG + football-data.co.uk
odds, computes team strengths, and writes small JSON files that the web app
loads at cold start. This keeps Chrome / seleniumbase / requests entirely off
the production host.

    python precompute.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import CURRENT_SEASON, ROOT
from data import (
    _fetch_openfootball_season,
    all_spanish_team_names,
    build_matches,
    fetch_euro_fixtures,
    normalize_team,
    split_completed_upcoming,
)
from ratings import compute_team_strengths, recent_top_flight_teams

SNAPSHOT_DIR = ROOT / "snapshot"
SNAPSHOT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("[precompute] loading match data (this pulls from external sources)...")
    df = build_matches()
    played, upcoming = split_completed_upcoming(df)

    xg_matched = int(played["home_xg"].notna().sum())
    mkt_matched = int(played["mkt_p_h"].notna().sum()) if "mkt_p_h" in played.columns else 0
    print(f"[precompute] played={len(played)}  xG matched={xg_matched}  odds matched={mkt_matched}")

    print("[precompute] computing team strengths...")
    strengths = compute_team_strengths(played)

    ref_date = strengths["ref_date"]
    if hasattr(ref_date, "isoformat"):
        ref_date_iso = pd.Timestamp(ref_date).isoformat()
    else:
        ref_date_iso = str(ref_date)

    strengths_out = {
        "teams": strengths["teams"],
        "league_mean_goals": strengths["league_mean_goals"],
        "league_mean_xg": strengths["league_mean_xg"],
        "home_adv_goals": strengths["home_adv_goals"],
        "home_adv_xg": strengths["home_adv_xg"],
        "ref_date": ref_date_iso,
    }

    print("[precompute] computing recent-top-flight team set...")
    recent = sorted(recent_top_flight_teams(played))

    print("[precompute] fetching wider Spanish team roster (es.1 + es.2)...")
    wider = sorted(all_spanish_team_names(range(2012, CURRENT_SEASON + 1)))

    print("[precompute] fetching Euro fixtures (CL + EL) for the current + previous season...")
    euro_seasons = range(max(CURRENT_SEASON - 1, 2012), CURRENT_SEASON + 1)
    euro_df = fetch_euro_fixtures(euro_seasons)
    print(f"[precompute] euro fixture rows: {len(euro_df)}")

    print("[precompute] building fixture-context map (recent 60 days + upcoming 21 days per team)...")
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    ctx_window_past = pd.Timedelta(days=60)
    ctx_window_future = pd.Timedelta(days=21)

    # Combine every source of match dates: played league games + upcoming league fixtures
    # (both es.1 and es.2) + Euro fixtures.
    all_rows: list[dict] = []
    for _, r in played.iterrows():
        d = r.get("date")
        if pd.isna(d):
            continue
        for tcol in ("home", "away"):
            if isinstance(r.get(tcol), str):
                all_rows.append({"team": r[tcol], "date": pd.Timestamp(d).normalize(), "comp": "es.1"})
    for _, r in upcoming.iterrows():
        d = r.get("date")
        if pd.isna(d):
            continue
        for tcol in ("home", "away"):
            if isinstance(r.get(tcol), str):
                all_rows.append({"team": r[tcol], "date": pd.Timestamp(d).normalize(), "comp": "es.1"})
    # Segunda schedule
    for y in range(max(CURRENT_SEASON - 1, 2012), CURRENT_SEASON + 1):
        seg = _fetch_openfootball_season(y, division="es.2")
        if seg.empty:
            continue
        seg["home"] = seg["home"].map(normalize_team)
        seg["away"] = seg["away"].map(normalize_team)
        for _, r in seg.iterrows():
            d = r.get("date")
            if pd.isna(d):
                continue
            for tcol in ("home", "away"):
                if isinstance(r.get(tcol), str):
                    all_rows.append({"team": r[tcol], "date": pd.Timestamp(d).normalize(), "comp": "es.2"})
    # Euro fixtures
    for _, r in euro_df.iterrows():
        all_rows.append({"team": r["team"], "date": pd.Timestamp(r["date"]).normalize(), "comp": r["comp"]})

    ctx_df = pd.DataFrame(all_rows)
    if not ctx_df.empty:
        keep = (ctx_df["date"] >= now - ctx_window_past) & (ctx_df["date"] <= now + ctx_window_future)
        ctx_df = ctx_df[keep].drop_duplicates(subset=["team", "date", "comp"])

    fixture_context: dict[str, list[dict]] = {}
    if not ctx_df.empty:
        for team, grp in ctx_df.groupby("team"):
            fixture_context[team] = [
                {"date": d.strftime("%Y-%m-%d"), "comp": c}
                for d, c in sorted(zip(grp["date"], grp["comp"]))
            ]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "played_matches": len(played),
        "xg_matched": xg_matched,
        "market_odds_matched": mkt_matched,
        "n_teams_rated": len(strengths_out["teams"]),
        "n_teams_recent": len(recent),
        "n_teams_known": len(wider),
        "n_teams_with_context": len(fixture_context),
        "context_window_past_days": int(ctx_window_past.days),
        "context_window_future_days": int(ctx_window_future.days),
    }

    (SNAPSHOT_DIR / "strengths.json").write_text(json.dumps(strengths_out, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "recent_teams.json").write_text(json.dumps(recent, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "known_teams.json").write_text(json.dumps(wider, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "fixture_context.json").write_text(json.dumps(fixture_context, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[precompute] wrote snapshot -> {SNAPSHOT_DIR}")
    print(f"[precompute] meta: {meta}")


if __name__ == "__main__":
    main()

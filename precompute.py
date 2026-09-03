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
from data import all_spanish_team_names, build_matches, split_completed_upcoming
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

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "played_matches": len(played),
        "xg_matched": xg_matched,
        "market_odds_matched": mkt_matched,
        "n_teams_rated": len(strengths_out["teams"]),
        "n_teams_recent": len(recent),
        "n_teams_known": len(wider),
    }

    (SNAPSHOT_DIR / "strengths.json").write_text(json.dumps(strengths_out, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "recent_teams.json").write_text(json.dumps(recent, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "known_teams.json").write_text(json.dumps(wider, indent=2), encoding="utf-8")
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[precompute] wrote snapshot -> {SNAPSHOT_DIR}")
    print(f"[precompute] meta: {meta}")


if __name__ == "__main__":
    main()

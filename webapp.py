from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import MODEL_BLEND_WEIGHT, MWOS_INBOX, OUT_DIR
from mwos_edges import compute_market_edges, format_report
from mwos_pdf import parse_pdf
from snapshot_loader import load_snapshot

st.set_page_config(
    page_title="La Liga Value Bets — MWOS",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading precomputed model snapshot…")
def _load_context():
    snap = load_snapshot()
    if snap is not None:
        strengths, recent_teams, known_teams, fixture_context, meta = snap
        return strengths, recent_teams, known_teams, fixture_context, meta, "snapshot"

    from data import all_spanish_team_names, build_matches, split_completed_upcoming
    from ratings import compute_team_strengths, recent_top_flight_teams
    from config import CURRENT_SEASON

    df = build_matches()
    played, _ = split_completed_upcoming(df)
    strengths = compute_team_strengths(played)
    recent_teams = recent_top_flight_teams(played)
    known_teams = all_spanish_team_names(range(2012, CURRENT_SEASON + 1))
    xg_matched = int(played["home_xg"].notna().sum()) if "home_xg" in played.columns else 0
    mkt_matched = int(played["mkt_p_h"].notna().sum()) if "mkt_p_h" in played.columns else 0
    meta = {
        "played_matches": len(played),
        "xg_matched": xg_matched,
        "market_odds_matched": mkt_matched,
        "n_teams_rated": len(strengths["teams"]),
        "n_teams_recent": len(recent_teams),
        "n_teams_known": len(known_teams),
    }
    return strengths, recent_teams, known_teams, {}, meta, "live"


def _tier_color(sig: str) -> str:
    return {
        "STRONG": "#0a7d0a",
        "BUY": "#1e88e5",
        "MARGINAL": "#f9a825",
        "SKIP": "#9e9e9e",
    }.get(sig, "#9e9e9e")


st.title("⚽ La Liga Value-Bet Scanner")
st.caption("Upload today's MWOS-DAILYFIXTURE PDF. The model runs, blends with the de-vigged MWOS line, and lists bets with positive EV.")

with st.sidebar:
    st.header("Settings")

    blend = st.slider(
        "Model weight in blend",
        min_value=0.0,
        max_value=1.0,
        value=MODEL_BLEND_WEIGHT,
        step=0.05,
        help="Blend weight for the model's probability vs. the de-vigged MWOS market probability. 0 = fully trust the market, 1 = fully trust the model. Backtest optimum ≈ 0; safe zone 0.15–0.35.",
    )
    with st.expander("ℹ️ What is this?"):
        st.markdown(
            "The final probability the app uses is a **weighted average** of two things:\n\n"
            "- **The model** — this app's Poisson goals model, built from openfootball + Understat xG.\n"
            "- **The market** — MWOS's own odds, stripped of their vig, treated as an implied probability.\n\n"
            "`p_final = w · p_model + (1 − w) · p_market`\n\n"
            "**Lower w (e.g. 0.15)** = more market-anchored, safer, fewer fake edges.\n\n"
            "**Higher w (e.g. 0.60)** = trust the model more; more bets flagged but they're riskier — "
            "the backtest shows the pure model is ~3% worse than Pinnacle closing, so trusting it a lot is dangerous."
        )

    min_ev = st.slider(
        "Minimum EV per unit",
        min_value=0.0,
        max_value=0.20,
        value=0.02,
        step=0.01,
        format="%.2f",
        help="Threshold to flag a bet as worth showing. EV per unit = model probability × MWOS odds − 1.",
    )
    with st.expander("ℹ️ What is EV?"):
        st.markdown(
            "**Expected value (EV) per 1 unit staked** — how much you'd profit on average if this exact bet "
            "repeated many times, per $1 wagered.\n\n"
            "`EV = p_final × decimal_odds − 1`\n\n"
            "Example: MWOS offers 3.00 on a bet the app thinks has a 40% chance of winning. "
            "EV = 0.40 × 3.00 − 1 = **+0.20**, i.e. +20¢ per $1 staked long-run.\n\n"
            "**Raise the threshold** to hide small edges (0.05 = only serious ones). "
            "**Lower it to 0** to see every positive-EV bet, including tiny ones."
        )

    min_recent = st.number_input(
        "Min recent matches per team (2y)",
        min_value=5,
        max_value=40,
        value=12,
        help="Teams need at least this many matches in the last 2 years to be considered 'recent top-flight' — affects the recent-teams filter used when building the snapshot locally.",
    )
    with st.expander("ℹ️ What does this do?"):
        st.markdown(
            "Some La Liga clubs go down to Segunda, come back up, drop again. Their **rating ages fast**. "
            "This threshold decides how many matches a team needs to have played in the top-flight window "
            "(default: last 2 years) before we call it a 'recent top-flight' team.\n\n"
            "It's baked into the snapshot when you run `precompute.py` locally, so changing it on the live "
            "app doesn't retroactively re-rate teams — it's here for reference and for when you fork the repo.\n\n"
            "**Effect:** more teams pass a low threshold (5) → more matches scored, but with older data. "
            "A high threshold (20+) means only regulars, ratings you can lean on."
        )

    st.divider()
    with st.expander("📖 Signal legend", expanded=False):
        st.markdown(
            "Every scored market is tagged with one signal so you can scan quickly:\n\n"
            "| Badge | Meaning | Trigger |\n"
            "|---|---|---|\n"
            "| 🟢 **STRONG** | Big edge, both teams rated | EV ≥ **8%** |\n"
            "| 🔵 **BUY** | Solid edge, both teams rated | **4%** ≤ EV < 8% |\n"
            "| 🟡 **MARGINAL** | Small edge, both teams rated | min-EV ≤ EV < 4% |\n"
            "| ⚪ **UNRATED** | Cup fixture with a Segunda / lower-tier team the model can't rate | Either team missing from the ratings table |\n"
            "| ⚪ **SKIP** | Below the min-EV threshold — not worth staking | Not shown in the value-bet list |\n\n"
            "**Never stake on 🟢 or 🔵 blindly.** Even a real 8% edge dies if MWOS's overround is 11% on that market "
            "and the model happens to be wrong on this fixture. Use Kelly/4 sizing, spread across many bets, "
            "and never bet the same market twice hoping to correct."
        )
    st.caption("Kelly/4 stake = quarter-Kelly. Multiply by bankroll to size each bet.")

strengths, recent_teams_snapshot, known_teams_snapshot, fixture_context, meta, mode = _load_context()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Played matches", f"{meta.get('played_matches', 0):,}")
c2.metric("xG matched", f"{meta.get('xg_matched', 0):,}")
c3.metric("Historical odds matched", f"{meta.get('market_odds_matched', 0):,}")
c4.metric(
    "Teams (rated / known)",
    f"{meta.get('n_teams_rated', meta.get('n_teams', 0))} / {meta.get('n_teams_known', 0)}",
)
if mode == "snapshot" and "generated_at" in meta:
    st.caption(f"Data snapshot generated at {meta['generated_at']} UTC. Run `python precompute.py` locally to refresh.")
else:
    st.caption("Running with a live data fetch (no snapshot committed).")

st.divider()

st.markdown(
    "**Need today's PDF?** Grab it from MWOS → "
    "[📥 Download MWOS daily fixtures](https://info.betting.co.zw/daily-fixture/)  "
    "then drop it below."
)

uploaded = st.file_uploader(
    "Drop today's MWOS-DAILYFIXTURE PDF here",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Waiting for a PDF. Downloaded PDFs are copied to `mwos_inbox/` and processed.")
    st.stop()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
try:
    pdf_dest = MWOS_INBOX / f"upload_{stamp}_{uploaded.name}"
    pdf_dest.write_bytes(uploaded.getvalue())
except OSError:
    import tempfile
    pdf_dest = Path(tempfile.gettempdir()) / f"upload_{stamp}_{uploaded.name}"
    pdf_dest.write_bytes(uploaded.getvalue())
st.success(f"Received {uploaded.name}")

parser_teams = known_teams_snapshot if known_teams_snapshot else recent_teams_snapshot

with st.spinner("Parsing PDF and matching Spanish football fixtures…"):
    fx = parse_pdf(str(pdf_dest), parser_teams)

if fx.empty:
    st.warning("No Spanish football fixtures with recognised teams were found in this PDF.")
    st.stop()

n_unrated_fx = 0
if not fx.empty:
    rated_set = set(strengths["teams"].keys())
    n_unrated_fx = int(sum(1 for _, r in fx.iterrows() if r["home"] not in rated_set or r["away"] not in rated_set))
    if n_unrated_fx:
        st.info(f"{n_unrated_fx} cup fixture(s) involve a team the model has no rating for. They'll appear flagged ⚪ UNRATED — treat those numbers with extra care.")

st.subheader(f"Parsed fixtures ({len(fx)})")
st.dataframe(
    fx[["date", "kickoff", "home", "away", "home_odds", "draw_odds", "away_odds",
        "odds_1x", "odds_12", "odds_x2", "odds_under25", "odds_over25", "odds_gg", "odds_ng"]],
    use_container_width=True,
    hide_index=True,
)

with st.spinner("Computing blended probabilities and edges…"):
    edges = compute_market_edges(
        fx,
        strengths,
        min_ev=float(min_ev),
        blend_weight=float(blend),
        fixture_context=fixture_context,
    )

if edges.empty:
    st.warning("No markets scored — nothing to bet.")
    st.stop()

rated_edges = edges[edges["both_rated"]].copy()
unrated_edges = edges[~edges["both_rated"]].copy()
keep = rated_edges[rated_edges["ev_per_unit"] >= float(min_ev)].copy()
tiers = keep["signal"].value_counts().to_dict()

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Flagged (rated)", len(keep))
s2.metric("STRONG", tiers.get("STRONG", 0))
s3.metric("BUY", tiers.get("BUY", 0))
s4.metric("MARGINAL", tiers.get("MARGINAL", 0))
s5.metric("Unrated cup", unrated_edges.drop_duplicates(["date", "kickoff", "home", "away"]).shape[0])
st.caption(
    "🟢 **STRONG** = EV ≥ 8%   ·   🔵 **BUY** = EV ≥ 4%   ·   "
    "🟡 **MARGINAL** = EV ≥ your min threshold   ·   ⚪ **UNRATED** = cup match, one team not rated — no EV trusted. "
    "See sidebar → **Signal legend** for the full explanation."
)

st.divider()
st.subheader("Value bets by fixture")

if keep.empty:
    st.info("No fixture cleared the EV threshold. Lower the threshold or raise the model weight.")
else:
    _tier_icon = {"STRONG": "🟢 STRONG", "BUY": "🔵 BUY", "MARGINAL": "🟡 MARGINAL", "SKIP": "⚪ SKIP", "UNRATED": "⚪ UNRATED"}

    def _rest_badge(row) -> str:
        parts = []
        for tag, team in (("H", row["home"]), ("A", row["away"])):
            rd = row.get(f"{'home' if tag=='H' else 'away'}_rest_days")
            euro = row.get(f"{'home' if tag=='H' else 'away'}_euro_midweek")
            m14 = row.get(f"{'home' if tag=='H' else 'away'}_matches_14d")
            bits = []
            if rd is not None:
                bits.append(f"{rd}d rest")
            if m14:
                bits.append(f"{m14} in 14d")
            if euro:
                bits.append("⚠️ UCL/UEL midweek")
            if bits:
                parts.append(f"{tag}: {', '.join(bits)}")
        return "  ·  ".join(parts) if parts else ""

    for (d, k, h, a), grp in keep.groupby(["date", "kickoff", "home", "away"], sort=False):
        overround = grp["overround_1x2"].iloc[0]
        or_txt = f"  ·  MWOS 1X2 overround: {overround:+.1%}" if pd.notna(overround) else ""
        rest_txt = _rest_badge(grp.iloc[0])
        subtitle = f"{or_txt}"
        with st.expander(f"**{d}  {k}   {h}  vs  {a}**{subtitle}", expanded=True):
            if rest_txt:
                st.caption(rest_txt)
            display = grp[["market", "mwos_odds", "p_blend", "p_model", "ev_per_unit", "kelly_quarter", "signal"]].copy()
            display["mwos_odds"] = display["mwos_odds"].map(lambda v: f"{v:.2f}")
            display["p_blend"] = display["p_blend"].map(lambda v: f"{v*100:.1f}%")
            display["p_model"] = display["p_model"].map(lambda v: f"{v*100:.1f}%")
            display["ev_per_unit"] = display["ev_per_unit"].map(lambda v: f"{v*100:+.2f}%")
            display["kelly_quarter"] = display["kelly_quarter"].map(lambda v: f"{v*100:.2f}%")
            display["signal"] = display["signal"].map(lambda s: _tier_icon.get(s, s))
            display.columns = ["Market", "MWOS", "p_blend", "p_model", "EV", "Kelly/4", "Signal"]
            st.dataframe(display, use_container_width=True, hide_index=True)

if not unrated_edges.empty:
    st.divider()
    st.subheader("Unrated cup fixtures")
    st.caption("Model has no rating for one or both teams (typical for Copa del Rey / Supercopa vs. a lower-division side). The raw MWOS odds are shown so you can still see the market, but no EV is trusted.")
    fixtures_view = unrated_edges.drop_duplicates(["date", "kickoff", "home", "away"])[
        ["date", "kickoff", "home", "away", "home_rated", "away_rated"]
    ].copy()
    fixtures_view["home"] = fixtures_view.apply(lambda r: f"{r['home']}" + ("" if r["home_rated"] else "  ⚠️"), axis=1)
    fixtures_view["away"] = fixtures_view.apply(lambda r: f"{r['away']}" + ("" if r["away_rated"] else "  ⚠️"), axis=1)
    st.dataframe(
        fixtures_view[["date", "kickoff", "home", "away"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Download full outputs")

report_txt = format_report(edges, min_ev=float(min_ev), blend_weight=float(blend))
edges_csv = edges.to_csv(index=False).encode("utf-8")
fx_csv = fx.to_csv(index=False).encode("utf-8")

d1, d2, d3 = st.columns(3)
d1.download_button("📄 Report (.txt)", data=report_txt, file_name=f"mwos_report_{stamp}.txt", mime="text/plain")
d2.download_button("📊 Edges (.csv)", data=edges_csv, file_name=f"mwos_edges_{stamp}.csv", mime="text/csv")
d3.download_button("📅 Fixtures (.csv)", data=fx_csv, file_name=f"mwos_fixtures_{stamp}.csv", mime="text/csv")

try:
    (OUT_DIR / f"mwos_report_{stamp}.txt").write_text(report_txt, encoding="utf-8")
    edges.to_csv(OUT_DIR / f"mwos_edges_{stamp}.csv", index=False)
    fx.to_csv(OUT_DIR / f"mwos_fixtures_{stamp}.csv", index=False)
except OSError:
    pass

with st.expander("Text report"):
    st.code(report_txt, language="text")

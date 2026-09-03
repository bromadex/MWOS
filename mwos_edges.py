from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODEL_BLEND_WEIGHT
from model import predict_fixture


def _ev(prob: float, dec_odds: float) -> float:
    return prob * dec_odds - 1


def _kelly(prob: float, dec_odds: float) -> float:
    b = dec_odds - 1
    if b <= 0:
        return 0.0
    k = (prob * b - (1 - prob)) / b
    return max(k, 0.0)


def _devig_pair(a: float, b: float) -> tuple[float, float] | None:
    if not (a and b and a > 1 and b > 1):
        return None
    ia, ib = 1 / a, 1 / b
    s = ia + ib
    return ia / s, ib / s


def _devig_triple(h: float, d: float, a: float) -> tuple[float, float, float] | None:
    if not (h and d and a and h > 1 and d > 1 and a > 1):
        return None
    ih, idr, ia = 1 / h, 1 / d, 1 / a
    s = ih + idr + ia
    return ih / s, idr / s, ia / s


def _blend(p_model: float, p_market: float | None, w: float) -> float:
    if p_market is None or np.isnan(p_market):
        return p_model
    return w * p_model + (1 - w) * p_market


def compute_market_edges(
    fx: pd.DataFrame,
    strengths: dict,
    min_ev: float = 0.02,
    blend_weight: float = MODEL_BLEND_WEIGHT,
) -> pd.DataFrame:
    """
    For each parsed MWOS fixture, run the model, blend with de-vigged MWOS odds,
    and compute EV against MWOS's actual price. Rows returned only where MWOS
    quoted a usable price.
    """
    rows = []
    for _, r in fx.iterrows():
        try:
            pred = predict_fixture(r["home"], r["away"], strengths)
        except KeyError:
            continue

        p_h_m, p_d_m, p_a_m = pred["p_home"], pred["p_draw"], pred["p_away"]
        p_o_m = pred["over_2_5"]
        p_u_m = 1 - p_o_m
        p_yy_m = pred["btts"]
        p_yn_m = 1 - p_yy_m

        mkt_1x2 = _devig_triple(r["home_odds"], r["draw_odds"], r["away_odds"])
        mkt_ou = _devig_pair(r["odds_under25"], r["odds_over25"])
        mkt_btts = _devig_pair(r["odds_gg"], r["odds_ng"])

        p_h = _blend(p_h_m, mkt_1x2[0] if mkt_1x2 else None, blend_weight)
        p_d = _blend(p_d_m, mkt_1x2[1] if mkt_1x2 else None, blend_weight)
        p_a = _blend(p_a_m, mkt_1x2[2] if mkt_1x2 else None, blend_weight)
        p_u = _blend(p_u_m, mkt_ou[0] if mkt_ou else None, blend_weight)
        p_o = _blend(p_o_m, mkt_ou[1] if mkt_ou else None, blend_weight)
        p_yy = _blend(p_yy_m, mkt_btts[0] if mkt_btts else None, blend_weight)
        p_yn = _blend(p_yn_m, mkt_btts[1] if mkt_btts else None, blend_weight)

        p_1x = p_h + p_d
        p_12 = p_h + p_a
        p_x2 = p_d + p_a

        markets = [
            ("HOME (1)",  r["home_odds"],   p_h,  p_h_m),
            ("DRAW (X)",  r["draw_odds"],   p_d,  p_d_m),
            ("AWAY (2)",  r["away_odds"],   p_a,  p_a_m),
            ("1X",        r["odds_1x"],     p_1x, p_h_m + p_d_m),
            ("12",        r["odds_12"],     p_12, p_h_m + p_a_m),
            ("X2",        r["odds_x2"],     p_x2, p_d_m + p_a_m),
            ("Under 2.5", r["odds_under25"], p_u,  p_u_m),
            ("Over 2.5",  r["odds_over25"], p_o,  p_o_m),
            ("BTTS Yes",  r["odds_gg"],     p_yy, p_yy_m),
            ("BTTS No",   r["odds_ng"],     p_yn, p_yn_m),
        ]

        overround_1x2 = None
        if all(r[c] and r[c] > 1 for c in ["home_odds", "draw_odds", "away_odds"]):
            overround_1x2 = (1 / r["home_odds"] + 1 / r["draw_odds"] + 1 / r["away_odds"]) - 1

        for name, dec_odds, p_blend, p_model in markets:
            if dec_odds is None or (isinstance(dec_odds, float) and np.isnan(dec_odds)) or dec_odds <= 1:
                continue
            ev = _ev(p_blend, dec_odds)
            rows.append(
                {
                    "date": r["date"].date().isoformat(),
                    "kickoff": r["kickoff"],
                    "home": r["home"],
                    "away": r["away"],
                    "market": name,
                    "mwos_odds": dec_odds,
                    "p_model": round(p_model, 4),
                    "p_blend": round(p_blend, 4),
                    "ev_per_unit": round(ev, 4),
                    "kelly_full": round(_kelly(p_blend, dec_odds), 4),
                    "kelly_quarter": round(_kelly(p_blend, dec_odds) * 0.25, 4),
                    "overround_1x2": round(overround_1x2, 4) if overround_1x2 is not None else None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _tier(e: float) -> str:
        if e >= 0.08:
            return "STRONG"
        if e >= 0.04:
            return "BUY"
        if e >= min_ev:
            return "MARGINAL"
        return "SKIP"

    df["signal"] = df["ev_per_unit"].apply(_tier)
    df = df.sort_values(["date", "kickoff", "home", "ev_per_unit"], ascending=[True, True, True, False]).reset_index(drop=True)
    return df


def format_report(edges: pd.DataFrame, min_ev: float = 0.02, blend_weight: float = MODEL_BLEND_WEIGHT) -> str:
    if edges.empty:
        return "no fixtures to report\n"

    lines = []
    lines.append("=" * 88)
    lines.append(f"MWOS DAILY VALUE-BET REPORT   (blend w_model = {blend_weight:.2f}, min EV = {min_ev:+.1%})")
    lines.append("=" * 88)

    keep = edges[edges["ev_per_unit"] >= min_ev].copy()
    if keep.empty:
        lines.append("no bets clear the min EV threshold today.")
        lines.append("")
        lines.append("all fixtures scanned (top row per fixture shown):")
        top = edges.sort_values(["date", "kickoff", "ev_per_unit"], ascending=[True, True, False])
        for (d, k, h, a), _grp in top.groupby(["date", "kickoff", "home", "away"], sort=False):
            g = _grp.head(1).iloc[0]
            lines.append(f"  {d} {k}  {h} vs {a}   best={g['market']:<10} EV={g['ev_per_unit']:+.2%}")
        return "\n".join(lines) + "\n"

    for (d, k, h, a), grp in keep.groupby(["date", "kickoff", "home", "away"], sort=False):
        lines.append("")
        lines.append(f"{d}  {k}   {h}  vs  {a}")
        overround = grp["overround_1x2"].iloc[0]
        if overround is not None and not pd.isna(overround):
            lines.append(f"   MWOS 1X2 overround: {overround:+.1%}")
        lines.append(f"   {'MARKET':<10} {'ODDS':>6} {'p_blend':>9} {'p_model':>9} {'EV':>8} {'KELLY/4':>9}  SIGNAL")
        for _, row in grp.iterrows():
            lines.append(
                f"   {row['market']:<10} {row['mwos_odds']:>6.2f} "
                f"{row['p_blend']*100:>8.1f}% {row['p_model']*100:>8.1f}% "
                f"{row['ev_per_unit']*100:>+7.2f}% {row['kelly_quarter']*100:>7.2f}%  {row['signal']}"
            )

    lines.append("")
    lines.append("-" * 88)
    strong = keep[keep["signal"] == "STRONG"]
    buy = keep[keep["signal"] == "BUY"]
    marginal = keep[keep["signal"] == "MARGINAL"]
    lines.append(f"summary: {len(strong)} STRONG, {len(buy)} BUY, {len(marginal)} MARGINAL")
    lines.append("staking guide: use KELLY/4 as the fraction of bankroll per bet. never chase.")
    lines.append("")
    return "\n".join(lines) + "\n"

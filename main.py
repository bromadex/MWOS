from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from backtest import naive_baselines, walk_forward
from config import MODEL_BLEND_WEIGHT, MWOS_INBOX, OUT_DIR
from data import all_spanish_team_names, build_matches, split_completed_upcoming
from model import predict_fixture
from mwos_edges import compute_market_edges, format_report
from mwos_pdf import parse_pdf
from ratings import compute_team_strengths, recent_top_flight_teams


def cmd_predict(args):
    print("[1/3] loading match data...")
    df = build_matches()
    played, upcoming = split_completed_upcoming(df)
    print(f"      played={len(played)}  xG matched={played['home_xg'].notna().sum()}  upcoming={len(upcoming)}")

    print("[2/3] computing team strengths...")
    strengths = compute_team_strengths(played)

    horizon = pd.Timestamp.now() + timedelta(days=args.days)
    up = upcoming[upcoming["date"] <= horizon].sort_values("date")

    print(f"[3/3] predicting {len(up)} upcoming fixtures within {args.days} days...")
    rows = []
    for _, r in up.iterrows():
        try:
            p = predict_fixture(r["home"], r["away"], strengths)
        except KeyError as e:
            print(f"      skip {r['home']} vs {r['away']}: {e}")
            continue
        rows.append({"date": r["date"].date().isoformat(), **p})

    out = pd.DataFrame(rows)
    if out.empty:
        print("no predictions produced")
        return

    dest = OUT_DIR / "predictions.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved {len(out)} rows -> {dest}")
    print(out.to_string(index=False))


def cmd_backtest(args):
    print("loading data...")
    df = build_matches()
    played, _ = split_completed_upcoming(df)
    print(f"backtesting on {len(played)} matches...\n")

    print("naive baselines (log-loss on the full played set):")
    b = naive_baselines(played)
    print(f"  n = {b['n']}")
    print(f"  empirical H/D/A share       : {b['sample_hda_share']}")
    print(f"  uniform (1/3 each)          : {b['uniform_1x2']:.4f}")
    print(f"  always home                 : {b['always_home']:.4f}")
    print(f"  league-average H/D/A prior  : {b['league_avg_1x2']:.4f}")
    print(f"  Pinnacle-class benchmark    : ~0.98 (reference from literature)\n")

    print("walk-forward log-loss: model vs Pinnacle-closing vs blends")
    res = walk_forward(
        played,
        half_lives=tuple(args.half_lives),
        blend_weights=tuple(args.blend_weights),
        n_folds=args.folds,
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res.to_string(index=False))
    res.to_csv(OUT_DIR / "backtest.csv", index=False)


def _latest_inbox_pdf() -> str | None:
    pdfs = sorted(MWOS_INBOX.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pdfs[0]) if pdfs else None


def _run_mwos(pdf_path: str, min_ev: float, min_recent: int, window_days: int, blend: float) -> None:
    print("[1/4] loading match data...")
    df = build_matches()
    played, _ = split_completed_upcoming(df)
    xg_matched = played["home_xg"].notna().sum() if "home_xg" in played.columns else 0
    print(f"      played={len(played)}  xG matched={xg_matched}")

    recent_teams = recent_top_flight_teams(played, min_matches=min_recent, window_days=window_days)
    print(f"      recent top-flight teams (>={min_recent} in {window_days}d): {len(recent_teams)}")

    print("[2/4] computing team strengths...")
    strengths = compute_team_strengths(played)

    from config import CURRENT_SEASON
    known_teams = all_spanish_team_names(range(2012, CURRENT_SEASON + 1))

    print(f"[3/4] parsing {pdf_path}...")
    fx = parse_pdf(pdf_path, known_teams)
    print(f"      parsed {len(fx)} Spanish football fixtures (Primera + Segunda + cup)")
    if fx.empty:
        print("no fixtures matched; nothing to compute")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    fx_dest = OUT_DIR / f"mwos_fixtures_{stamp}.csv"
    fx.to_csv(fx_dest, index=False)
    print(f"      fixtures -> {fx_dest}")

    print(f"[4/4] blending model with de-vigged MWOS (w_model={blend:.2f}) and scoring markets...")
    edges = compute_market_edges(fx, strengths, min_ev=min_ev, blend_weight=blend)
    if edges.empty:
        print("no bets produced")
        return

    edges_dest = OUT_DIR / f"mwos_edges_{stamp}.csv"
    edges.to_csv(edges_dest, index=False)

    report = format_report(edges, min_ev=min_ev, blend_weight=blend)
    report_dest = OUT_DIR / f"mwos_report_{stamp}.txt"
    report_dest.write_text(report, encoding="utf-8")

    print(f"\nedges  -> {edges_dest}")
    print(f"report -> {report_dest}\n")
    print(report)


def cmd_mwos(args):
    _run_mwos(
        args.pdf,
        min_ev=args.min_ev,
        min_recent=args.min_recent_matches,
        window_days=args.recent_window_days,
        blend=args.blend,
    )


def cmd_daily(args):
    pdf = _latest_inbox_pdf()
    if not pdf:
        print(f"no PDF found in {MWOS_INBOX}")
        print("drop today's MWOS-DAILYFIXTURE PDF there and re-run.")
        return
    print(f"using latest inbox PDF: {pdf}")
    _run_mwos(
        pdf,
        min_ev=args.min_ev,
        min_recent=args.min_recent_matches,
        window_days=args.recent_window_days,
        blend=args.blend,
    )


def build_parser():
    p = argparse.ArgumentParser(description="La Liga xG Poisson model + MWOS value bets")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="run model on upcoming fixtures")
    pp.add_argument("--days", type=int, default=14)
    pp.set_defaults(func=cmd_predict)

    bp = sub.add_parser("backtest", help="walk-forward backtest")
    bp.add_argument("--half-lives", type=int, nargs="+", default=[540])
    bp.add_argument("--blend-weights", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    bp.add_argument("--folds", type=int, default=5)
    bp.set_defaults(func=cmd_backtest)

    mp = sub.add_parser("mwos", help="parse a specific MWOS daily PDF and compute value bets")
    mp.add_argument("pdf", type=str, help="path to MWOS-DAILYFIXTURE PDF")
    mp.add_argument("--min-ev", type=float, default=0.02, help="min expected value per unit to flag as BUY")
    mp.add_argument("--min-recent-matches", type=int, default=12)
    mp.add_argument("--recent-window-days", type=int, default=730)
    mp.add_argument("--blend", type=float, default=MODEL_BLEND_WEIGHT, help="model weight in blend (0=all market, 1=all model)")
    mp.set_defaults(func=cmd_mwos)

    dp = sub.add_parser("daily", help=f"auto-pick newest PDF from {MWOS_INBOX} and run")
    dp.add_argument("--min-ev", type=float, default=0.02)
    dp.add_argument("--min-recent-matches", type=int, default=12)
    dp.add_argument("--recent-window-days", type=int, default=730)
    dp.add_argument("--blend", type=float, default=MODEL_BLEND_WEIGHT)
    dp.set_defaults(func=cmd_daily)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)

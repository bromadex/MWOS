# MWOS Value-Bet Scanner

La Liga Poisson + xG + Dixon-Coles model with a Streamlit web app that scores value bets from an uploaded MWOS-DAILYFIXTURE PDF.

## What it does

1. Loads a precomputed **team-strength snapshot** (built from openfootball results + Understat xG + football-data.co.uk historical odds).
2. Parses your daily **MWOS PDF** to extract fixtures and 10 markets per match (1/X/2, DC, O/U 2.5, BTTS).
3. Runs the model, **blends the model probability with the de-vigged MWOS line**, and computes EV + quarter-Kelly stake per market.
4. Displays value bets grouped by fixture and offers CSV/TXT download.

## Architecture

- `precompute.py` — **run locally** to refresh the snapshot (needs `soccerdata`, pulls external data).
- `snapshot/` — small JSON files committed to the repo (`strengths.json`, `recent_teams.json`, `meta.json`).
- `webapp.py` — Streamlit app. On cold start it loads the snapshot, then only does PDF parsing + math per upload.
- `main.py` — CLI (predict / backtest / mwos / daily). Optional; the web app is the main interface.

## Local development

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python precompute.py         # refresh the snapshot
python -m streamlit run webapp.py
```

Or double-click `run_web.bat`.

## Refreshing the model

Whenever you want the ratings updated:

```bash
python precompute.py
git add snapshot/
git commit -m "refresh snapshot"
git push
```

Streamlit Cloud auto-deploys on push.

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (public or private both work).
2. Go to https://share.streamlit.io → New app.
3. Point it at your repo, branch `main`, main file `webapp.py`.
4. Deploy. The app installs `requirements.txt` only (which excludes `soccerdata` / Chrome deps) and reads the committed snapshot. First cold start is a few seconds.

The `snapshot/` folder must be committed — the cloud host cannot rebuild it (no headless Chrome).

## The CLI (still supported)

```bash
python main.py daily             # auto-pick newest PDF from mwos_inbox/
python main.py mwos path\to.pdf  # explicit PDF path
python main.py backtest          # walk-forward log-loss vs Pinnacle closing
```

## Files & directories

```
config.py         constants, paths, blend weight
data.py           openfootball + Understat + odds joins  (LOCAL ONLY)
odds_history.py   football-data.co.uk fetcher            (LOCAL ONLY)
ratings.py        recency-decayed, shrunk team strengths
model.py          Poisson + Dixon-Coles score matrix
mwos_pdf.py       PDF -> fixtures dataframe
mwos_edges.py     blends model with de-vigged MWOS, computes EV
snapshot_loader.py loads precomputed strengths (cloud-safe)
precompute.py     rebuilds the snapshot (LOCAL ONLY)
backtest.py       walk-forward + baseline log-loss
webapp.py         Streamlit UI  (cloud entry point)
main.py           CLI
snapshot/         committed team strengths JSON
data/             cached raw pulls (gitignored)
output/           reports/CSVs from CLI runs (gitignored)
mwos_inbox/       drop PDFs here for `daily` CLI (gitignored)
```

## Notes on the model

- **Backtest** (5,310 La Liga matches): model log-loss 0.998 vs Pinnacle closing 0.971. Model is close to but not sharper than the sharpest book. It should be used as a decision-support tool, not a silver bullet.
- **Blending** the model with the de-vigged MWOS line (default `w_model = 0.35`) collapses noisy edges into realistic ones. Lower weight = safer; higher = more aggressive.
- **Kelly/4** = quarter-Kelly recommended stake. Never full-Kelly.

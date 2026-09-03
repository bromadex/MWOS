from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
for _d in (DATA_DIR, OUT_DIR):
    try:
        _d.mkdir(exist_ok=True)
    except OSError:
        pass

LEAGUE = "la-liga"
OPENFOOTBALL_REPO = "https://raw.githubusercontent.com/openfootball/football.json/master"
UNDERSTAT_LEAGUE = "La_liga"
FIRST_XG_SEASON = 2014
CURRENT_SEASON = 2026

RECENCY_HALF_LIFE_DAYS = 540
SHRINKAGE_MATCHES = 8.0
STRENGTH_FLOOR = 0.50
STRENGTH_CEILING = 1.80
XG_LAMBDA_WEIGHT = 0.70
MIN_EXPECTED_GOALS = 0.15
SCORE_MATRIX_MAX_GOALS = 12

MIN_RECENT_MATCHES = 12
RECENT_WINDOW_DAYS = 730

MODEL_BLEND_WEIGHT = 0.35
MWOS_INBOX = ROOT / "mwos_inbox"
try:
    MWOS_INBOX.mkdir(exist_ok=True)
except OSError:
    pass

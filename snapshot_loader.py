from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import ROOT

SNAPSHOT_DIR = ROOT / "snapshot"


def load_snapshot() -> tuple[dict, set[str], dict] | None:
    s_path = SNAPSHOT_DIR / "strengths.json"
    r_path = SNAPSHOT_DIR / "recent_teams.json"
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
    meta = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else {}
    return strengths, recent_teams, meta

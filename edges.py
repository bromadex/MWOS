from __future__ import annotations

from config import KALSHI_FEE_RATE, MIN_EDGE_BUY, MIN_EDGE_STRONG


def signal(edge: float) -> str:
    if edge >= MIN_EDGE_STRONG:
        return "STRONG"
    if edge >= MIN_EDGE_BUY:
        return "BUY"
    if edge >= 0:
        return "MARGINAL"
    return "AVOID"


def compute_edge(model_prob: float, yes_price: float, fee: float = KALSHI_FEE_RATE) -> dict:
    yes_edge = model_prob - yes_price - fee
    no_edge = (1 - model_prob) - (1 - yes_price) - fee

    if yes_edge >= no_edge:
        best = yes_edge
        side = "YES"
    else:
        best = no_edge
        side = "NO"

    return {
        "yes_price": round(yes_price, 4),
        "yes_edge": round(yes_edge, 4),
        "no_edge": round(no_edge, 4),
        "recommend_side": side if best > 0 else "NONE",
        "best_edge": round(best, 4),
        "signal": signal(best),
    }

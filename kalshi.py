from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from config import KALSHI_BASE

load_dotenv()


class KalshiClient:
    def __init__(self, env: str | None = None, key_id: str | None = None, private_key_path: str | None = None):
        self.env = env or os.getenv("KALSHI_ENV", "demo")
        self.base = KALSHI_BASE[self.env]
        self.key_id = key_id or os.getenv("KALSHI_KEY_ID")
        pk_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self._pk = None
        if pk_path and Path(pk_path).exists():
            with open(pk_path, "rb") as f:
                self._pk = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, method: str, path: str) -> dict:
        if not (self.key_id and self._pk):
            return {}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode("utf-8")
        sig = self._pk.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self.base + path
        headers = {"accept": "application/json", **self._sign("GET", "/trade-api/v2" + path)}
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_events(self, series_ticker: str | None = None, status: str = "open", limit: int = 200) -> list:
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        out = []
        cursor = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params=params)
            out.extend(data.get("events", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return out

    def list_markets(self, event_ticker: str | None = None, status: str = "open", limit: int = 200) -> list:
        params = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        out = []
        cursor = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params=params)
            out.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return out

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}/orderbook").get("orderbook", {})


def yes_bid_ask(ob: dict) -> tuple[float | None, float | None]:
    yes_side = ob.get("yes") or []
    no_side = ob.get("no") or []
    yes_bid = max((int(p) for p, _ in yes_side), default=None)
    yes_ask = None
    if no_side:
        no_bid = max(int(p) for p, _ in no_side)
        yes_ask = 100 - no_bid
    return (yes_bid / 100 if yes_bid is not None else None,
            yes_ask / 100 if yes_ask is not None else None)


def market_yes_price(market: dict, ob: dict | None = None) -> float | None:
    if ob:
        bid, ask = yes_bid_ask(ob)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        if bid is not None:
            return bid
        if ask is not None:
            return ask
    last = market.get("last_price")
    if last is not None:
        return last / 100
    return None

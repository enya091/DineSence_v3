
# services/analyzer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from collections import defaultdict, deque
import time
import pandas as pd

ROLLING_SEC = 60
TIME_BIN_SEC = 300
ZONE_DEFAULT = "unknown"

EVENT_SCORE = {
    "smile": +1.0,
    "nod": +0.8,
    "lean_forward": +0.3,
    "hand_raise": 0.0,
    "neutral": 0.0,
    "shake": -0.8,
    "frown": -0.8,
    "lean_back": -0.3,
    "dislike": -1.0,
    "wait": -0.5,
}

def to_bin(ts: float, bin_sec: int = TIME_BIN_SEC) -> int:
    return int(ts // bin_sec) * bin_sec

@dataclass
class StatPoint:
    ts_bin: int
    zone: str
    pos: int
    neu: int
    neg: int
    score_sum: float
    events: int

class EmotionSatisfactionAnalyzer:
    def __init__(self, rolling_sec:int=ROLLING_SEC):
        self.rolling_sec = rolling_sec
        self.buffer = deque()
        self.grid = defaultdict(lambda: defaultdict(lambda: StatPoint(0, ZONE_DEFAULT, 0,0,0,0.0,0)))

    @staticmethod
    def _bucket(score: float) -> str:
        if score > 0.2: return "pos"
        if score < -0.2: return "neg"
        return "neu"

    def ingest(self, events: Iterable):
        import time as _t
        now = _t.time()
        for ev in events:
            score = EVENT_SCORE.get(ev.event_type, 0.0)
            b = self._bucket(score)
            self.buffer.append((ev.ts, ev.zone or ZONE_DEFAULT, b, score))
            ts_bin = to_bin(ev.ts); zone = ev.zone or ZONE_DEFAULT
            if self.grid[ts_bin][zone].ts_bin == 0:
                self.grid[ts_bin][zone] = StatPoint(ts_bin, zone, 0,0,0,0.0,0)
            sp = self.grid[ts_bin][zone]
            if b == "pos": sp.pos += 1
            elif b == "neg": sp.neg += 1
            else: sp.neu += 1
            sp.score_sum += score; sp.events += 1
        while self.buffer and (now - self.buffer[0][0]) > self.rolling_sec:
            self.buffer.popleft()

    def get_realtime_stats(self) -> Dict[str, Any]:
        pos=neu=neg=0; score_sum=0.0; n=0
        for _,_,b,s in self.buffer:
            if b=="pos": pos+=1
            elif b=="neg": neg+=1
            else: neu+=1
            score_sum += s; n += 1
        avg = (score_sum/n) if n else 0.0
        return {"window_sec": self.rolling_sec, "count":{"pos":pos,"neu":neu,"neg":neg,"total":n}, "avg_score":avg}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for ts_bin, zones in self.grid.items():
            for zone, sp in zones.items():
                rows.append({"ts_bin":ts_bin,"zone":zone,"pos":sp.pos,"neu":sp.neu,"neg":sp.neg,
                             "events":sp.events,"score_sum":sp.score_sum,
                             "avg_score": (sp.score_sum/sp.events) if sp.events else 0.0})
        import pandas as _pd
        if not rows: return _pd.DataFrame(columns=["ts_bin","zone","pos","neu","neg","events","score_sum","avg_score"])
        return _pd.DataFrame(rows).sort_values(["ts_bin","zone"])

    def satisfaction_curve(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty: return df
        return df.groupby("ts_bin", as_index=False)["avg_score"].mean().sort_values("ts_bin")

    def heatmap_table(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty: return df
        return df.pivot(index="zone", columns="ts_bin", values="avg_score").fillna(0.0)

    def summary_report(self) -> Dict[str, Any]:
        df = self.to_dataframe()
        if df.empty: return {"events":0,"avg_score":0.0,"best_zone":None,"worst_zone":None}
        agg = df.groupby("zone")["avg_score"].mean().sort_values(ascending=False)
        return {"events": int(df["events"].sum()), "avg_score": float(df["avg_score"].mean()),
                "best_zone": None if agg.empty else agg.index[0],
                "worst_zone": None if agg.empty else agg.index[-1]}

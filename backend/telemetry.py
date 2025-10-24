# -*- coding: utf-8 -*-
import os, csv, time, datetime, threading
from typing import Optional, Dict, Any

_TELEMETRY_DIR = os.path.join("telemetry")
os.makedirs(_TELEMETRY_DIR, exist_ok=True)
_DEFAULT_LOG = os.path.join(_TELEMETRY_DIR, "log.csv")

class Telemetry:
    def __init__(self, path: Optional[str]=None):
        self.path = path or _DEFAULT_LOG
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","session","event","detail","elapsed_sec"])

    def event(self, session: str, event: str, detail: str="", elapsed_sec: float=0.0):
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.datetime.utcnow().isoformat(), session, event, detail, "{:.3f}".format(elapsed_sec)])

telemetry = Telemetry()

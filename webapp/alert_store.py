"""Small SQLite repository for operator alert state."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


VALID_ACTIONS = {
    "acknowledge": "acknowledged",
    "resolve": "resolved",
    "false-positive": "false_positive",
}


class AlertStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_key TEXT PRIMARY KEY,
                    segment_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    risk_score_pct REAL NOT NULL,
                    alert_level TEXT NOT NULL,
                    event_type TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    note TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT 'operator',
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def make_key(detection: dict) -> str:
        identity = "|".join(
            str(detection.get(field, ""))
            for field in ("segment_id", "timestamp", "risk_score_pct", "pressure", "flow_rate")
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def upsert(self, detection: dict) -> str:
        key = detection.get("alert_key") or self.make_key(detection)
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO alerts (
                    alert_key, segment_id, timestamp, risk_score_pct, alert_level,
                    event_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alert_key) DO UPDATE SET
                    risk_score_pct=excluded.risk_score_pct,
                    alert_level=excluded.alert_level,
                    event_type=excluded.event_type,
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    int(detection["segment_id"]),
                    str(detection["timestamp"]),
                    float(detection["risk_score_pct"]),
                    detection["alert_level"],
                    detection.get("event_type", "unknown"),
                    now,
                    now,
                ),
            )
        return key

    def enrich(self, detections: list[dict]) -> list[dict]:
        if not detections:
            return []
        keys = [self.upsert(detection) for detection in detections]
        placeholders = ",".join("?" for _ in keys)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT alert_key, status, note, actor, acknowledged_at, resolved_at FROM alerts WHERE alert_key IN ({placeholders})",
                keys,
            ).fetchall()
        state_by_key = {row["alert_key"]: dict(row) for row in rows}
        enriched = []
        for detection, key in zip(detections, keys):
            enriched.append({**detection, **state_by_key.get(key, {}), "alert_key": key})
        return enriched

    def update(self, alert_key: str, action: str, note: str = "", actor: str = "operator") -> dict:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported alert action: {action}")
        status = VALID_ACTIONS[action]
        now = datetime.now(timezone.utc).isoformat()
        timestamp_column = "acknowledged_at" if action == "acknowledge" else "resolved_at"
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT alert_key FROM alerts WHERE alert_key = ?", (alert_key,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown alert: {alert_key}")
            connection.execute(
                f"UPDATE alerts SET status = ?, note = ?, actor = ?, {timestamp_column} = ?, updated_at = ? WHERE alert_key = ?",
                (status, note[:1000], actor[:120], now, now, alert_key),
            )
            row = connection.execute(
                "SELECT alert_key, status, note, actor, acknowledged_at, resolved_at FROM alerts WHERE alert_key = ?",
                (alert_key,),
            ).fetchone()
        return dict(row)

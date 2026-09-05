import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from alert_store import AlertStore  # noqa: E402


class AlertStoreTests(unittest.TestCase):
    def test_alert_can_be_enriched_acknowledged_and_resolved(self):
        detection = {
            "segment_id": 7,
            "timestamp": "2026-09-05T10:00:00+00:00",
            "risk_score_pct": 78.2,
            "alert_level": "Critical Leak Threat",
            "event_type": "leak",
            "pressure": 42.0,
            "flow_rate": 1.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            store = AlertStore(Path(directory) / "alerts.db")
            enriched = store.enrich([detection])[0]
            self.assertEqual(enriched["status"], "open")
            acknowledged = store.update(enriched["alert_key"], "acknowledge", actor="tester")
            self.assertEqual(acknowledged["status"], "acknowledged")
            resolved = store.update(enriched["alert_key"], "resolve", actor="tester")
            self.assertEqual(resolved["status"], "resolved")


if __name__ == "__main__":
    unittest.main()

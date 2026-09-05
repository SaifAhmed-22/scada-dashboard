import sys
from pathlib import Path
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from scada_adapter import load_config, normalize_scada_frame  # noqa: E402


class ScadaAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "config" / "titas_scada.yml")

    def test_current_canonical_schema_is_accepted(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00"],
                "segment_id": [1],
                "pressure": [70],
                "flow_rate": [4],
                "temperature": [30],
                "valve_status": [1],
                "pump_state": [1],
                "pump_speed": [1200],
                "compressor_state": [1],
                "energy_consumption": [20],
                "target": [0],
            }
        )
        normalized, report = normalize_scada_frame(frame, self.config, require_labels=True)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(report["invalid_rows_removed"], 0)
        self.assertEqual(str(normalized.loc[0, "timestamp"].tz), "UTC")

    def test_unit_conversion_and_invalid_range_reporting(self):
        config = dict(self.config)
        config["columns"] = {**self.config["columns"], "pressure": "p_psi", "flow_rate": "q_m3h"}
        config["units"] = {
            **self.config["units"],
            "pressure": {"source": "psi", "target": "bar"},
            "flow_rate": {"source": "m3/h", "target": "m3/s"},
        }
        frame = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
                "segment_id": [1, 1],
                "p_psi": [100, 5000],
                "q_m3h": [3600, 3600],
                "temperature": [30, 30],
                "valve_status": [1, 1],
                "pump_state": [1, 1],
                "pump_speed": [1200, 1200],
                "compressor_state": [1, 1],
                "energy_consumption": [20, 20],
                "target": [0, 0],
            }
        )
        normalized, report = normalize_scada_frame(frame, config, require_labels=True)
        self.assertEqual(len(normalized), 1)
        self.assertAlmostEqual(normalized.iloc[0]["pressure"], 6.89475729, places=5)
        self.assertAlmostEqual(normalized.iloc[0]["flow_rate"], 1.0, places=5)
        self.assertEqual(report["invalid_rows_removed"], 1)


if __name__ == "__main__":
    unittest.main()

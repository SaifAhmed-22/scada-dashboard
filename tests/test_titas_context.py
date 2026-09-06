import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from titas_context import add_titas_context_features, backpressure_expected_flow_mmscfd


class TitasContextTests(unittest.TestCase):
    def test_context_join(self):
        frame = pd.DataFrame({
            "well_id": ["TT-1", "TT-11"],
            "wellhead_pressure_psia": [2415.0, 2010.0],
            "wellhead_temperature_f": [150.0, 150.0],
            "flow_rate_mmscfd": [23.4, 23.6],
        })
        out = add_titas_context_features(frame)
        self.assertAlmostEqual(out.loc[0, "observed_flow_mmscfd"], 23.4)
        self.assertAlmostEqual(out.loc[1, "tubing_id_min_in"], 2.658)
        self.assertGreater(out.loc[0, "flowline_area_max_in2"], out.loc[1, "flowline_area_min_in2"])
        self.assertAlmostEqual(out.loc[0, "wht_delta_from_historical_f"], 0.0)

    def test_missing_well_id_fails_loudly(self):
        frame = pd.DataFrame({"segment_id": [1]})
        with self.assertRaises(ValueError):
            add_titas_context_features(frame)

    def test_backpressure_formula(self):
        q = backpressure_expected_flow_mmscfd(3350.0, 3000.0, 0.0006, 0.80)
        self.assertGreater(float(q), 0.0)


if __name__ == "__main__":
    unittest.main()

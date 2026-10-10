import unittest

import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from model_pipeline import _grouped_time_series_splits  # noqa: E402


class GroupedTimeSeriesSplitTests(unittest.TestCase):
    def test_rows_with_same_timestamp_stay_in_one_fold(self):
        timestamps = pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 00:00",
            "2026-01-01 00:01", "2026-01-01 00:01",
            "2026-01-01 00:02", "2026-01-01 00:02",
            "2026-01-01 00:03", "2026-01-01 00:03",
            "2026-01-01 00:04", "2026-01-01 00:04",
            "2026-01-01 00:05", "2026-01-01 00:05",
            "2026-01-01 00:06", "2026-01-01 00:06",
        ])

        folds = list(_grouped_time_series_splits(timestamps, n_splits=3))

        self.assertEqual(len(folds), 3)
        for train_indices, validation_indices in folds:
            train_times = set(timestamps[train_indices])
            validation_times = set(timestamps[validation_indices])
            self.assertTrue(train_times.isdisjoint(validation_times))
            self.assertLess(max(train_times), min(validation_times))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_table1_i2p import write_csv
from summarize_i2p_comparison import (
    ALL_METHOD_FIELDS,
    add_rankings,
    compact_indices,
    parse_named_path,
    weighted_harmonic_mean,
)


def candidate(
    vector_type: str,
    schedule: str,
    strength: float,
    unsafe_rate: float,
    relative_reduction: float,
    image_clip: float,
    prompt_clip: float = 25.0,
) -> dict:
    return {
        "vector_type": vector_type,
        "schedule": schedule,
        "strength": strength,
        "unsafe_rate": unsafe_rate,
        "relative_unsafe_reduction": relative_reduction,
        "mean_image_clip_to_baseline": image_clip,
        "mean_prompt_clip": prompt_clip,
    }


class I2PComparisonSummaryTests(unittest.TestCase):
    def test_all_methods_csv_schema_includes_variant_id(self) -> None:
        row = dict.fromkeys(ALL_METHOD_FIELDS, "")
        row["variant_id"] = "test-variant"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_methods.csv"
            write_csv(path, ALL_METHOD_FIELDS, [row])
            with path.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))
        self.assertEqual(written["variant_id"], "test-variant")

    def test_parse_named_path_and_compact_indices(self) -> None:
        name, path = parse_named_path("default=outputs/default")
        self.assertEqual(name, "default")
        self.assertEqual(str(path), "outputs/default")
        self.assertEqual(compact_indices([0, 1, 2, 4, 6, 7]), "0-2,4,6-7")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_named_path("outputs/default")

    def test_weighted_harmonic_mean_requires_both_objectives(self) -> None:
        self.assertEqual(weighted_harmonic_mean(0.0, 0.9, 0.65, 0.35), 0.0)
        score = weighted_harmonic_mean(0.8, 0.9, 0.65, 0.35)
        self.assertGreater(score, 0.8)
        self.assertLess(score, 0.9)

    def test_rankings_select_vector_leaders_and_global_objectives(self) -> None:
        rows = [
            candidate(
                "tokenwise_difference",
                "default_preserving",
                35.0,
                0.30,
                0.55,
                0.94,
            ),
            candidate(
                "tokenwise_difference",
                "default_strong",
                75.0,
                0.05,
                0.93,
                0.55,
            ),
            candidate(
                "tokenwise_consistent_difference",
                "consistent_balanced",
                50.0,
                0.18,
                0.80,
                0.91,
            ),
            candidate(
                "tokenwise_consistent_difference",
                "consistent_strong",
                75.0,
                0.00,
                1.00,
                0.50,
            ),
            {
                **candidate("none", "baseline", 0.0, 0.70, 0.0, 1.0),
                "schedule": "baseline",
            },
        ]

        add_rankings(
            rows,
            good_suppression_max_unsafe_rate=0.25,
            suppression_weight=0.65,
            clip_weight=0.35,
            top_per_vector=1,
        )

        balanced = next(row for row in rows if row["schedule"] == "consistent_balanced")
        strong = next(row for row in rows if row["schedule"] == "consistent_strong")
        baseline = next(row for row in rows if row["schedule"] == "baseline")

        self.assertIn("Top consistent trade-off #1", balanced["selection_tags"])
        self.assertIn("Best balanced trade-off", balanced["selection_tags"])
        self.assertIn("Best CLIP with good suppression", balanced["selection_tags"])
        self.assertIn("Best suppression", strong["selection_tags"])
        self.assertEqual(baseline["selection_tags"], "Baseline reference")


if __name__ == "__main__":
    unittest.main()

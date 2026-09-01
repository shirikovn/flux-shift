from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

from evaluate_table1_i2p import (
    ImageRecord,
    build_paired_summary,
    evaluate_group,
    resolve_variant_id,
    wilson_interval,
)
from src.utils.hashing import sha256_file_set


class FakeDetector:
    def __init__(
        self,
        detections: list[list[dict]],
    ) -> None:
        self.detections = detections
        self.offset = 0

    def detect_batch(
        self,
        paths: list[str],
        batch_size: int,
    ) -> list[list[dict]]:
        del batch_size

        start = self.offset
        self.offset += len(paths)
        return self.detections[start : self.offset]


def make_records(
    schedule: str,
    strength: float,
    variant_id: str,
) -> list[ImageRecord]:
    return [
        ImageRecord(
            case_name=f"case_{index}",
            prompt=f"prompt {index}",
            seed=index,
            schedule=schedule,
            strength=strength,
            variant_id=variant_id,
            path=Path(f"case_{index}.png"),
        )
        for index in range(4)
    ]


class I2PDevelopmentEvaluationTests(unittest.TestCase):
    def test_wilson_interval_contains_rate(
        self,
    ) -> None:
        low, high = wilson_interval(
            successes=20,
            total=100,
        )

        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.2)
        self.assertEqual(
            wilson_interval(0, 0),
            (0.0, 0.0),
        )

    def test_paired_metrics_capture_rescue_and_regression(
        self,
    ) -> None:
        baseline_detections = [
            [
                {
                    "class": ("FEMALE_BREAST_EXPOSED"),
                    "score": 0.9,
                }
            ],
            [],
            [
                {
                    "class": "ANUS_EXPOSED",
                    "score": 0.7,
                }
            ],
            [
                {
                    "class": ("FEMALE_BREAST_EXPOSED"),
                    "score": 0.55,
                }
            ],
        ]
        shifted_detections = [
            [
                {
                    "class": ("FEMALE_BREAST_EXPOSED"),
                    "score": 0.2,
                }
            ],
            [
                {
                    "class": ("MALE_GENITALIA_EXPOSED"),
                    "score": 0.8,
                }
            ],
            [
                {
                    "class": "ANUS_EXPOSED",
                    "score": 0.8,
                }
            ],
            [],
        ]

        baseline = evaluate_group(
            detector=FakeDetector(baseline_detections),
            records=make_records(
                schedule="baseline",
                strength=0.0,
                variant_id="baseline",
            ),
            threshold=0.6,
            batch_size=2,
        )
        shifted = evaluate_group(
            detector=FakeDetector(shifted_detections),
            records=make_records(
                schedule="full_shift",
                strength=500.0,
                variant_id="variant-a",
            ),
            threshold=0.6,
            batch_size=2,
        )

        summary, rows = build_paired_summary(
            baseline=baseline,
            shifted=shifted,
        )

        self.assertEqual(
            baseline.images_with_any,
            2,
        )
        self.assertEqual(
            baseline.counts["total"],
            2,
        )
        self.assertEqual(
            baseline.evaluations[3].max_counted_score,
            0.55,
        )
        self.assertEqual(summary["rescued"], 1)
        self.assertEqual(summary["regressed"], 1)
        self.assertEqual(
            summary["unchanged_unsafe"],
            1,
        )
        self.assertEqual(
            summary["unchanged_safe"],
            1,
        )
        self.assertEqual(len(rows), 4)

    def test_variant_id_changes_with_algorithm(
        self,
    ) -> None:
        common = {
            "schedule": "full_shift",
            "specification": {
                "intervention": {
                    "operation": "erase",
                    "steering": {
                        "strength": 500.0,
                        "controller": {
                            "restore_token_norm": False,
                        },
                    },
                }
            },
        }
        restored = {
            **common,
            "specification": {
                "intervention": {
                    "operation": "erase",
                    "steering": {
                        "strength": 500.0,
                        "controller": {
                            "restore_token_norm": True,
                        },
                    },
                }
            },
        }

        self.assertNotEqual(
            resolve_variant_id(
                common,
                "full_shift",
            ),
            resolve_variant_id(
                restored,
                "full_shift",
            ),
        )

    def test_artifact_fingerprint_tracks_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.bin"
            second = Path(directory) / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            initial = sha256_file_set(
                [
                    ("first", first),
                    ("second", second),
                ]
            )
            reordered = sha256_file_set(
                [
                    ("second", second),
                    ("first", first),
                ]
            )
            second.write_bytes(b"changed")
            changed = sha256_file_set(
                [
                    ("first", first),
                    ("second", second),
                ]
            )

        self.assertEqual(initial, reordered)
        self.assertNotEqual(initial, changed)


if __name__ == "__main__":
    unittest.main()

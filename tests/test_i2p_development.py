from __future__ import annotations

import csv
import tempfile
import unittest

from pathlib import Path

import yaml

from evaluate_table1_i2p import (
    ImageRecord,
    build_paired_summary,
    evaluate_group,
    resolve_variant_id,
    wilson_interval,
)
from prepare_preservation_review import (
    build_review_rows,
    write_review_csv,
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
    def test_manual_stress_dataset_has_20_unique_adult_cases(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/i2p_stress_manual_20.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["i2p_index"] for row in rows}), 20)
        self.assertEqual(len({row["sd_seed"] for row in rows}), 20)
        self.assertTrue(all("adult" in row["prompt"].lower() for row in rows))

    def test_step0_cutoff_ablation_is_complete_static_matrix(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "src/configs/table1_i2p_step0_cutoff_ablation.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        schedules = config["experiment"]["schedules"]

        self.assertEqual(len(schedules), 6)
        self.assertEqual(
            sum(len(schedule["strengths"]) for schedule in schedules),
            18,
        )

        for cutoff in (12, 15, 18):
            for pooled in (False, True):
                suffix = "pooled" if pooled else "no_pooled"
                name = f"b0_{cutoff}_step0_{suffix}"
                schedule = next(
                    item for item in schedules if item["name"] == name
                )

                self.assertEqual(schedule["blocks"], list(range(cutoff + 1)))
                self.assertEqual(schedule["steps"], [0])
                self.assertEqual(schedule["strengths"], [20.0, 30.0, 45.0])
                self.assertFalse(schedule["use_classifier"])
                self.assertEqual(schedule["use_pooled"], pooled)

    def test_general_control_dataset_is_balanced_and_non_nudity(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data/i2p_general_control_12.csv"
        )
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["i2p_index"] for row in rows}), 12)
        self.assertEqual(len({row["sd_seed"] for row in rows}), 12)
        self.assertEqual(
            {kind: sum(row["control_type"] == kind for row in rows) for kind in {
                "person",
                "non_person",
            }},
            {"person": 6, "non_person": 6},
        )

        forbidden = {
            "nude",
            "nudity",
            "naked",
            "breast",
            "genital",
            "explicit",
        }
        for row in rows:
            words = set(row["prompt"].lower().replace("-", " ").split())
            self.assertTrue(words.isdisjoint(forbidden))

    def test_focused_preservation_matrix_has_expected_schedules(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "src/configs/table1_i2p_focused_preservation.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        schedules = config["experiment"]["schedules"]

        self.assertEqual(len(schedules), 8)
        self.assertEqual(
            config["focused"]["token_strengths"],
            [35.0, 40.0, 45.0],
        )
        self.assertEqual(
            config["focused"]["blocks_0_14"],
            list(range(15)),
        )
        self.assertEqual(
            config["focused"]["blocks_0_15"],
            list(range(16)),
        )

        expected_pooled = {
            "no_pooled": (False, 0.0),
            "pooled_0p5": (True, 0.5),
            "pooled_1": (True, 1.0),
            "pooled_2": (True, 2.0),
        }
        for cutoff in (14, 15):
            for suffix, (enabled, pooled_strength) in expected_pooled.items():
                name = f"b0_{cutoff}_step0_{suffix}"
                schedule = next(
                    item for item in schedules if item["name"] == name
                )
                self.assertEqual(
                    schedule["blocks"],
                    f"${{focused.blocks_0_{cutoff}}}",
                )
                self.assertEqual(schedule["steps"], [0])
                self.assertEqual(
                    schedule["strengths"],
                    "${focused.token_strengths}",
                )
                self.assertFalse(schedule["use_classifier"])
                self.assertEqual(schedule["use_pooled"], enabled)
                self.assertEqual(schedule["pooled_strength"], pooled_strength)
                self.assertEqual(schedule["pooled_similarity_mode"], "positive")

    def test_focused_datasets_have_eight_prompts_and_two_seeds_each(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data"
        for filename in (
            "i2p_stress_focused_8x2.csv",
            "i2p_general_focused_8x2.csv",
        ):
            with (root / filename).open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["i2p_index"] for row in rows}), 16)
            self.assertEqual(len({row["sd_seed"] for row in rows}), 16)
            self.assertEqual(len({row["prompt_id"] for row in rows}), 8)
            for prompt_id in {row["prompt_id"] for row in rows}:
                replicas = [row for row in rows if row["prompt_id"] == prompt_id]
                self.assertEqual(len(replicas), 2)
                self.assertEqual(
                    {row["replicate"] for row in replicas},
                    {"1", "2"},
                )
                self.assertEqual(len({row["prompt"] for row in replicas}), 1)

        with (root / "i2p_general_focused_8x2.csv").open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            general_rows = list(csv.DictReader(handle))
        self.assertEqual(
            {
                kind: len(
                    {
                        row["prompt_id"]
                        for row in general_rows
                        if row["control_type"] == kind
                    }
                )
                for kind in {"person", "non_person"}
            },
            {"person": 4, "non_person": 4},
        )

    def test_preservation_review_pairs_and_filters_records(self) -> None:
        records = [
            ImageRecord(
                case_name="case_1",
                prompt="prompt",
                seed=1,
                schedule="baseline",
                strength=0.0,
                variant_id="baseline",
                path=Path("baseline.png"),
            ),
            ImageRecord(
                case_name="case_1",
                prompt="prompt",
                seed=1,
                schedule="candidate_a",
                strength=40.0,
                variant_id="a",
                path=Path("candidate_a.png"),
            ),
            ImageRecord(
                case_name="case_1",
                prompt="prompt",
                seed=1,
                schedule="candidate_b",
                strength=45.0,
                variant_id="b",
                path=Path("candidate_b.png"),
            ),
        ]

        rows = build_review_rows(
            records,
            schedules={"candidate_a"},
            strengths={40.0},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schedule"], "candidate_a")
        self.assertEqual(rows[0]["token_strength"], 40.0)
        self.assertTrue(str(rows[0]["baseline_path"]).endswith("baseline.png"))
        self.assertEqual(rows[0]["acceptable"], "")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "review.csv"
            write_review_csv(output, rows)
            with self.assertRaises(FileExistsError):
                write_review_csv(output, rows)

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
        self.assertEqual(summary["schedule"], "full_shift")
        self.assertTrue(
            all(row["schedule"] == "full_shift" for row in rows)
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

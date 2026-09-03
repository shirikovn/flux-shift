from __future__ import annotations

import argparse
import statistics

from pathlib import Path
from typing import Any

import joblib
from omegaconf import OmegaConf


EXPECTED_ACTIVATION_LOCATION = "transformer_block_output_text"
EXPECTED_BLOCKS = set(range(19))


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize official-compatible SHIFT artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/nudity_block_output"),
    )
    return parser.parse_args()


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected!r}, got {actual!r}")


def describe(values: list[float]) -> str:
    return (
        f"min={min(values):.4f}, "
        f"mean={statistics.fmean(values):.4f}, "
        f"max={max(values):.4f}"
    )


def main() -> None:
    root = parse_args().root
    vector_root = root / "dit" / "vectors"
    classifier_root = root / "svm_training" / "classifiers"
    pooled_root = root / "pooled" / "pooled"

    vector_metadata = load_mapping(vector_root / "metadata.yaml")
    classifier_metadata = load_mapping(classifier_root / "metadata.yaml")

    require_equal(
        "vector activation location",
        vector_metadata.get("activation_location"),
        EXPECTED_ACTIVATION_LOCATION,
    )
    require_equal(
        "classifier activation location",
        classifier_metadata.get("activation_location"),
        EXPECTED_ACTIVATION_LOCATION,
    )
    require_equal(
        "classifier normalization",
        classifier_metadata.get("feature_normalization"),
        "sample_l2",
    )
    require_equal("ensemble size", classifier_metadata.get("ensemble_size"), 2)
    require_equal("classifier steps", classifier_metadata.get("step_indices"), [0])

    locations = classifier_metadata.get("locations")
    if not isinstance(locations, list):
        raise TypeError("Classifier metadata has no locations list.")

    blocks = {int(location["block_index"]) for location in locations}
    require_equal("classifier blocks", blocks, EXPECTED_BLOCKS)

    balanced_accuracies: list[float] = []
    probability_gaps: list[float] = []
    split_strategy = str(classifier_metadata.get("split_strategy", "unknown"))

    for location in locations:
        block = int(location["block_index"])
        step = int(location["step_index"])
        require_equal(f"block {block} step", step, 0)

        classifier_path = classifier_root / f"block_{block:02d}" / "step_00_classifier.joblib"
        classifier = joblib.load(classifier_path)
        models = getattr(classifier, "models", None)
        if not isinstance(models, list) or len(models) != 2:
            raise RuntimeError(f"{classifier_path} is not a two-member ensemble.")

        balanced_accuracies.append(float(location["validation_balanced_accuracy"]))
        probability_gaps.append(float(location["validation_probability_gap"]))

    vector_files = sorted(vector_root.glob("block_*/step_00_vector.pt"))
    consistent_vector_files = sorted(
        vector_root.glob("block_*/step_00_consistent_vector.pt")
    )
    classifier_files = sorted(classifier_root.glob("block_*/step_00_classifier.joblib"))
    require_equal("step-0 vector count", len(vector_files), 19)
    require_equal("step-0 classifier count", len(classifier_files), 19)

    for filename in ("pooled_vector.pt", "target_embedding.pt", "metadata.yaml"):
        path = pooled_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Artifact root: {root}")
    print(f"Activation location: {EXPECTED_ACTIVATION_LOCATION}")
    print("SVM normalization: sample_l2")
    print("SVM ensemble: 2 members per block")
    print(f"SVM split: {split_strategy}")
    print("Source locations: 19 blocks x step 0")
    if consistent_vector_files:
        require_equal(
            "step-0 consistency-weighted vector count",
            len(consistent_vector_files),
            19,
        )
        vector_locations = vector_metadata.get("locations")
        if not isinstance(vector_locations, list):
            raise TypeError("Vector metadata has no locations list.")
        consistency_means = [
            float(
                location["tokenwise_consistent_difference"][
                    "token_consistency_mean"
                ]
            )
            for location in vector_locations
        ]
        print(
            "Token-wise directional consistency: "
            f"{describe(consistency_means)}"
        )
    print(f"Validation balanced accuracy: {describe(balanced_accuracies)}")
    print(f"Validation probability gap: {describe(probability_gaps)}")
    print("Artifact validation: OK")


if __name__ == "__main__":
    main()

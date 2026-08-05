from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference = read_image(args.reference)
    candidate = read_image(args.candidate)

    if reference.shape != candidate.shape:
        print(
            "Shape mismatch:",
            reference.shape,
            candidate.shape,
        )
        return 2

    difference = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))

    exact_pixels = np.array_equal(
        reference,
        candidate,
    )

    print(
        "Reference SHA256:",
        file_sha256(args.reference),
    )
    print(
        "Candidate SHA256:",
        file_sha256(args.candidate),
    )
    print("Exact pixel equality:", exact_pixels)
    print("Max absolute difference:", difference.max())
    print(
        "Mean absolute difference:",
        float(difference.mean()),
    )
    print(
        "Changed channel values:",
        int(np.count_nonzero(difference)),
    )

    return 0 if exact_pixels else 1


if __name__ == "__main__":
    sys.exit(main())

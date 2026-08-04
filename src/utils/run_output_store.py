from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image


@dataclass(frozen=True)
class RunOutputPaths:
    image_path: Path
    record_path: Path
    image_relative_path: str
    record_relative_path: str


class RunOutputStore:
    """
    Store generated images and their records atomically.

    A run is considered complete only when:

    1. Its record exists.
    2. Its image exists.
    3. The record matches the requested run specification.
    4. The image passes validation.
    """

    VALID_MODES = {
        "resume",
        "overwrite",
        "error",
    }

    def __init__(
        self,
        output_dir: str | Path,
        config: DictConfig | dict[str, Any] | None,
        logger: logging.Logger,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.records_dir = self.output_dir / "records"
        self.logger = logger

        config_dict = self._to_dict(config)

        self.mode = str(config_dict.get("mode", "resume")).strip().lower()

        self.verify_images = bool(config_dict.get("verify_images", True))

        self.repair_incomplete = bool(
            config_dict.get(
                "repair_incomplete",
                True,
            )
        )

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported resume mode: {self.mode!r}. "
                f"Available modes: {sorted(self.VALID_MODES)}"
            )

        self.images_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.records_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _to_dict(
        config: DictConfig | dict[str, Any] | None,
    ) -> dict[str, Any]:
        if config is None:
            return {}

        if isinstance(config, DictConfig):
            value = OmegaConf.to_container(
                config,
                resolve=True,
            )

            if not isinstance(value, dict):
                raise TypeError("Resume configuration must be a mapping.")

            return value

        return dict(config)

    @classmethod
    def hash_specification(
        cls,
        specification: dict[str, Any],
    ) -> str:
        """
        Return a stable SHA-256 hash of a run specification.
        """
        plain_specification = cls._to_plain_value(specification)

        serialized = json.dumps(
            plain_specification,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(serialized).hexdigest()

    def build_paths(
        self,
        run_id: str,
        filename: str,
    ) -> RunOutputPaths:
        image_path = self.images_dir / filename
        record_path = self.records_dir / f"{run_id}.yaml"

        return RunOutputPaths(
            image_path=image_path,
            record_path=record_path,
            image_relative_path=str(image_path.relative_to(self.output_dir)),
            record_relative_path=str(record_path.relative_to(self.output_dir)),
        )

    def prepare(
        self,
        paths: RunOutputPaths,
        run_id: str,
        specification_hash: str,
        specification: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """
        Decide whether to resume, overwrite, or generate.

        Returns:
            completed_record:
                Existing record when generation can be skipped.

            action:
                generated, skipped_existing, repaired, or
                overwritten.
        """
        record_exists = paths.record_path.is_file()
        image_exists = paths.image_path.is_file()

        if self.mode == "error":
            if record_exists or image_exists:
                raise FileExistsError(
                    "Output already exists for "
                    f"run_id={run_id}. "
                    "Use resume.mode=resume or overwrite."
                )

            return None, "generated"

        if self.mode == "overwrite":
            # Remove the completion marker before generation.
            # If generation is interrupted, the old image alone
            # will not be treated as a completed run.
            paths.record_path.unlink(missing_ok=True)

            return (
                None,
                "overwritten" if image_exists or record_exists else "generated",
            )

        # Resume mode.
        if record_exists:
            try:
                record = self._load_record(paths.record_path)

                self._validate_completed_record(
                    record=record,
                    paths=paths,
                    run_id=run_id,
                    specification_hash=(specification_hash),
                    specification=specification,
                )

            except Exception as error:
                return self._handle_incomplete(
                    paths=paths,
                    run_id=run_id,
                    reason=str(error),
                )

            resumed_record = dict(record)
            resumed_record["resume_action"] = "skipped_existing"

            self.logger.info(
                "Skipping completed run: %s",
                run_id,
            )

            return (
                resumed_record,
                "skipped_existing",
            )

        if image_exists:
            return self._handle_incomplete(
                paths=paths,
                run_id=run_id,
                reason=("Image exists but its completed record " "is missing."),
            )

        return None, "generated"

    def _handle_incomplete(
        self,
        paths: RunOutputPaths,
        run_id: str,
        reason: str,
    ) -> tuple[None, str]:
        if not self.repair_incomplete:
            raise RuntimeError(f"Incomplete output for run_id={run_id}: " f"{reason}")

        self.logger.warning(
            "Repairing incomplete run %s: %s",
            run_id,
            reason,
        )

        paths.record_path.unlink(missing_ok=True)
        paths.image_path.unlink(missing_ok=True)

        return None, "repaired"

    def save_completed(
        self,
        image: Image.Image,
        record: dict[str, Any],
        paths: RunOutputPaths,
    ) -> dict[str, Any]:
        """
        Atomically save the image first and its completion
        record second.

        If execution stops between the two operations, resume
        mode sees an orphan image and regenerates it.
        """
        image_bytes = self._serialize_png(image)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        self.atomic_write_bytes(
            path=paths.image_path,
            data=image_bytes,
        )

        completed_record = dict(record)
        completed_record["status"] = "completed"
        completed_record["image"] = {
            "relative_path": (paths.image_relative_path),
            "sha256": image_sha256,
            "size_bytes": len(image_bytes),
            "width": int(image.width),
            "height": int(image.height),
            "mode": str(image.mode),
            "format": "PNG",
        }

        self.atomic_save_yaml(
            data=completed_record,
            path=paths.record_path,
        )

        return completed_record

    def _validate_completed_record(
        self,
        record: dict[str, Any],
        paths: RunOutputPaths,
        run_id: str,
        specification_hash: str,
        specification: dict[str, Any],
    ) -> None:
        if record.get("status") != "completed":
            raise RuntimeError("Record status is not completed.")

        if record.get("run_id") != run_id:
            raise RuntimeError("Record run_id does not match.")

        if record.get("specification_hash") != specification_hash:
            raise RuntimeError("Run specification hash does not match.")

        if self._to_plain_value(record.get("specification")) != self._to_plain_value(specification):
            raise RuntimeError("Stored run specification does not match.")

        image_record = record.get("image")

        if not isinstance(image_record, dict):
            raise RuntimeError("Record has no image metadata.")

        if image_record.get("relative_path") != paths.image_relative_path:
            raise RuntimeError("Stored image path does not match.")

        if not paths.image_path.is_file():
            raise RuntimeError("Completed record exists, but image " "file is missing.")

        if not self.verify_images:
            return

        self._verify_png(paths.image_path)

        expected_size = image_record.get("size_bytes")

        actual_size = paths.image_path.stat().st_size

        if expected_size is not None and int(expected_size) != actual_size:
            raise RuntimeError("Image size does not match its record.")

        expected_sha256 = image_record.get("sha256")

        actual_sha256 = self.sha256_file(paths.image_path)

        if expected_sha256 != actual_sha256:
            raise RuntimeError("Image SHA-256 does not match its record.")

    @staticmethod
    def _verify_png(path: Path) -> None:
        try:
            with Image.open(path) as image:
                image_format = image.format
                image.verify()
        except Exception as error:
            raise RuntimeError(f"Invalid image file: {path}") from error

        if image_format != "PNG":
            raise RuntimeError(f"Expected PNG, received {image_format!r}.")

    @staticmethod
    def _serialize_png(
        image: Image.Image,
    ) -> bytes:
        buffer = BytesIO()
        image.save(
            buffer,
            format="PNG",
        )
        return buffer.getvalue()

    @classmethod
    def atomic_save_yaml(
        cls,
        data: Any,
        path: str | Path,
    ) -> None:
        plain_data = cls._to_plain_value(data)

        yaml_text = OmegaConf.to_yaml(
            OmegaConf.create(plain_data),
            resolve=True,
        )

        cls.atomic_write_bytes(
            path=Path(path),
            data=yaml_text.encode("utf-8"),
        )

    @staticmethod
    def atomic_write_bytes(
        path: str | Path,
        data: bytes,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = destination.with_name(f".{destination.name}." f"{uuid.uuid4().hex}.tmp")

        try:
            with temporary_path.open("xb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temporary_path,
                destination,
            )

            RunOutputStore._fsync_directory(destination.parent)

        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(
        directory: Path,
    ) -> None:
        """
        Best-effort directory synchronization.

        Some operating systems do not support fsync on a
        directory, so failure here is intentionally ignored.
        """
        try:
            descriptor = os.open(
                directory,
                os.O_RDONLY,
            )
        except OSError:
            return

        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def sha256_file(
        path: str | Path,
    ) -> str:
        digest = hashlib.sha256()

        with Path(path).open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def _load_record(
        cls,
        path: Path,
    ) -> dict[str, Any]:
        config = OmegaConf.load(path)

        value = OmegaConf.to_container(
            config,
            resolve=True,
        )

        if not isinstance(value, dict):
            raise TypeError(f"Run record is not a mapping: {path}")

        return value

    @classmethod
    def _to_plain_value(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            (DictConfig, ListConfig),
        ):
            return OmegaConf.to_container(
                value,
                resolve=True,
            )

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {str(key): cls._to_plain_value(item) for key, item in value.items()}

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [cls._to_plain_value(item) for item in value]

        if (
            isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
            or value is None
        ):
            return value

        return str(value)

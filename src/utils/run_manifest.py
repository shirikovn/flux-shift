from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

PACKAGE_DISTRIBUTIONS = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "hydra-core",
    "omegaconf",
    "numpy",
    "scikit-learn",
    "Pillow",
    "safetensors",
]


class RunManifest:
    """
    Save structured metadata about one experiment run.

    The manifest is written once at the beginning with
    status="running" and updated when stages or the full run finish.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        config: DictConfig,
        device: torch.device,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "run_manifest.yaml"

        self.run_name = str(run_name)
        self.config = config
        self.device = device

        self._start_monotonic: float | None = None
        self._manifest: dict[str, Any] = {}

    def __enter__(self) -> "RunManifest":
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._start_monotonic = time.perf_counter()

        if self._uses_cuda():
            torch.cuda.synchronize(self._cuda_index())
            torch.cuda.reset_peak_memory_stats(self._cuda_index())

        self._manifest = {
            "schema_version": 1,
            "run_name": self.run_name,
            "status": "running",
            "started_at_utc": self._utc_now(),
            "finished_at_utc": None,
            "duration_seconds": None,
            "command": {
                "executable": sys.executable,
                "arguments": list(sys.argv),
                "command_line": " ".join([sys.executable, *sys.argv]),
                "working_directory": str(Path.cwd().resolve()),
            },
            "git": self._collect_git_info(),
            "environment": self._collect_environment_info(),
            "packages": self._collect_package_versions(),
            "hardware": self._collect_hardware_info(),
            "configuration": self._to_plain_value(self.config),
            "stages": {},
            "results": {},
            "resources": {
                "cuda_memory": None,
            },
            "error": None,
        }

        self._save()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        del traceback

        self._synchronize_cuda()

        finish_monotonic = time.perf_counter()

        if self._start_monotonic is None:
            duration_seconds = None
        else:
            duration_seconds = finish_monotonic - self._start_monotonic

        self._manifest["finished_at_utc"] = self._utc_now()
        self._manifest["duration_seconds"] = duration_seconds
        self._manifest["resources"]["cuda_memory"] = self._collect_cuda_memory()

        if exc_type is None:
            self._manifest["status"] = "completed"
        else:
            self._manifest["status"] = "failed"
            self._manifest["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }

        self._save()

        # Never suppress the original exception.
        return False

    @contextmanager
    def stage(
        self,
        name: str,
    ) -> Iterator[None]:
        """
        Measure one logical stage, such as model loading or
        activation collection.
        """
        stage_name = str(name)

        if stage_name in self._manifest["stages"]:
            raise ValueError(f"Duplicate manifest stage: {stage_name!r}")

        self._synchronize_cuda()

        started_at = self._utc_now()
        started_monotonic = time.perf_counter()

        stage_record: dict[str, Any] = {
            "status": "running",
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "duration_seconds": None,
            "error": None,
        }

        self._manifest["stages"][stage_name] = stage_record
        self._save()

        try:
            yield
        except Exception as error:
            self._synchronize_cuda()

            stage_record["status"] = "failed"
            stage_record["finished_at_utc"] = self._utc_now()
            stage_record["duration_seconds"] = time.perf_counter() - started_monotonic
            stage_record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

            self._save()
            raise

        self._synchronize_cuda()

        stage_record["status"] = "completed"
        stage_record["finished_at_utc"] = self._utc_now()
        stage_record["duration_seconds"] = time.perf_counter() - started_monotonic

        self._save()

    def add_result(
        self,
        name: str,
        value: Any,
    ) -> None:
        self._manifest["results"][str(name)] = self._to_plain_value(value)
        self._save()

    def _uses_cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _cuda_index(self) -> int:
        if self.device.index is not None:
            return int(self.device.index)

        return int(torch.cuda.current_device())

    def _synchronize_cuda(self) -> None:
        if self._uses_cuda():
            torch.cuda.synchronize(self._cuda_index())

    def _collect_cuda_memory(
        self,
    ) -> dict[str, Any] | None:
        if not self._uses_cuda():
            return None

        index = self._cuda_index()

        allocated = int(torch.cuda.memory_allocated(index))
        reserved = int(torch.cuda.memory_reserved(index))
        peak_allocated = int(torch.cuda.max_memory_allocated(index))
        peak_reserved = int(torch.cuda.max_memory_reserved(index))

        return {
            "device_index": index,
            "allocated_bytes_at_finish": allocated,
            "reserved_bytes_at_finish": reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_gib_at_finish": (self._bytes_to_gib(allocated)),
            "reserved_gib_at_finish": (self._bytes_to_gib(reserved)),
            "peak_allocated_gib": (self._bytes_to_gib(peak_allocated)),
            "peak_reserved_gib": (self._bytes_to_gib(peak_reserved)),
        }

    def _collect_hardware_info(
        self,
    ) -> dict[str, Any]:
        hardware: dict[str, Any] = {
            "cpu": {
                "processor": platform.processor() or None,
                "logical_core_count": os.cpu_count(),
            },
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "gpu": None,
        }

        if self._uses_cuda():
            index = self._cuda_index()
            properties = torch.cuda.get_device_properties(index)

            hardware["gpu"] = {
                "index": index,
                "name": properties.name,
                "compute_capability": (f"{properties.major}." f"{properties.minor}"),
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_gib": self._bytes_to_gib(int(properties.total_memory)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }

        return hardware

    @staticmethod
    def _collect_environment_info() -> dict[str, Any]:
        return {
            "python_version": platform.python_version(),
            "python_implementation": (platform.python_implementation()),
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "operating_system_release": (platform.release()),
            "machine": platform.machine(),
            "hostname": platform.node(),
        }

    @staticmethod
    def _collect_package_versions() -> dict[str, Any]:
        versions: dict[str, Any] = {}

        for distribution in PACKAGE_DISTRIBUTIONS:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[distribution] = None

        return versions

    @classmethod
    def _collect_git_info(cls) -> dict[str, Any]:
        repository_root = cls._run_git(
            "rev-parse",
            "--show-toplevel",
        )

        if repository_root is None:
            return {
                "available": False,
                "repository_root": None,
                "commit": None,
                "branch": None,
                "dirty": None,
                "changed_file_count": None,
            }

        commit = cls._run_git("rev-parse", "HEAD")
        branch = cls._run_git(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        )
        status = cls._run_git(
            "status",
            "--porcelain",
        )

        changed_lines = [] if not status else [line for line in status.splitlines() if line.strip()]

        return {
            "available": True,
            "repository_root": repository_root,
            "commit": commit,
            "branch": branch,
            "dirty": bool(changed_lines),
            "changed_file_count": len(changed_lines),
        }

    @staticmethod
    def _run_git(
        *arguments: str,
    ) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            return None

        output = result.stdout.strip()
        return output or None

    def _save(self) -> None:
        temporary_path = self.path.with_suffix(".yaml.tmp")

        plain_manifest = self._to_plain_value(self._manifest)

        OmegaConf.save(
            config=OmegaConf.create(plain_manifest),
            f=temporary_path,
        )

        temporary_path.replace(self.path)

    @classmethod
    def _to_plain_value(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(
                value,
                resolve=True,
            )

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, torch.device):
            return str(value)

        if isinstance(value, torch.dtype):
            return str(value)

        if isinstance(value, dict):
            return {str(key): cls._to_plain_value(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
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

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _bytes_to_gib(value: int) -> float:
        return round(
            value / (1024**3),
            4,
        )

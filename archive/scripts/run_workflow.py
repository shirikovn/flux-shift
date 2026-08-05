from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

VALID_TARGETS = {
    "prepare",
    "generate",
    "all",
    "verify",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the complete SHIFT artifact preparation " "and generation workflow.")
    )

    parser.add_argument(
        "target",
        choices=sorted(VALID_TARGETS),
        help=(
            "prepare: build reusable artifacts; "
            "generate: generate images; "
            "all: prepare and generate; "
            "verify: validate cached artifacts."
        ),
    )

    parser.add_argument(
        "--config",
        default=("src/configs/workflow/" "cyberpunk.yaml"),
        help="Path to the workflow YAML config.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=("Run cached stages even when their " "output checks already pass."),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help=("Run only a particular stage. " "May be supplied multiple times."),
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help=("OmegaConf dotlist overrides, for example: " "--set parameters.dit_gamma=30"),
    )

    return parser.parse_args()


def load_config(
    config_path: Path,
    overrides: list[str],
) -> DictConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Workflow config does not exist: " f"{config_path}")

    config = OmegaConf.load(config_path)

    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)

        config = OmegaConf.merge(
            config,
            override_config,
        )

    OmegaConf.resolve(config)

    return config


def substitute_runtime_values(
    value: Any,
    timestamp: str,
) -> str:
    return str(value).replace(
        "{timestamp}",
        timestamp,
    )


def resolve_path(
    project_root: Path,
    raw_path: str,
    timestamp: str,
) -> Path:
    value = substitute_runtime_values(
        raw_path,
        timestamp,
    )

    path = Path(value)

    if not path.is_absolute():
        path = project_root / path

    return path


def check_output(
    project_root: Path,
    check: dict[str, Any],
    timestamp: str,
) -> tuple[bool, str]:
    if "path" in check:
        path = resolve_path(
            project_root=project_root,
            raw_path=str(check["path"]),
            timestamp=timestamp,
        )

        kind = str(
            check.get(
                "kind",
                "any",
            )
        )

        if kind == "file":
            passed = path.is_file()
        elif kind == "directory":
            passed = path.is_dir()
        elif kind == "any":
            passed = path.exists()
        else:
            raise ValueError(f"Unknown output kind: {kind!r}")

        return (
            passed,
            f"{kind}: {path}",
        )

    if "glob" in check:
        raw_pattern = substitute_runtime_values(
            check["glob"],
            timestamp,
        )

        pattern_path = Path(raw_pattern)

        if not pattern_path.is_absolute():
            pattern_path = project_root / pattern_path

        matches = glob.glob(
            str(pattern_path),
            recursive=True,
        )

        min_count = int(
            check.get(
                "min_count",
                1,
            )
        )

        passed = len(matches) >= min_count

        description = f"glob: {pattern_path} " f"({len(matches)}/{min_count})"

        return passed, description

    raise ValueError("Every output check requires " "either 'path' or 'glob'.")


def validate_stage(
    project_root: Path,
    stage: dict[str, Any],
    timestamp: str,
    print_results: bool = False,
) -> bool:
    checks = stage.get(
        "checks",
        [],
    )

    if not checks:
        return False

    passed_all = True

    for check in checks:
        passed, description = check_output(
            project_root=project_root,
            check=check,
            timestamp=timestamp,
        )

        passed_all = passed_all and passed

        if print_results:
            marker = "OK" if passed else "MISSING"

            print(f"    [{marker}] " f"{description}")

    return passed_all


def select_stages(
    config: DictConfig,
    target: str,
    explicit_stages: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    stages = OmegaConf.to_container(
        config.stages,
        resolve=True,
    )

    if not isinstance(stages, dict):
        raise TypeError("workflow.stages must be a mapping.")

    if explicit_stages:
        unknown = set(explicit_stages) - set(stages)

        if unknown:
            raise ValueError("Unknown workflow stages: " f"{sorted(unknown)}")

        return [
            (
                name,
                stages[name],
            )
            for name in explicit_stages
        ]

    if target == "verify":
        return [
            (name, stage)
            for name, stage in stages.items()
            if bool(
                stage.get(
                    "cache",
                    True,
                )
            )
        ]

    if target == "all":
        return list(stages.items())

    return [(name, stage) for name, stage in stages.items() if stage.get("group") == target]


def run_stage(
    project_root: Path,
    stage_name: str,
    stage: dict[str, Any],
    timestamp: str,
    force: bool,
    dry_run: bool,
) -> None:
    description = str(
        stage.get(
            "description",
            stage_name,
        )
    )

    cached = bool(
        stage.get(
            "cache",
            True,
        )
    )

    print()
    print("=" * 80)
    print(f"Stage: {stage_name}")
    print(description)
    print("=" * 80)

    if (
        cached
        and not force
        and validate_stage(
            project_root=project_root,
            stage=stage,
            timestamp=timestamp,
        )
    ):
        print("Artifacts already exist. " "Skipping stage.")
        return

    raw_command = stage.get("command")

    if not isinstance(
        raw_command,
        list,
    ):
        raise TypeError(f"Stage {stage_name!r} " "requires a command list.")

    command = [
        substitute_runtime_values(
            value=item,
            timestamp=timestamp,
        )
        for item in raw_command
    ]

    print("Command:\n  " + " \\\n    ".join(command))

    if dry_run:
        print("Dry-run: command was not executed.")
        return

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage {stage_name!r} failed " f"with exit code " f"{result.returncode}."
        )

    if not validate_stage(
        project_root=project_root,
        stage=stage,
        timestamp=timestamp,
        print_results=True,
    ):
        raise RuntimeError(
            f"Stage {stage_name!r} finished, " "but its expected artifacts " "were not found."
        )

    print(f"Stage {stage_name!r} completed.")


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()

    config = load_config(
        config_path=config_path,
        overrides=args.overrides,
    )

    project_root = Path(str(config.project_root)).resolve()

    if not project_root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: " f"{project_root}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    selected_stages = select_stages(
        config=config,
        target=args.target,
        explicit_stages=args.stage,
    )

    if not selected_stages:
        raise RuntimeError("No workflow stages were selected.")

    print(
        "Workflow config:",
        config_path,
    )

    print(
        "Project root:",
        project_root,
    )

    print(
        "Selected stages:",
        [name for name, _ in selected_stages],
    )

    if args.target == "verify":
        failed_stages: list[str] = []

        for stage_name, stage in selected_stages:
            print()
            print(f"Stage: {stage_name}")

            passed = validate_stage(
                project_root=project_root,
                stage=stage,
                timestamp=timestamp,
                print_results=True,
            )

            if not passed:
                failed_stages.append(stage_name)

        if failed_stages:
            print(
                "\nMissing or incomplete stages:",
                failed_stages,
            )

            sys.exit(1)

        print("\nAll reusable artifacts are valid.")
        return

    for stage_name, stage in selected_stages:
        run_stage(
            project_root=project_root,
            stage_name=stage_name,
            stage=stage,
            timestamp=timestamp,
            force=args.force,
            dry_run=args.dry_run,
        )

    print()
    print("=" * 80)
    print("Workflow completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()

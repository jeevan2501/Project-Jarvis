"""Goal contracts and evidence evaluation for autonomous software tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GoalContract:
    """Requirements inferred from a user's software-creation request."""

    software_task: bool
    requires_folder: bool
    requires_python_source: bool
    requires_tests: bool
    requires_application_execution: bool
    gui_application: bool

    @classmethod
    def from_goal(cls, goal: str) -> "GoalContract":
        text = goal.lower()

        software_verbs = (
            "create",
            "build",
            "develop",
            "make",
            "implement",
            "write",
            "generate",
        )

        software_artifacts = (
            "application",
            "app",
            "program",
            "software",
            "script",
            "calculator",
            "python",
            ".py",
            "website",
            "tool",
        )

        software_task = (
            any(word in text for word in software_verbs)
            and any(word in text for word in software_artifacts)
        )

        # A calculator/Python/application request is unambiguously
        # a software-development task even if the wording is short.
        if any(
            token in text
            for token in (
                "python calculator",
                "python script",
                "calculator",
                "python application",
                "python app",
            )
        ):
            software_task = True

        requires_folder = any(
            token in text
            for token in (
                "folder",
                "directory",
                "inside that",
                "inside the folder",
            )
        )

        requires_python_source = (
            "python" in text
            or ".py" in text
            or "calculator" in text
        )

        # Verification is optional unless the user explicitly requests it.
        #
        # Example:
        #   "Create a Python calculator"
        #       -> create only
        #
        #   "Create a Python calculator and test it"
        #       -> create + test
        #
        #   "Create a Python calculator and run it"
        #       -> create + execute
        verification_requested = any(
            token in text
            for token in (
                "test",
                "tests",
                "tested",
                "testing",
                "verify",
                "verified",
                "verification",
                "validate",
                "validated",
                "validation",
                "check whether it works",
            )
        )

        execution_requested = any(
            token in text
            for token in (
                "run it",
                "run the application",
                "run the app",
                "execute it",
                "execute the application",
                "execute the app",
                "launch it",
                "launch the application",
                "launch the app",
                "start it",
                "start the application",
                "start the app",
                "verify that it works",
                "verify it works",
                "confirm that it works",
                "confirm it works",
                "make sure it works",
                "ensure that it works",
            )
        )

        requires_tests = software_task and verification_requested

        requires_application_execution = (
            software_task
            and execution_requested
        )

        gui_application = any(
            token in text
            for token in (
                "gui",
                "graphical user interface",
                "tkinter",
                "desktop application",
                "windowed application",
                "window",
            )
        )

        return cls(
            software_task=software_task,
            requires_folder=requires_folder,
            requires_python_source=requires_python_source,
            requires_tests=requires_tests,
            requires_application_execution=requires_application_execution,
            gui_application=gui_application,
        )

    @staticmethod
    def _metadata(result: Any) -> dict[str, Any]:
        metadata = getattr(result, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _operation(result: Any) -> str:
        metadata = GoalContract._metadata(result)
        return str(metadata.get("operation", "")).lower()

    @staticmethod
    def _path(result: Any) -> str:
        metadata = GoalContract._metadata(result)
        return str(metadata.get("path", ""))

    @staticmethod
    def _successful(result: Any) -> bool:
        return bool(getattr(result, "success", False))

    @classmethod
    def evidence(cls, results: Iterable[Any]) -> dict[str, bool]:
        """Aggregate concrete evidence from all tool results."""

        evidence = {
            "folder_created": False,
            "python_source_created": False,
            "tests_created": False,
            "tests_passed": False,
            "application_executed": False,
            "startup_verified": False,
        }

        for result in results:
            if not cls._successful(result):
                continue

            operation = cls._operation(result)
            metadata = cls._metadata(result)
            path = cls._path(result)
            tool_name = str(
                getattr(result, "tool_name", "")
            ).lower()

            if operation == "folder_create" or tool_name == "folder_create":
                evidence["folder_created"] = True

            if operation == "write_file":
                suffix = Path(path).suffix.lower()
                filename = Path(path).name.lower()

                if suffix == ".py":
                    if (
                        filename.startswith("test_")
                        or filename.startswith("tests_")
                        or filename == "test.py"
                    ):
                        evidence["tests_created"] = True
                    else:
                        evidence["python_source_created"] = True

            if operation == "run_tests":
                evidence["tests_passed"] = (
                    metadata.get("returncode") == 0
                )

            if operation == "run_python":
                evidence["application_executed"] = (
                    metadata.get("returncode") == 0
                )

            if operation == "verify_startup":
                evidence["startup_verified"] = bool(
                    metadata.get("startup_verified", False)
                )

        return evidence

    def verify(self, results: Iterable[Any]) -> tuple[bool, dict[str, bool]]:
        """Return completion status and the supporting evidence."""

        evidence = self.evidence(results)

        if not self.software_task:
            return True, evidence

        required = []

        if self.requires_folder:
            required.append("folder_created")

        if self.requires_python_source:
            required.append("python_source_created")

        if self.requires_tests:
            required.append("tests_created")
            required.append("tests_passed")

        if self.requires_application_execution:
            if self.gui_application:
                required.append("startup_verified")
            else:
                required.append("application_executed")

        complete = all(
            evidence[name]
            for name in required
        )

        return complete, evidence

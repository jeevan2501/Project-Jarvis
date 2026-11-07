"""Controlled Python code and project execution tool for JARVIS."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


# All project operations are restricted to this directory.
_WORKSPACE_ROOT = Path.home().resolve()

def _safe_path(path: str | Path) -> Path:
    """Resolve a path and ensure it stays inside the JARVIS workspace."""

    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = _WORKSPACE_ROOT / candidate

    candidate = candidate.resolve()

    try:
        candidate.relative_to(_WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path is outside the JARVIS workspace: {candidate}"
        ) from exc

    return candidate


def _project_root(path: str | Path) -> Path:
    """Return a validated project directory."""

    project = _safe_path(path)

    if project == _WORKSPACE_ROOT:
        raise ValueError(
            "A project must be inside the JARVIS workspace."
        )

    return project


def _venv_python(project: Path) -> Path:
    """Return the Python executable inside a project virtual environment."""

    if sys.platform == "win32":
        python = project / ".venv" / "Scripts" / "python.exe"
    else:
        python = project / ".venv" / "bin" / "python"

    return python


def _format_output(
    stdout: str,
    stderr: str,
    max_output: int,
) -> str:
    output = stdout

    if stderr:
        output += ("\n" if output else "") + stderr

    if not output:
        output = "(no output)"

    if len(output) > max_output:
        output = (
            output[:max_output]
            + "\n... (output truncated)"
        )

    return output


@ToolRegistry.register("code_interpreter")
class CodeInterpreterTool(BaseTool):
    """Execute Python and perform controlled project operations."""

    tool_id = "code_interpreter"

    def __init__(
        self,
        timeout: int = 300,
        max_output: int = 20000,
    ):
        self._timeout = timeout
        self._max_output = max_output

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description=(
                "JARVIS local software development tool. "
                "Use this tool to create, modify, test, execute, debug, "
                "and build complete software projects. "
                "\n\n"
                "For a new Python project, follow this workflow: "
                "1. create_venv, "
                "2. write_file for the source files, "
                "3. write_file for tests and documentation, "
                "4. install_packages if dependencies are required, "
                "5. run_tests, "
                "6. fix failures only when tests report failures, "
                "7. verify the application using the verification operation appropriate "
                "to its application type: use run_python for CLI/scripts and "
                "verify_startup for GUI/event-loop applications. "
                "\n\n"
                "IMPORTANT: After a write_file operation succeeds, do NOT "
                "rewrite the same file unless there is an actual error or "
                "test failure requiring a correction. Continue to the next "
                "required project step. Do not repeatedly call write_file "
                "with alternative versions of the same file. "
                "\n\n"
                "Do not report a project as complete until it has been "
                "actually tested or executed when verification was requested."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "create_venv",
                            "install_packages",
                            "write_file",
                            "read_file",
                            "list_files",
                            "run_python",
                            "verify_startup",
                            "run_tests",
                            "run_command",
                        ],
                        "description": (
                            "Choose exactly one operation. "
                            "create_venv: create the project's .venv. "
                            "install_packages: install Python packages into the project's "
                            "virtual environment. "
                            "write_file: create or modify source code, tests, configuration, "
                            "requirements, or documentation. "
                            "read_file: inspect an existing project file. "
                            "list_files: inspect the project directory. "
                            "run_python: run an existing Python project file using its .venv. "
                            "If the program uses input() or otherwise requires interactive input, "
                            "you MUST provide the stdin parameter with the required input values. "
                            "Do not run an interactive program without stdin when verification "
                            "requires user input. "
                            "verify_startup: launch an application briefly to verify that it "
                            "starts successfully. Use this specifically for GUI applications "
                            "where normal run_python would remain alive in an event loop. "
                            "The verification process is automatically terminated after the "
                            "startup check. Use startup_timeout only with verify_startup; "
                            "never use startup_timeout with run_python. "
                            "run_tests: execute the project's test suite and return the "
                            "actual test results. "
                            "run_command: execute local development commands such as "
                            "python, pip, uv, git, npm, node, pytest, and build commands."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Project directory or file path. Can be an absolute path "
                            "inside the user's Windows profile or a project-relative path."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Complete contents of the file to create or modify. "
                            "For Python and other source files, preserve the exact "
                            "program structure and use real newline characters between "
                            "lines. Do NOT collapse multiple statements, functions, "
                            "imports, or blocks onto one line. Include proper indentation. "
                            "The content must be directly runnable as the target file."
                        ),
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Python packages to install into the project's virtual "
                            "environment, for example ['requests', 'numpy']."
                        ),
                    },
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional command-line arguments passed to the Python "
                            "script when using run_python."
                        ),
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Optional standard input supplied to an interactive Python "
                            "program. Use real newline characters between inputs. "
                            "For example, a calculator requiring choice, first number, "
                            "and second number can be tested with "
                            "'1\\n10\\n20\\n'. "
                            "If the program calls input(), provide stdin when running it."
                        ),
                    },
                    "startup_timeout": {
                        "type": "number",
                        "description": (
                            "Number of seconds to allow an application to initialize "
                            "during verify_startup. Defaults to 3 seconds."
                        ),
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "Local development command to execute with run_command. "
                            "Examples: python, pip, uv, git, npm, node, pytest, "
                            "PowerShell, or build commands."
                        ),
                    },
                },
                "required": ["operation"],
            },
            category="code",
            metadata={"structured_allow_object_text": True},
        )

    def execute(self, **params: Any) -> ToolResult:
        """Execute one code-interpreter operation."""

        operation = str(params.get("operation", "")).strip().lower()

        # ---------------------------------------------------------
        # Project / development operations
        # ---------------------------------------------------------

        if operation == "create_venv":
            return self._create_venv(
                str(params.get("path", ""))
            )

        if operation == "install_packages":
            return self._install_packages(
                str(params.get("path", "")),
                params.get("packages", []),
            )

        if operation == "write_file":
            return self._write_file(
                str(params.get("path", "")),
                str(params.get("content", "")),
            )

        if operation == "read_file":
            return self._read_file(
                str(params.get("path", ""))
            )

        if operation == "list_files":
            return self._list_files(
                str(params.get("path", ""))
            )

        if operation == "run_python":
            return self._run_python(
                str(params.get("path", "")),
                params.get("arguments", []),
                params.get("stdin", ""),
            )

        if operation == "verify_startup":
            return self._verify_startup(
                str(params.get("path", "")),
                params.get("startup_timeout", 3),
            )

        if operation == "run_tests":
            return self._run_tests(
                str(params.get("path", ""))
            )

        if operation == "run_command":
            return self._run_command(
                str(params.get("command", "")),
                str(params.get("path", "")),
            )

        if not operation:
            return ToolResult(
                tool_name=self.tool_id,
                content="No operation provided.",
                success=False,
                metadata={
                    "operation": "",
                },
            )

        return ToolResult(
            tool_name=self.tool_id,
            content=f"Unknown code interpreter operation: {operation}",
            success=False,
            metadata={
                "operation": operation,
            },
        )

    def _create_venv(self, path: str) -> ToolResult:
        """Create a virtual environment inside a project."""

        project = _project_root(path)
        project.mkdir(parents=True, exist_ok=True)

        venv_path = project / ".venv"

        if venv_path.exists():
            python = _venv_python(project)

            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    f"Virtual environment already exists: "
                    f"{venv_path}\n"
                    f"Python: {python}"
                ),
                success=python.exists(),
                metadata={
                    "operation": "create_venv",
                    "path": str(venv_path),
                },
            )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                str(venv_path),
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            cwd=str(project),
        )

        output = _format_output(
            result.stdout,
            result.stderr,
            self._max_output,
        )
        if result.returncode != 0:
            if (
                "EOFError" in output
                or "EOF when reading a line" in output
            ):
                output = (
                    "Python program requires interactive input but no stdin "
                    "was provided. Re-run run_python with the required "
                    "stdin values."
                    f"\n\n{output}"
                )

        if result.returncode != 0:
            return ToolResult(
                tool_name=self.tool_id,
                content=output,
                success=False,
                metadata={
                    "operation": "create_venv",
                    "returncode": result.returncode,
                },
            )

        return ToolResult(
            tool_name=self.tool_id,
            content=(
                f"Virtual environment created successfully.\n"
                f"Project: {project}\n"
                f"Environment: {venv_path}\n"
                f"Python: {_venv_python(project)}"
            ),
            success=True,
            metadata={
                "operation": "create_venv",
                "path": str(venv_path),
            },
        )

    def _install_packages(
        self,
        path: str,
        packages: Any,
    ) -> ToolResult:
        """Install packages into a project's virtual environment."""

        project = _project_root(path)

        if not isinstance(packages, list):
            return ToolResult(
                tool_name=self.tool_id,
                content="packages must be a list.",
                success=False,
            )

        packages = [
            str(package).strip()
            for package in packages
            if str(package).strip()
        ]

        if not packages:
            return ToolResult(
                tool_name=self.tool_id,
                content="No packages requested; nothing to install.",
                success=True,
                metadata={"operation": "install_packages"},
            )

        python = _venv_python(project)

        if not python.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    f"Virtual environment not found: {python}. "
                    "Create the virtual environment first."
                ),
                success=False,
            )

        result = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                *packages,
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            cwd=str(project),
        )

        output = _format_output(
            result.stdout,
            result.stderr,
            self._max_output,
        )

        return ToolResult(
            tool_name=self.tool_id,
            content=output,
            success=result.returncode == 0,
            metadata={
                "operation": "install_packages",
                "packages": packages,
                "returncode": result.returncode,
            },
        )

    def _write_file(
        self,
        path: str,
        content: str,
    ) -> ToolResult:
        """Write a project file inside the workspace."""

        if not path:
            return ToolResult(
                tool_name=self.tool_id,
                content="No file path provided.",
                success=False,
            )

        target = _safe_path(path)

        if (
            target.suffix == ".py"
            and "\\n" in content
            and "\n" not in content
        ):
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    "The Python source was provided with literal escaped "
                    "\\n sequences instead of actual newline characters. "
                    "Rewrite the file using real newline characters and "
                    "preserve normal Python indentation."
                ),
                success=False,
                metadata={
                    "operation": "write_file",
                    "path": str(target),
                    "invalid_python_format": True,
                },
            )

        # Never allow a file operation to write inside .venv.
        if ".venv" in target.parts:
            return ToolResult(
                tool_name=self.tool_id,
                content="Writing inside .venv is not allowed.",
                success=False,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            content,
            encoding="utf-8",
        )

        return ToolResult(
            tool_name=self.tool_id,
            content=(
                f"File written successfully: {target}\n"
                "IMPORTANT: Writing a project file does NOT mean the project "
                "is complete or working. If this is part of a software project, "
                "continue by creating the required files, running tests or the "
                "application, inspecting the actual result, and fixing any "
                "errors before reporting completion."
            ),
            success=True,
            metadata={
                "operation": "write_file",
                "path": str(target),
                "project_file": True,
                "verified": False,
            },
        )

    def _read_file(self, path: str) -> ToolResult:
        """Read a project file inside the workspace."""

        target = _safe_path(path)

        if not target.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"File not found: {target}",
                success=False,
            )

        if not target.is_file():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Not a file: {target}",
                success=False,
            )

        content = target.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if len(content) > self._max_output:
            content = (
                content[:self._max_output]
                + "\n... (output truncated)"
            )

        return ToolResult(
            tool_name=self.tool_id,
            content=content,
            success=True,
            metadata={
                "operation": "read_file",
                "path": str(target),
            },
        )

    def _list_files(self, path: str) -> ToolResult:
        """List files and directories inside the workspace."""

        directory = _safe_path(path or ".")

        if not directory.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Directory not found: {directory}",
                success=False,
            )

        if not directory.is_dir():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Not a directory: {directory}",
                success=False,
            )

        entries = []

        for item in sorted(
            directory.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        ):
            entries.append(
                f"{'[DIR]' if item.is_dir() else '[FILE]'} "
                f"{item.name}"
            )

        return ToolResult(
            tool_name=self.tool_id,
            content="\n".join(entries) or "(empty directory)",
            success=True,
            metadata={
                "operation": "list_files",
                "path": str(directory),
            },
        )

    def _run_python(
        self,
        path: str,
        arguments: Any,
        stdin: Any = "",
    ) -> ToolResult:
        """Run a Python project file using its project's venv."""

        script = _safe_path(path)

        if not script.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Python file not found: {script}",
                success=False,
            )

        if not script.is_file():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Not a file: {script}",
                success=False,
            )

        project = script.parent
        python = _venv_python(project)
        venv_created = False

        # Automatically create the project's virtual environment
        # when run_python is requested and .venv does not exist.
        if not python.exists():
            venv_path = project / ".venv"

            try:
                venv_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "venv",
                        str(venv_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(project),
                )

            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Timed out while creating the project "
                        "virtual environment."
                    ),
                    success=False,
                )

            if venv_result.returncode != 0:
                output = _format_output(
                    venv_result.stdout,
                    venv_result.stderr,
                    self._max_output,
                )

                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Failed to create the project virtual environment.\n"
                        + output
                    ),
                    success=False,
                    metadata={
                        "operation": "run_python",
                        "path": str(project),
                        "returncode": venv_result.returncode,
                        "venv_created": False,
                    },
                )
            python = _venv_python(project)
            venv_created = True
        if not isinstance(arguments, list):
            arguments = []

        if stdin is None:
            stdin = ""

        if not isinstance(stdin, str):
            stdin = str(stdin)

        try:
            result = subprocess.run(
                [
                    str(python),
                    str(script),
                    *[str(argument) for argument in arguments],
                ],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(project),
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    f"Python execution timed out after {self._timeout} seconds. "
                    "The program may be waiting for interactive stdin/input(). "
                    "If the program uses input(), rerun it with the required "
                    "values supplied through the stdin parameter. "
                    "Do not rewrite the program merely because it timed out."
                ),
                success=False,
                metadata={
                    "operation": "run_python",
                    "path": str(script),
                    "venv_created": venv_created,
                    "timeout": True,
                    "requires_stdin": True,
                },
            )

        output = _format_output(
            result.stdout,
            result.stderr,
            self._max_output,
        )

        prefix = (
            "Automatically created the project virtual environment.\n"
            if venv_created
            else ""
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        return ToolResult(
            tool_name=self.tool_id,
            content=prefix + output,
            success=result.returncode == 0,
            metadata={
                "operation": "run_python",
                "path": str(script),
                "returncode": result.returncode,
                "venv_created": venv_created,
                "stdout_present": bool(stdout.strip()),
                "stderr_present": bool(stderr.strip()),
                "stdout": stdout[:self._max_output],
                "stderr": stderr[:self._max_output],
            },
        )

    def _verify_startup(
        self,
        path: str,
        startup_timeout: Any = 3,
    ) -> ToolResult:
        """Launch an application briefly and verify that it starts."""

        script = _safe_path(path)

        if not script.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Python file not found: {script}",
                success=False,
                metadata={
                    "operation": "verify_startup",
                    "path": str(script),
                    "startup_verified": False,
                },
            )

        if not script.is_file():
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Not a file: {script}",
                success=False,
                metadata={
                    "operation": "verify_startup",
                    "path": str(script),
                    "startup_verified": False,
                },
            )

        project = script.parent
        python = _venv_python(project)
        venv_created = False

        # Bootstrap the project's virtual environment if necessary.
        if not python.exists():
            venv_path = project / ".venv"

            try:
                venv_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "venv",
                        str(venv_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(project),
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Timed out while creating the project virtual "
                        "environment."
                    ),
                    success=False,
                    metadata={
                        "operation": "verify_startup",
                        "path": str(script),
                        "startup_verified": False,
                        "venv_created": False,
                    },
                )

            if venv_result.returncode != 0:
                output = _format_output(
                    venv_result.stdout,
                    venv_result.stderr,
                    self._max_output,
                )

                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Failed to create the project virtual "
                        f"environment.\n{output}"
                    ),
                    success=False,
                    metadata={
                        "operation": "verify_startup",
                        "path": str(script),
                        "startup_verified": False,
                        "venv_created": False,
                        "returncode": venv_result.returncode,
                    },
                )

            python = _venv_python(project)
            venv_created = True

        if not python.exists():
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    "Project virtual environment was created, but "
                    "the Python executable was not found."
                ),
                success=False,
                metadata={
                    "operation": "verify_startup",
                    "path": str(script),
                    "startup_verified": False,
                    "venv_created": venv_created,
                },
            )

        try:
            startup_seconds = float(startup_timeout)
        except (TypeError, ValueError):
            startup_seconds = 3.0

        startup_seconds = max(1.0, min(startup_seconds, 10.0))

        process = None

        try:
            process = subprocess.Popen(
                [
                    str(python),
                    str(script),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(project),
            )

            try:
                stdout, stderr = process.communicate(
                    timeout=startup_seconds,
                )

                # The process exited during the startup window.
                output = _format_output(
                    stdout or "",
                    stderr or "",
                    self._max_output,
                )

                if process.returncode == 0:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content=(
                            "Application started and exited successfully "
                            "during the startup verification window.\n"
                            + output
                        ),
                        success=True,
                        metadata={
                            "operation": "verify_startup",
                            "path": str(script),
                            "startup_verified": True,
                            "process_exited": True,
                            "returncode": process.returncode,
                            "venv_created": venv_created,
                            "startup_timeout": startup_seconds,
                        },
                    )

                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Application failed during startup verification.\n"
                        + output
                    ),
                    success=False,
                    metadata={
                        "operation": "verify_startup",
                        "path": str(script),
                        "startup_verified": False,
                        "process_exited": True,
                        "returncode": process.returncode,
                        "venv_created": venv_created,
                        "startup_timeout": startup_seconds,
                    },
                )

            except subprocess.TimeoutExpired:
                # The application is still alive after the startup window.
                # For GUI applications this is normally expected because
                # the event loop keeps the process alive.
                process.terminate()

                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()

                output = _format_output(
                    stdout or "",
                    stderr or "",
                    self._max_output,
                )

                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Application remained running for the startup "
                        f"verification window ({startup_seconds:g} seconds), "
                        "indicating that the application started successfully. "
                        "The verification process was then terminated."
                        + (
                            f"\nStartup output:\n{output}"
                            if output != "(no output)"
                            else ""
                        )
                    ),
                    success=True,
                    metadata={
                        "operation": "verify_startup",
                        "path": str(script),
                        "startup_verified": True,
                        "process_exited": False,
                        "returncode": None,
                        "venv_created": venv_created,
                        "startup_timeout": startup_seconds,
                    },
                )

        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()

            return ToolResult(
                tool_name=self.tool_id,
                content=f"Application startup verification failed: {exc}",
                success=False,
                metadata={
                    "operation": "verify_startup",
                    "path": str(script),
                    "startup_verified": False,
                    "venv_created": venv_created,
                },
            )

    def _run_tests(self, path: str) -> ToolResult:
        """Run unittest discovery using the project's virtual environment."""

        project = _project_root(path)
        python = _venv_python(project)

        # Automatically bootstrap the project's virtual environment
        # when run_python is requested for a project that does not have one.
        if not python.exists():
            venv_path = project / ".venv"

            try:
                venv_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "venv",
                        str(venv_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(project),
                )

                if venv_result.returncode != 0:
                    output = _format_output(
                        venv_result.stdout,
                        venv_result.stderr,
                        self._max_output,
                    )

                    return ToolResult(
                        tool_name=self.tool_id,
                        content=(
                            "Failed to create the project virtual environment.\n"
                            + output
                        ),
                        success=False,
                        metadata={
                            "operation": "run_tests",
                            "path": str(project),
                            "returncode": venv_result.returncode,
                            "venv_created": False,
                        },
                    )

                python = _venv_python(project)

            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Timed out while creating the project virtual "
                        "environment."
                    ),
                    success=False,
                )

        result = subprocess.run(
            [
                str(python),
                "-m",
                "unittest",
                "discover",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            cwd=str(project),
        )

        output = _format_output(
            result.stdout,
            result.stderr,
            self._max_output,
        )

        return ToolResult(
            tool_name=self.tool_id,
            content=output,
            success=result.returncode == 0,
            metadata={
                "operation": "run_tests",
                "path": str(project),
                "returncode": result.returncode,
            },
        )
    def _run_command(
        self,
        command: str,
        path: str = "",
    ) -> ToolResult:
        """Execute a local development command."""

        if not command.strip():
            return ToolResult(
                tool_name=self.tool_id,
                content="No command provided.",
                success=False,
            )

        if path:
            cwd = _safe_path(path)

            if not cwd.exists():
                return ToolResult(
                    tool_name=self.tool_id,
                    content=f"Working directory not found: {cwd}",
                    success=False,
                )

            if not cwd.is_dir():
                return ToolResult(
                    tool_name=self.tool_id,
                    content=f"Working directory is not a directory: {cwd}",
                    success=False,
                )
        else:
            cwd = Path.home()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(cwd),
            )

            output = _format_output(
                result.stdout,
                result.stderr,
                self._max_output,
            )

            return ToolResult(
                tool_name=self.tool_id,
                content=output,
                success=result.returncode == 0,
                metadata={
                    "operation": "run_command",
                    "command": command,
                    "cwd": str(cwd),
                    "returncode": result.returncode,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    f"Command timed out after "
                    f"{self._timeout} seconds."
                ),
                success=False,
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Command execution error: {exc}",
                success=False,
            )


__all__ = ["CodeInterpreterTool"]

"""Tests for the controlled code interpreter tool."""

from __future__ import annotations

from pathlib import Path

import pytest

import openjarvis.tools.code_interpreter as code_interpreter_module
from openjarvis.core.registry import ToolRegistry
from openjarvis.tools.code_interpreter import CodeInterpreterTool


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Use an isolated temporary directory as the JARVIS workspace."""

    monkeypatch.setattr(
        code_interpreter_module,
        "_WORKSPACE_ROOT",
        tmp_path.resolve(),
    )

    return tmp_path


@pytest.fixture
def project(workspace):
    """Create an isolated project directory."""

    project_path = workspace / "calculator"
    project_path.mkdir()
    return project_path


class TestCodeInterpreterTool:
    def test_spec_name_and_category(self):
        tool = CodeInterpreterTool()

        assert tool.spec.name == "code_interpreter"
        assert tool.spec.category == "code"

    def test_spec_uses_operation_api(self):
        tool = CodeInterpreterTool()

        properties = tool.spec.parameters["properties"]

        assert "operation" in properties
        assert "code" not in properties
        assert "operation" in tool.spec.parameters["required"]

    def test_spec_contains_required_operations(self):
        tool = CodeInterpreterTool()

        operations = tool.spec.parameters["properties"]["operation"]["enum"]

        assert "create_venv" in operations
        assert "write_file" in operations
        assert "read_file" in operations
        assert "list_files" in operations
        assert "run_python" in operations
        assert "verify_startup" in operations
        assert "run_tests" in operations
        assert "run_command" in operations

    def test_to_openai_function(self):
        tool = CodeInterpreterTool()

        fn = tool.to_openai_function()

        assert fn["type"] == "function"
        assert fn["function"]["name"] == "code_interpreter"

        properties = fn["function"]["parameters"]["properties"]

        assert "operation" in properties
        assert "code" not in properties

    def test_no_operation_provided(self):
        tool = CodeInterpreterTool()

        result = tool.execute()

        assert result.success is False
        assert "No operation" in result.content

    def test_unknown_operation(self):
        tool = CodeInterpreterTool()

        result = tool.execute(
            operation="does_not_exist",
        )

        assert result.success is False
        assert "Unknown code interpreter operation" in result.content

    def test_write_and_read_file(self, project):
        tool = CodeInterpreterTool()

        file_path = project / "calculator.py"
        source = "print(2 + 2)\n"

        write_result = tool.execute(
            operation="write_file",
            path=str(file_path),
            content=source,
        )

        assert write_result.success is True
        assert file_path.exists()

        read_result = tool.execute(
            operation="read_file",
            path=str(file_path),
        )

        assert read_result.success is True
        assert read_result.content == source

    def test_write_python_file_rejects_literal_newlines(self, project):
        tool = CodeInterpreterTool()

        file_path = project / "calculator.py"

        result = tool.execute(
            operation="write_file",
            path=str(file_path),
            content="print(1)\\nprint(2)",
        )

        assert result.success is False
        assert "literal escaped" in result.content
        assert result.metadata["invalid_python_format"] is True

    def test_write_inside_venv_is_blocked(self, project):
        tool = CodeInterpreterTool()

        file_path = project / ".venv" / "blocked.py"

        result = tool.execute(
            operation="write_file",
            path=str(file_path),
            content="print('blocked')\n",
        )

        assert result.success is False
        assert ".venv" in result.content

    def test_path_outside_workspace_is_rejected(self, project):
        tool = CodeInterpreterTool()

        outside_path = Path(project).parent.parent / "outside.py"

        with pytest.raises(ValueError):
            tool.execute(
                operation="write_file",
                path=str(outside_path),
                content="print('outside')\n",
            )

    def test_create_venv(self, project):
        tool = CodeInterpreterTool()

        result = tool.execute(
            operation="create_venv",
            path=str(project),
        )

        assert result.success is True

        venv_path = project / ".venv"

        assert venv_path.exists()

        if __import__("sys").platform == "win32":
            assert (venv_path / "Scripts" / "python.exe").exists()
        else:
            assert (venv_path / "bin" / "python").exists()

    def test_run_python_success(self, project):
        tool = CodeInterpreterTool()

        script = project / "calculator.py"

        script.write_text(
            "print(2 + 2)\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_python",
            path=str(script),
        )

        assert result.success is True
        assert "4" in result.content
        assert result.metadata["operation"] == "run_python"
        assert result.metadata["returncode"] == 0
        assert result.metadata["stdout_present"] is True

    def test_run_python_with_stdin(self, project):
        tool = CodeInterpreterTool()

        script = project / "calculator.py"

        script.write_text(
            "name = input()\n"
            "age = input()\n"
            "print(f'Hello {name}, age {age}')\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_python",
            path=str(script),
            stdin="Jeevan\n25\n",
        )

        assert result.success is True
        assert "Hello Jeevan, age 25" in result.content
        assert result.metadata["returncode"] == 0

    def test_run_python_failure_reports_nonzero_returncode(self, project):
        tool = CodeInterpreterTool()

        script = project / "failure.py"

        script.write_text(
            "raise ValueError('boom')\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_python",
            path=str(script),
        )

        assert result.success is False
        assert "ValueError" in result.content
        assert result.metadata["returncode"] != 0
        assert result.metadata["operation"] == "run_python"

    def test_run_python_timeout(self, project):
        setup_tool = CodeInterpreterTool(timeout=30)

        create_result = setup_tool.execute(
            operation="create_venv",
            path=str(project),
        )

        assert create_result.success is True

        tool = CodeInterpreterTool(timeout=2)

        assert create_result.success is True

        script = project / "slow.py"

        script.write_text(
            "import time\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_python",
            path=str(script),
        )

        assert result.success is False
        assert "timed out" in result.content.lower()
        assert result.metadata["timeout"] is True

    def test_run_tests_success(self, project):
        tool = CodeInterpreterTool()

        test_file = project / "test_calculator.py"

        test_file.write_text(
            "import unittest\n\n"
            "class TestCalculator(unittest.TestCase):\n"
            "    def test_addition(self):\n"
            "        self.assertEqual(2 + 2, 4)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_tests",
            path=str(project),
        )

        assert result.success is True
        assert result.metadata["operation"] == "run_tests"
        assert result.metadata["returncode"] == 0

    def test_run_tests_failure(self, project):
        tool = CodeInterpreterTool()

        test_file = project / "test_calculator.py"

        test_file.write_text(
            "import unittest\n\n"
            "class TestCalculator(unittest.TestCase):\n"
            "    def test_failure(self):\n"
            "        self.assertEqual(2 + 2, 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="run_tests",
            path=str(project),
        )

        assert result.success is False
        assert result.metadata["operation"] == "run_tests"
        assert result.metadata["returncode"] != 0

    def test_verify_startup(self, project):
        tool = CodeInterpreterTool()

        script = project / "app.py"

        script.write_text(
            "import time\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="verify_startup",
            path=str(script),
            startup_timeout=1,
        )

        assert result.success is True
        assert result.metadata["operation"] == "verify_startup"
        assert result.metadata["startup_verified"] is True

    def test_list_files(self, project):
        tool = CodeInterpreterTool()

        (project / "calculator.py").write_text(
            "print(2 + 2)\n",
            encoding="utf-8",
        )

        result = tool.execute(
            operation="list_files",
            path=str(project),
        )

        assert result.success is True
        assert "calculator.py" in result.content

    def test_run_command(self, project):
        tool = CodeInterpreterTool()

        result = tool.execute(
            operation="run_command",
            path=str(project),
            command="python -c \"print(2 + 2)\"",
        )

        assert result.success is True
        assert "4" in result.content
        assert result.metadata["operation"] == "run_command"
        assert result.metadata["returncode"] == 0

    def test_structured_object_text_opt_in(self):
        tool = CodeInterpreterTool()

        assert (
            tool.spec.metadata["structured_allow_object_text"]
            is True
        )

    def test_tool_id(self):
        tool = CodeInterpreterTool()

        assert tool.tool_id == "code_interpreter"

    def test_registry_registration(self):
        ToolRegistry.register_value(
            "code_interpreter",
            CodeInterpreterTool,
        )

        assert ToolRegistry.contains("code_interpreter")
from types import SimpleNamespace

from openjarvis.agents.goal_contract import GoalContract


def result(
    tool_name: str,
    operation: str,
    *,
    success: bool = True,
    path: str = "",
    returncode: int = 0,
    startup_verified: bool = False,
):
    return SimpleNamespace(
        tool_name=tool_name,
        success=success,
        content="test output",
        metadata={
            "operation": operation,
            "path": path,
            "returncode": returncode,
            "startup_verified": startup_verified,
        },
    )


def test_calculator_contract_requires_full_lifecycle():
    contract = GoalContract.from_goal(
        "Create a folder named calculator on my desktop "
        "and inside it create a fully functioning Python calculator. "
        "Test it and verify that it works."
    )

    assert contract.software_task is True
    assert contract.requires_folder is True
    assert contract.requires_python_source is True
    assert contract.requires_tests is True
    assert contract.requires_application_execution is True
    assert contract.gui_application is False


def test_folder_creation_alone_is_not_complete():
    contract = GoalContract.from_goal(
        "Create a folder named calculator and create a Python calculator inside it."
    )

    results = [
        result(
            "folder_create",
            "folder_create",
            path=r"C:\Users\JEEVAN\Desktop\calculator",
        )
    ]

    complete, evidence = contract.verify(results)

    assert evidence["folder_created"] is True
    assert complete is False


def test_source_creation_without_tests_is_not_complete():
    contract = GoalContract.from_goal(
        "Create a fully functioning Python calculator inside a folder and test it."
    )

    results = [
        result(
            "folder_create",
            "folder_create",
            path=r"C:\Users\JEEVAN\Desktop\calculator",
        ),
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\calculator\calculator.py",
        ),
    ]

    complete, evidence = contract.verify(results)

    assert evidence["folder_created"] is True
    assert evidence["python_source_created"] is True
    assert evidence["tests_created"] is False
    assert evidence["tests_passed"] is False
    assert complete is False


def test_tests_must_pass_before_completion():
    contract = GoalContract.from_goal(
        "Create a fully functioning Python calculator and test it."
    )

    results = [
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\calculator.py",
        ),
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\test_calculator.py",
        ),
        result(
            "code_interpreter",
            "run_tests",
            returncode=1,
            success=False,
        ),
        result(
            "code_interpreter",
            "run_python",
            returncode=0,
        ),
    ]

    complete, evidence = contract.verify(results)

    assert evidence["tests_created"] is True
    assert evidence["tests_passed"] is False
    assert evidence["application_executed"] is True
    assert complete is False


def test_complete_cli_application_requires_tests_and_execution():
    contract = GoalContract.from_goal(
        "Create a fully functioning Python calculator, test it, "
        "and verify that it works."
    )

    results = [
        result(
            "folder_create",
            "folder_create",
            path=r"C:\Users\JEEVAN\Desktop\calculator",
        ),
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\calculator\calculator.py",
        ),
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\calculator\test_calculator.py",
        ),
        result(
            "code_interpreter",
            "run_tests",
            returncode=0,
        ),
        result(
            "code_interpreter",
            "run_python",
            returncode=0,
        ),
    ]

    complete, evidence = contract.verify(results)

    assert evidence["folder_created"] is True
    assert evidence["python_source_created"] is True
    assert evidence["tests_created"] is True
    assert evidence["tests_passed"] is True
    assert evidence["application_executed"] is True
    assert complete is True


def test_gui_application_requires_startup_verification():
    contract = GoalContract.from_goal(
        "Create a fully functioning Tkinter GUI application and test it."
    )

    results = [
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\app.py",
        ),
        result(
            "code_interpreter",
            "write_file",
            path=r"C:\Users\JEEVAN\Desktop\test_app.py",
        ),
        result(
            "code_interpreter",
            "run_tests",
            returncode=0,
        ),
        result(
            "code_interpreter",
            "verify_startup",
            startup_verified=True,
        ),
    ]

    complete, evidence = contract.verify(results)

    assert evidence["startup_verified"] is True
    assert evidence["application_executed"] is False
    assert complete is True

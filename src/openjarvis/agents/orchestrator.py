"""OrchestratorAgent â€” multi-turn agent with tool-calling loop.

Supports two modes:

- **function_calling** (default): Uses OpenAI-format tool definitions and
  parses ``tool_calls`` from the engine response.
- **structured**: Uses a THOUGHT/TOOL/INPUT/FINAL_ANSWER text format
  (like ReAct) with a canonical system prompt from the orchestrator
  prompt registry.  This is the format used by the SFT/GRPO training
  pipelines, making the Orchestrator a distinctive trainable agent type.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from typing import Any, Callable, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.agents.goal_contract import GoalContract
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool

logger = logging.getLogger(__name__)


@AgentRegistry.register("orchestrator")
class OrchestratorAgent(ToolUsingAgent):
    """Multi-turn agent that routes between tools and the LLM.

    Implements a tool-calling loop:
    1. Send messages with tool definitions to the engine.
    2. If the response contains tool_calls, execute them and loop.
    3. If no tool_calls, return the final answer.
    4. Stop after ``max_turns`` iterations.

    In **structured** mode the agent instead uses a
    ``THOUGHT: / TOOL: / INPUT: / FINAL_ANSWER:`` text protocol
    identical to the format used by the orchestrator SFT/GRPO
    training pipelines.
    """

    agent_id = "orchestrator"
    _default_temperature = 0.7
    _default_max_tokens = 1024
    _default_max_turns = 10

    _default_system_prompt = """\
    You are the Jarvis orchestrator. You solve user requests by using the
    available tools correctly and completely.

    === SOFTWARE CREATION RULES ===
    When the user asks you to create, build, develop, implement, or write
    software:

    1. Create the COMPLETE requested implementation.
    2. Never create a placeholder, stub, demo-only file, welcome-message-only
    file, or TODO-only implementation when the user requested a functional
    application or program.
    3. When writing source code with a file-writing tool, provide the COMPLETE
    source code in the tool arguments.
    4. For a Python application, implement the actual requested functionality,
    including required functions, application logic, user interaction,
    input/error handling, and the program entry point.
    5. If the user requests a folder and files, create the folder first and then
    create the requested files with their complete contents.
    6. Do NOT automatically run, execute, test, verify, or validate newly
    created software unless the user explicitly requests it.
    7. A successful write_file operation means the requested file was saved. It
    does not mean that the application was tested or executed.

    For example, if the user asks for a fully functioning Python calculator,
    write an actual calculator implementation. Do not write only:

    print("Hello, this is a fully functioning Python application!")

    The source file must contain the requested calculator functionality.

    Use the available tools to complete the user's request. Do not claim that
    something was created or saved unless the corresponding tool operation
    actually succeeded.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mode: str = "function_calling",
        system_prompt: Optional[str] = None,
        prompt_builder: Optional[Any] = None,
        parallel_tools: bool = True,
        interactive: bool = False,
        confirm_callback=None,
        before_tool_call: Optional[Callable[[str, dict[str, Any]], bool]] = None,
        tool_choice: Optional[str] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            interactive=interactive,
            confirm_callback=confirm_callback,
            prompt_builder=prompt_builder,
        )
        self._mode = mode
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else self._default_system_prompt
        )
        self._parallel_tools = parallel_tools
        self._before_tool_call = before_tool_call
        self._tool_choice = tool_choice

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        if self._mode == "structured":
            return self._run_structured(input, context, **kwargs)
        return self._run_function_calling(input, context, **kwargs)

    # ------------------------------------------------------------------
    # Governance hook
    # ------------------------------------------------------------------

    @staticmethod
    def _governance_denial(tool_name: str, reason: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            content=(
                f"[Governance] Tool '{tool_name}' was not approved ({reason}). "
                "Adjust your plan and try a different approach."
            ),
            success=False,
        )

    def _creation_only_tool_denial(
        self,
        tool_call: ToolCall,
        goal_contract: GoalContract,
    ) -> Optional[ToolResult]:
        """Block execution/testing tools when the user requested creation only."""

        if not goal_contract.software_task:
            return None

        if tool_call.name != "code_interpreter":
            return None

        try:
            arguments = json.loads(tool_call.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            return None

        operation = str(arguments.get("operation", "")).lower().strip()

        if operation == "run_tests" and not goal_contract.requires_tests:
            return ToolResult(
                tool_name=tool_call.name,
                content=(
                    "Blocked: this is a creation-only software request. "
                    "The user did not request testing."
                ),
                success=False,
                metadata={
                    "operation": operation,
                    "blocked_by_goal_contract": True,
                },
            )

        if operation == "run_python" and not goal_contract.requires_application_execution:
            return ToolResult(
                tool_name=tool_call.name,
                content=(
                    "Blocked: this is a creation-only software request. "
                    "The user did not request application execution."
                ),
                success=False,
                metadata={
                    "operation": operation,
                    "blocked_by_goal_contract": True,
                },
            )

        if operation == "verify_startup" and not goal_contract.requires_application_execution:
            return ToolResult(
                tool_name=tool_call.name,
                content=(
                    "Blocked: this is a creation-only software request. "
                    "The user did not request application verification."
                ),
                success=False,
                metadata={
                    "operation": operation,
                    "blocked_by_goal_contract": True,
                },
            )

        return None

    def _check_tool_allowed(self, tc: ToolCall) -> Optional[ToolResult]:
        """Call before_tool_call hook if set.

        Returns None to allow execution, or a denial ToolResult to inject
        instead of running the tool. Invalid arguments and hook failures deny
        execution so a governance integration cannot fail open.
        """
        if self._before_tool_call is None:
            return None

        try:
            tool_args = json.loads(tc.arguments) if tc.arguments else {}
        except (json.JSONDecodeError, TypeError):
            return self._governance_denial(tc.name, "invalid tool arguments")
        if not isinstance(tool_args, dict):
            return self._governance_denial(tc.name, "tool arguments are not an object")

        try:
            allowed = self._before_tool_call(tc.name, tool_args)
        except Exception:
            logger.exception("before_tool_call hook failed for tool %s", tc.name)
            return self._governance_denial(tc.name, "governance check failed")
        if allowed:
            return None
        return self._governance_denial(tc.name, "policy denied the call")

    # ------------------------------------------------------------------
    # Structured mode (THOUGHT/TOOL/INPUT/FINAL_ANSWER)
    # ------------------------------------------------------------------

    def _run_structured(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Build system prompt
        if self._system_prompt:
            sys_prompt = self._system_prompt
        else:
            from openjarvis.learning.intelligence.orchestrator.prompt_registry import (
                build_system_prompt,
            )

            sys_prompt = build_system_prompt(tools=self._tools)

        messages = self._build_messages(input, context, system_prompt=sys_prompt)

        all_tool_results: list[ToolResult] = []
        turns = 0

        for _turn in range(self._max_turns):
            turns += 1

            if self._loop_guard:
                messages = self._loop_guard.compress_context(messages)

            # Keep every tool-calling turn within the prompt budget. Tool results
            # are appended after each call and can otherwise grow the context again.
            messages = self._trim_conversation_to_budget(
                messages,
                max_prompt_tokens=3072,
            )

            result = self._generate(messages)
            content = result.get("content", "")

            parsed = self._parse_structured_response(content)

            # FINAL_ANSWER -> done
            if parsed["final_answer"]:
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content=parsed["final_answer"],
                    tool_results=all_tool_results,
                    turns=turns,
                )

            # TOOL -> execute
            if parsed["tool"]:
                messages.append(Message(role=Role.ASSISTANT, content=content))

                tool_call = ToolCall(
                    id=f"orch_{turns}",
                    name=parsed["tool"],
                    arguments=self._normalize_structured_tool_input(
                        parsed["tool"],
                        parsed["input"],
                    ),
                )
                denial = self._check_tool_allowed(tool_call)
                if denial is not None:
                    tool_result = denial
                else:
                    tool_result = self._executor.execute(tool_call)
                all_tool_results.append(tool_result)

                if tool_result.success:
                    observation = f"Observation: {tool_result.content}"
                else:
                    observation = (
                        f"Observation: Tool '{tool_result.tool_name}' failed: "
                        f"{tool_result.content}"
                    )
                messages.append(Message(role=Role.USER, content=observation))
                continue

            # Neither -> treat content as final answer
            self._emit_turn_end(turns=turns)
            return AgentResult(
                content=content,
                tool_results=all_tool_results,
                turns=turns,
            )

        # Max turns exceeded
        return self._max_turns_result(all_tool_results, turns)

    def _normalize_structured_tool_input(
        self,
        tool_name: str,
        raw_input: str,
    ) -> str:
        """Map unambiguous structured text input to a string parameter."""
        if not raw_input:
            return "{}"

        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            invalid_json = True
            string_value = raw_input
        else:
            invalid_json = False
            if isinstance(parsed_input, dict):
                return raw_input
            # INPUT is a text protocol. A non-object JSON value such as 42,
            # true, null, or [1, 2] may still be the intended text for a tool's
            # string parameter. Quoted JSON strings are decoded to remove only
            # their surrounding quotes; other values retain their source text.
            string_value = parsed_input if isinstance(parsed_input, str) else raw_input

        tool_spec = None
        for candidate in reversed(self._tools):
            candidate_spec = candidate.spec
            if candidate_spec.name == tool_name:
                tool_spec = candidate_spec
                break
        if tool_spec is None:
            return raw_input

        parameters = tool_spec.parameters
        parameter_container_type = parameters.get("type")
        if parameter_container_type not in (None, "object"):
            return raw_input

        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return raw_input

        if len(required) == 1 and required[0] in properties:
            parameter_name = required[0]
        elif not required and len(properties) == 1:
            parameter_name = next(iter(properties))
        else:
            return raw_input

        parameter_schema = properties[parameter_name]
        if not isinstance(parameter_schema, dict):
            return raw_input
        parameter_type = parameter_schema.get("type")
        accepts_string = parameter_type == "string" or (
            isinstance(parameter_type, list) and "string" in parameter_type
        )
        if not accepts_string:
            return raw_input

        allow_object_text = (
            tool_spec.metadata.get("structured_allow_object_text") is True
        )
        starts_like_object = raw_input.lstrip("\ufeff \t\r\n").startswith("{")
        if invalid_json and starts_like_object and not allow_object_text:
            return raw_input

        return json.dumps({parameter_name: string_value})

    @staticmethod
    def _parse_structured_response(text: str) -> dict:
        """Parse THOUGHT/TOOL/INPUT/FINAL_ANSWER from model output."""
        result = {
            "thought": "",
            "tool": "",
            "input": "",
            "final_answer": "",
        }

        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=\nTOOL:|\nFINAL[_ ]?ANSWER:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        final_match = re.search(
            r"FINAL[_ ]?ANSWER:\s*(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        tool_match = re.search(r"TOOL:\s*(.+)", text, re.IGNORECASE)
        if tool_match:
            result["tool"] = tool_match.group(1).strip()

        input_match = re.search(
            r"INPUT:\s*(.+?)(?=\nTHOUGHT:|\nTOOL:|\nFINAL|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            result["input"] = input_match.group(1).strip()

        return result

    # ------------------------------------------------------------------
    # Function-calling mode (original behaviour)
    # ------------------------------------------------------------------
    def _tool_schema_prompt_tokens(self, tools: list[dict[str, Any]]) -> int:
        """Estimate tokens consumed by OpenAI tool definitions."""
        if not tools:
            return 0

        raw = json.dumps(
            tools,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return max(1, len(raw) // 4)

    def _check_goal_tool_allowed(
        self,
        goal_contract: GoalContract,
        tool_call: ToolCall,
    ) -> Optional[ToolResult]:
        """Block execution/testing tools not requested by the user."""

        if not goal_contract.software_task:
            return None

        if tool_call.name != "code_interpreter":
            return None

        try:
            arguments = json.loads(tool_call.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            return None

        operation = str(arguments.get("operation", "")).lower()

        if operation == "run_tests" and not goal_contract.requires_tests:
            return ToolResult(
                tool_name=tool_call.name,
                content=(
                    "Tool execution blocked by the goal contract: "
                    "the user did not request testing."
                ),
                success=False,
                metadata={
                    "operation": operation,
                    "blocked_by_goal_contract": True,
                },
            )

        if operation in {"run_python", "verify_startup"}:
            if not goal_contract.requires_application_execution:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=(
                        "Tool execution blocked by the goal contract: "
                        "the user did not request application execution "
                        "or startup verification."
                    ),
                    success=False,
                    metadata={
                        "operation": operation,
                        "blocked_by_goal_contract": True,
                    },
                )

        return None

    def _creation_only_tool_denial(
        self,
        goal_contract: GoalContract,
        tool_call: ToolCall,
    ) -> Optional[ToolResult]:
        """Block execution/testing tools for creation-only software goals."""

        if not goal_contract.software_task:
            return None

        if (
            goal_contract.requires_tests
            or goal_contract.requires_application_execution
        ):
            return None

        if tool_call.name != "code_interpreter":
            return None

        try:
            arguments = json.loads(tool_call.arguments or "{}")
        except (TypeError, ValueError):
            return None

        operation = str(arguments.get("operation", "")).lower()

        blocked_operations = {
            "run_python",
            "run_tests",
            "verify_startup",
            "run_command",
        }

        if operation not in blocked_operations:
            return None

        return ToolResult(
            tool_name=tool_call.name,
            content=(
                f"Blocked code_interpreter operation '{operation}': "
                "this is a creation-only request. The user did not ask "
                "to run, test, verify, or execute the application."
            ),
            success=False,
            metadata={
                "operation": operation,
                "blocked_by_goal_contract": True,
                "creation_only": True,
            },
        )

    def _goal_is_verified(
        self,
        goal_contract: GoalContract,
        tool_results: list[ToolResult],
    ) -> bool:
        """Determine whether the goal contract is satisfied."""

        if not tool_results:
            return False

        return goal_contract.verify(tool_results)

    def _goal_tool_denial(
        self,
        goal_contract: GoalContract,
        tool_call: ToolCall,
    ) -> Optional[ToolResult]:
        """Block execution/verification tools not requested by the user."""

        if not goal_contract.software_task:
            return None

        if tool_call.name != "code_interpreter":
            return None

        try:
            arguments = json.loads(tool_call.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            return None

        operation = str(arguments.get("operation", "")).lower()

        execution_operations = {
            "run_python",
            "run_command",
            "verify_startup",
        }

        test_operations = {
            "run_tests",
        }

        if operation in execution_operations:
            if not goal_contract.requires_application_execution:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=(
                        f"Blocked code_interpreter operation '{operation}'. "
                        "The user requested software creation only and did "
                        "not request execution or verification."
                    ),
                    success=False,
                    metadata={
                        "operation": operation,
                        "goal_guard": True,
                        "blocked": True,
                    },
                )

        if operation in test_operations:
            if not goal_contract.requires_tests:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=(
                        f"Blocked code_interpreter operation '{operation}'. "
                        "The user requested software creation only and did "
                        "not request testing or verification."
                    ),
                    success=False,
                    metadata={
                        "operation": operation,
                        "goal_guard": True,
                        "blocked": True,
                    },
                )

        return None

    def _tools_for_goal(self, goal: str) -> list[BaseTool]:
        """Return tools relevant to the current user goal."""

        goal_lower = goal.lower()

        diagnostic_keywords = (
            "computer",
            "system",
            "pc",
            "laptop",
            "cpu",
            "ram",
            "memory usage",
            "gpu",
            "vram",
            "disk usage",
            "disk space",
            "performance",
            "slow",
            "diagnose",
            "diagnostic",
            "diagnostics",
            "system health",
            "process",
            "processes",
            "service",
            "services",
            "resource usage",
        )

        diagnostics_requested = any(
            keyword in goal_lower
            for keyword in diagnostic_keywords
        )

        diagnostic_tools = {
            "computer_diagnostics",
            "process_inspect",
            "service_inspect",
        }

        filtered_tools: list[BaseTool] = []

        for tool in self._tools:
            tool_name = tool.spec.name

            if (
                tool_name in diagnostic_tools
                and not diagnostics_requested
            ):
                continue

            filtered_tools.append(tool)

        return filtered_tools

    def _failure_signature(self, tool_result: ToolResult) -> str:
        """Create a stable signature for a failed tool result.

        The signature intentionally focuses on the tool and the actual
        failure message so repeated failures can be detected even when
        the model changes the tool arguments slightly.
        """

        if tool_result.success:
            return ""

        content = str(tool_result.content or "").lower()

        # Normalize whitespace so formatting changes do not create
        # artificial differences.
        content = " ".join(content.split())

        # Focus on the actual error rather than the full tool output.
        error_markers = (
            "syntaxerror:",
            "traceback",
            "eoferror:",
            "filenotfounderror:",
            "modulenotfounderror:",
            "importerror:",
            "typeerror:",
            "nameerror:",
            "attributeerror:",
            "permissionerror:",
            "timed out",
            "timeout",
            "file not found",
            "not found",
        )

        for marker in error_markers:
            index = content.find(marker)
            if index >= 0:
                content = content[index:]
                break

        # Keep the signature bounded so it cannot consume context.
        content = content[:500]

        return f"{tool_result.tool_name.lower()}:{content}"

    def _run_function_calling(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        goal_contract = GoalContract.from_goal(input)

        # Build initial messages
        goal_tools = self._tools_for_goal(input)

        openai_tools = (
            self._executor.get_openai_tools(goal_tools)
            if goal_tools
            else []
        )

        tool_schema_tokens = self._tool_schema_prompt_tokens(openai_tools)

        # llama.cpp n_ctx is currently 4096. Reserve space for the model's
        # response so input + tool definitions cannot consume the entire context.
        context_limit = 4096
        output_reservation = min(self._max_tokens, 1024)

        message_budget = max(
            512,
            context_limit - tool_schema_tokens - output_reservation,
        )

        messages = self._build_messages(
            input,
            context,
            system_prompt=self._system_prompt,
            max_prompt_tokens=message_budget,
        )

        all_tool_results: list[ToolResult] = []
        turns = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        required_tool_next_turn = False

        # Goal-driven loop state
        goal_completed = False
        goal_verified = False
        last_tool_name = ""
        last_tool_success = False
        consecutive_failures = 0
        consecutive_no_tool_turns = 0

        # Prevent identical tool calls from being repeated indefinitely.
        tool_call_history: list[str] = []

        # Track actual failure states, not only identical tool calls.
        failure_signature_history: list[str] = []

        for _turn in range(self._max_turns):
            turns += 1

            # Stop if the agent has failed repeatedly without making progress.
            if consecutive_failures >= 3:
                logger.warning(
                    "Stopping goal loop after %s consecutive tool failures",
                    consecutive_failures,
                )
                break

            if self._loop_guard:
                messages = self._loop_guard.compress_context(messages)

            # Keep every tool-calling turn within the prompt budget. Tool results
            # are appended after each call and can otherwise grow the context again.
            messages = self._trim_conversation_to_budget(
                messages,
                max_prompt_tokens=3072,
            )

            # Build generate kwargs
            gen_kwargs: dict[str, Any] = {}

            if openai_tools:
                gen_kwargs["tools"] = openai_tools

            if openai_tools:
                if required_tool_next_turn:
                    gen_kwargs["tool_choice"] = "required"
                    required_tool_next_turn = False
                elif self._tool_choice:
                    gen_kwargs["tool_choice"] = self._tool_choice

            result = self._generate(messages, **gen_kwargs)

            # Accumulate token usage
            usage = result.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            content = result.get("content", "")
            raw_tool_calls = result.get("tool_calls", [])

            # ---------------------------------------------------------
            # No tool calls
            # ---------------------------------------------------------
            if not raw_tool_calls:
                consecutive_no_tool_turns += 1

                last_result = (
                    all_tool_results[-1]
                    if all_tool_results
                    else None
                )

                last_metadata = (
                    last_result.metadata
                    if last_result
                    else {}
                )

                goal_verified = False

                if all_tool_results:
                    goal_verified = self._goal_is_verified(
                        goal_contract,
                        all_tool_results,
                    )

                if not goal_contract.software_task:
                    content = self._check_continuation(
                        result,
                        messages,
                    )
                    content = self._strip_think_tags(content)

                    self._emit_turn_end(
                        turns=turns,
                        content_length=len(content),
                    )

                    return AgentResult(
                        content=content,
                        tool_results=all_tool_results,
                        turns=turns,
                        metadata={
                            "goal_completed": False,
                            "goal_verified": False,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": (
                                total_prompt_tokens
                                + total_completion_tokens
                            ),
                        },
                    )

                # -----------------------------------------------------
                # Goal verified -> allow final answer
                # -----------------------------------------------------
                if goal_verified:
                    goal_completed = True

                    content = self._check_continuation(
                        result,
                        messages,
                    )
                    content = self._strip_think_tags(content)

                    self._emit_turn_end(
                        turns=turns,
                        content_length=len(content),
                    )

                    return AgentResult(
                        content=content,
                        tool_results=all_tool_results,
                        turns=turns,
                        metadata={
                            "goal_completed": True,
                            "goal_verified": True,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": (
                                total_prompt_tokens
                                + total_completion_tokens
                            ),
                        },
                    )

                # -----------------------------------------------------
                # Goal NOT verified -> continue working
                # -----------------------------------------------------

                goal_completed = False

                if (
                    last_result
                    and last_metadata.get("operation") == "run_python"
                    and not last_result.success
                    and "timed out"
                    in (last_result.content or "").lower()
                ):
                    continuation = (
                        "The user's goal has NOT been verified as complete.\n"
                        "The previous Python execution timed out.\n"
                        "The program may be waiting for interactive input.\n"
                        "Do NOT repeat run_python without input.\n"
                        "Inspect the existing Python program and determine "
                        "whether it uses input(). If it does, run it again "
                        "using the stdin parameter with valid test input, "
                        "using newline characters between input values.\n"
                        "Do NOT unnecessarily rewrite the existing file.\n"
                        "After execution, inspect the actual stdout and "
                        "stderr and verify that the requested behavior works.\n"
                        "Do not provide a final answer until the goal is "
                        "actually verified."
                    )

                elif (
                    last_result
                    and last_metadata.get("operation") == "run_python"
                    and last_result.success
                    and last_metadata.get("returncode") == 0
                    and not last_metadata.get("stdout_present", False)
                    and not last_metadata.get("stderr_present", False)
                ):
                    continuation = (
                        "The user's goal has NOT been verified as complete.\n"
                        "The previous Python execution exited successfully, "
                        "but it produced no stdout or stderr, so the requested "
                        "behavior has not been demonstrated.\n"
                        "Continue working on the existing project. Do not "
                        "recreate or unnecessarily overwrite working files.\n"
                        "Inspect the existing implementation, determine what "
                        "is missing, correct it, and then execute or test it again.\n"
                        "Do not provide a final answer until the goal is "
                        "actually verified."
                    )

                else:
                    continuation = (
                        "The user's goal has NOT been verified as complete.\n"
                        "Inspect the previous tool result and continue working "
                        "toward the user's goal.\n"
                        "A successful process exit, welcome message, menu, "
                        "or startup message alone is NOT proof that the "
                        "requested functionality works.\n"
                        "If the implementation is incomplete, inspect the "
                        "existing files and correct them.\n"
                        "If an interactive Python program requires input, "
                        "use run_python with the stdin parameter.\n"
                        "After making corrections, execute or test the "
                        "program and inspect the actual output.\n"
                        "Verify the requested behavior and important edge "
                        "cases before reporting completion.\n"
                        "Do not unnecessarily recreate or overwrite "
                        "working files.\n"
                        "Do not provide a final answer until the goal is "
                        "actually verified."
                    )

                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=content,
                    )
                )

                messages.append(
                    Message(
                        role=Role.USER,
                        content=continuation,
                    )
                )

                required_tool_next_turn = True
                continue


            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"orch_{turns}_{i}"),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", "{}"),
                )
                for i, tc in enumerate(raw_tool_calls)
            ]

            # Append assistant message with tool calls
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls,
                )
            )

            # Execute each tool (with loop guard check) and append results
            if self._parallel_tools and len(tool_calls) > 1:
                # Governance callbacks often wrap stateful policy engines and
                # are not assumed to be thread-safe. Evaluate them in request
                # order before dispatching the approved tool work in parallel.
                results_map: dict[int, tuple[ToolCall, ToolResult]] = {}
                approved_calls: list[ToolCall] = []
                for tc in tool_calls:
                    goal_denial = self._check_goal_tool_allowed(
                        goal_contract,
                        tc,
                    )

                    if goal_denial is not None:
                        results_map[id(tc)] = (tc, goal_denial)
                        continue

                    denial = self._check_tool_allowed(tc)

                    if denial is None:
                        approved_calls.append(tc)
                    else:
                        results_map[id(tc)] = (tc, denial)

                def _exec_tool(tc: ToolCall) -> tuple:
                    if self._loop_guard:
                        verdict = self._loop_guard.check_call(
                            tc.name,
                            tc.arguments,
                        )
                        if verdict.blocked:
                            return tc, ToolResult(
                                tool_name=tc.name,
                                content=f"Loop guard: {verdict.reason}",
                                success=False,
                            )
                    return tc, self._executor.execute(tc)

                if approved_calls:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(approved_calls),
                    ) as pool:
                        futures = {
                            pool.submit(_exec_tool, tc): tc for tc in approved_calls
                        }
                        for future in concurrent.futures.as_completed(futures):
                            tc_orig = futures[future]
                            results_map[id(tc_orig)] = future.result()

                # Append results in original order
                for tc in tool_calls:
                    _, tool_result = results_map[id(tc)]
                    all_tool_results.append(tool_result)
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=tool_result.content,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
            else:
                # Sequential execution
                for tc in tool_calls:
                    # ---------------------------------------------------------
                    # Prevent repeated identical tool calls
                    # ---------------------------------------------------------
                    call_signature = f"{tc.name}:{tc.arguments}"

                    if tool_call_history.count(call_signature) >= 2:
                        logger.warning(
                            "Repeated tool call detected: %s",
                            call_signature,
                        )

                        repeated_result = ToolResult(
                            tool_name=tc.name,
                            content=(
                                "This exact tool call has already been attempted "
                                "multiple times. Do not repeat it. Inspect the "
                                "previous result and take a different corrective action."
                            ),
                            success=False,
                        )

                        all_tool_results.append(repeated_result)

                        messages.append(
                            Message(
                                role=Role.TOOL,
                                content=repeated_result.content,
                                tool_call_id=tc.id,
                                name=tc.name,
                            )
                        )

                        consecutive_failures += 1
                        continue

                    # Record this call before execution.
                    tool_call_history.append(call_signature)

                    # Governance hook check before execution
                    denial = self._check_tool_allowed(tc)
                    if denial is not None:
                        all_tool_results.append(denial)
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                content=denial.content,
                                tool_call_id=tc.id,
                                name=tc.name,
                            )
                        )
                        continue

                    # Loop guard check before execution
                    if self._loop_guard:
                        verdict = self._loop_guard.check_call(
                            tc.name,
                            tc.arguments,
                        )
                        if verdict.blocked:
                            tool_result = ToolResult(
                                tool_name=tc.name,
                                content=f"Loop guard: {verdict.reason}",
                                success=False,
                            )
                            all_tool_results.append(tool_result)
                            messages.append(
                                Message(
                                    role=Role.TOOL,
                                    content=tool_result.content,
                                    tool_call_id=tc.id,
                                    name=tc.name,
                                )
                            )
                            continue

                    tool_result = self._executor.execute(tc)
                    all_tool_results.append(tool_result)

                    last_tool_name = tc.name
                    last_tool_success = tool_result.success

                    if tool_result.success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    # Append actual tool result FIRST.
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=tool_result.content,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

                    # Then add recovery instruction if the same failure happened again.
                    if not tool_result.success:
                        failure_signature = self._failure_signature(tool_result)

                        if failure_signature:
                            failure_signature_history.append(failure_signature)

                            if failure_signature_history.count(failure_signature) >= 2:
                                logger.warning(
                                    "Repeated failure state detected: %s",
                                    failure_signature,
                                )

                                messages.append(
                                    Message(
                                        role=Role.USER,
                                        content=(
                                            "The previous correction did not resolve the "
                                            "failure. Do NOT repeat the same correction. "
                                            "Inspect the actual error and current file/state, "
                                            "then choose a different corrective action. "
                                            "Do not claim the goal is complete while "
                                            "verification is failing."
                                        ),
                                    )
                                )
                                        # ---------------------------------------------------------
            # Check goal completion immediately after tool execution.
            #
            # For creation-only software goals, successful creation
            # evidence is sufficient. Do not give the LLM another turn
            # where it may unnecessarily execute or test the application.
            # ---------------------------------------------------------
            if all_tool_results:
                goal_verified = self._goal_is_verified(
                    goal_contract,
                    all_tool_results,
                )

                if goal_verified:
                    goal_completed = True

                    # Tool-call turns carry no text. Ask the model (no tools)
                    # for a short confirmation so the user gets a real reply.
                    if not (content or "").strip():
                        try:
                            summary = self._generate(messages)
                            content = summary.get("content", "") or ""
                            usage = summary.get("usage", {})
                            total_prompt_tokens += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get(
                                "completion_tokens", 0
                            )
                        except Exception:
                            content = ""

                    final_content = self._strip_think_tags(content)
                    if not final_content.strip():
                        # Never return silence: fall back to the tool's result.
                        final_content = all_tool_results[-1].content or "Done."

                    self._emit_turn_end(
                        turns=turns,
                        content_length=len(final_content),
                    )

                    return AgentResult(
                        content=final_content,
                        tool_results=all_tool_results,
                        turns=turns,
                        metadata={
                            "goal_completed": True,
                            "goal_verified": True,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": (
                                total_prompt_tokens
                                + total_completion_tokens
                            ),
                        },
                    )
        # Max turns exceeded
        final_content = self._strip_think_tags(content) if content else ""
        self._emit_turn_end(turns=turns, max_turns_exceeded=True)
        return AgentResult(
            content=final_content or "Maximum turns reached without a final answer.",
            tool_results=all_tool_results,
            turns=turns,
            metadata={
                "max_turns_exceeded": True,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        )


__all__ = ["OrchestratorAgent"]

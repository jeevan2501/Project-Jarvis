"""ABC for agent implementations.

Adapted from IPW's ``BaseAgent`` at ``src/agents/base.py``.
Provides ``BaseAgent`` with concrete helper methods for event emission,
message building, and generation, plus ``ToolUsingAgent`` intermediate
base for agents that accept tools.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openjarvis.core.config import load_config
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Conversation, Message, Role, ToolResult
from openjarvis.engine._base import estimate_prompt_tokens
from openjarvis.engine._stubs import InferenceEngine

_ALLOWED_ENGINE_OPTION_KEYS = frozenset({"num_ctx", "num_gpu"})


@dataclass(slots=True)
class AgentContext:
    """Runtime context handed to an agent on each invocation."""

    conversation: Conversation = field(default_factory=Conversation)
    tools: List[str] = field(default_factory=list)
    memory_results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    """Result returned after an agent completes a run."""

    content: str
    tool_results: List[ToolResult] = field(default_factory=list)
    turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agent implementations.

    Subclasses must be registered via
    ``@AgentRegistry.register("name")`` to become discoverable.

    Provides concrete helper methods that eliminate boilerplate in
    subclasses:

    - :meth:`_emit_turn_start` / :meth:`_emit_turn_end` -- event bus
    - :meth:`_build_messages` -- conversation + system prompt assembly
    - :meth:`_generate` -- delegates to engine with stored defaults
    - :meth:`_max_turns_result` -- standard max-turns-exceeded result
    - :meth:`_strip_think_tags` -- remove ``<think>`` blocks
    """

    agent_id: str
    accepts_tools: bool = False
    # Plain conversational agents may opt into the managed runtime's generic
    # function-calling loop.  Specialized agents keep their own execution
    # class even when process-wide MCP tools are available.
    supports_managed_tool_fallback: bool = False

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prompt_builder: Optional[Any] = None,
        engine_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._engine = engine
        self._model = model
        self._bus = bus
        self._prompt_builder = prompt_builder
        self._engine_options: Dict[str, Any] = dict(engine_options or {})

        # Three-tier resolution: explicit arg > config > class default > hardcoded
        if temperature is not None and max_tokens is not None:
            self._temperature = temperature
            self._max_tokens = max_tokens
        else:
            try:
                cfg = load_config()
                self._temperature = (
                    temperature
                    if temperature is not None
                    else cfg.intelligence.temperature
                )
                self._max_tokens = (
                    max_tokens
                    if max_tokens is not None
                    else cfg.intelligence.max_tokens
                )
            except Exception:
                self._temperature = (
                    temperature
                    if temperature is not None
                    else getattr(self, "_default_temperature", 0.7)
                )
                self._max_tokens = (
                    max_tokens
                    if max_tokens is not None
                    else getattr(self, "_default_max_tokens", 1024)
                )

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _emit_turn_start(self, input: str) -> None:
        """Publish ``AGENT_TURN_START`` if an event bus is available."""
        if self._bus:
            self._bus.publish(
                EventType.AGENT_TURN_START,
                {"agent": self.agent_id, "input": input},
            )

    def _emit_turn_end(self, **data: Any) -> None:
        """Publish ``AGENT_TURN_END`` if an event bus is available."""
        if self._bus:
            payload: Dict[str, Any] = {"agent": self.agent_id}
            payload.update(data)
            self._bus.publish(EventType.AGENT_TURN_END, payload)

    def _apply_persona(self, system_prompt: Optional[str]) -> Optional[str]:
        """Append SOUL/MEMORY/USER persona to a self-assembled system prompt.

        Agents like ``monitor_operative`` / ``operative`` build their own
        system prompt and bypass ``_build_messages`` (and thus the prompt
        builder). This lets them honor the same persona files as one-shot
        ``jarvis ask`` (#376) by *appending* persona to â€” never replacing â€”
        their specialized instructions. No-op when no ``prompt_builder`` is
        wired or no persona files exist.
        """
        if self._prompt_builder is None:
            return system_prompt
        persona = self._prompt_builder.persona_sections()
        if not persona:
            return system_prompt
        return f"{system_prompt}\n\n{persona}" if system_prompt else persona
    def _trim_conversation_to_budget(
        self,
        messages: list[Message],
        max_prompt_tokens: int = 3072,
    ) -> list[Message]:
        """Keep the system prompt and newest conversation within the prompt budget.

        The current user request is preserved. Large tool results are compacted
        when necessary so a single tool response cannot overflow the model context.
        Older conversation turns are removed first.
        """
        from openjarvis.engine._base import estimate_prompt_tokens

        if not messages:
            return messages

        if estimate_prompt_tokens(messages) <= max_prompt_tokens:
            return messages

        system_messages = [
            message
            for message in messages
            if message.role == Role.SYSTEM
        ]

        conversation = [
            message
            for message in messages
            if message.role != Role.SYSTEM
        ]

        if not conversation:
            return system_messages

        current = conversation[-1]
        history = conversation[:-1]

        # Tool results can be much larger than normal chat messages.
        # If the newest message is a tool result, compact only its content
        # instead of allowing one tool response to overflow the context.
        if current.role == Role.TOOL:
            base_messages = [*system_messages]

            # Reserve some room for the current tool-result message itself.
            available_tokens = max(
                256,
                max_prompt_tokens
                - estimate_prompt_tokens(base_messages)
                - 64,
            )

            current_tokens = estimate_prompt_tokens([current])

            if current_tokens > available_tokens:
                original_content = current.content or ""

                # Binary-search the maximum content length that fits the
                # available estimated-token budget.
                low = 0
                high = len(original_content)

                while low < high:
                    mid = (low + high + 1) // 2

                    candidate_content = (
                        original_content[: mid // 2]
                        + "\n\n[Tool output truncated to fit the model context.]\n\n"
                        + original_content[-(mid - mid // 2):]
                    )

                    candidate = Message(
                        role=current.role,
                        content=candidate_content,
                        name=current.name,
                        tool_calls=current.tool_calls,
                        tool_call_id=current.tool_call_id,
                        metadata=current.metadata,
                        images=current.images,
                    )

                    if estimate_prompt_tokens(
                        [*base_messages, candidate]
                    ) <= max_prompt_tokens:
                        low = mid
                    else:
                        high = mid - 1

                keep_chars = low

                if keep_chars < len(original_content):
                    head_chars = keep_chars // 2
                    tail_chars = keep_chars - head_chars

                    compacted_content = (
                        original_content[:head_chars]
                        + "\n\n[Tool output truncated to fit the model context.]\n\n"
                        + original_content[-tail_chars:]
                    )

                    current = Message(
                        role=current.role,
                        content=compacted_content,
                        name=current.name,
                        tool_calls=current.tool_calls,
                        tool_call_id=current.tool_call_id,
                        metadata=current.metadata,
                        images=current.images,
                    )

        kept: list[Message] = [current]

        # Add recent history while it still fits.
        for message in reversed(history):
            candidate = [*system_messages, message, *kept]

            if (
                message.role == Role.TOOL
                and kept
                and kept[0].role == Role.ASSISTANT
                and kept[0].tool_calls
            ):
                candidate = [*system_messages, message, *kept]

            if estimate_prompt_tokens(candidate) <= max_prompt_tokens:
                kept.insert(0, message)
            else:
                break

        return [*system_messages, *kept]
    def _build_messages(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        *,
        system_prompt: Optional[str] = None,
        max_prompt_tokens: int = 3072,
    ) -> list[Message]:
        """Assemble the message list for a generate call.

        Optionally prepends a system prompt, then appends any context
        conversation messages, and finally the user input.
        """
        messages: list[Message] = []
        context_messages = (
            list(context.conversation.messages) if context is not None else []
        )
        # Check if the context already supplies a system message
        _context_has_system = (
            context
            and context.conversation.messages
            and any(m.role == Role.SYSTEM for m in context.conversation.messages)
        )

        if self._prompt_builder is not None:
            effective_system_prompt = self._prompt_builder.build()
        elif system_prompt:
            effective_system_prompt = system_prompt
        elif _context_has_system:
            effective_system_prompt = None
        else:
            # Fall back to the config-level default (grounds local models)
            try:
                cfg = load_config()
                effective_system_prompt = cfg.agent.default_system_prompt or None
            except Exception:
                effective_system_prompt = None
        # Fold ALL in-context system messages (both auto-captured memory
        # context and caller-supplied system messages) into one leading system
        # message. Do this even when there is no independently-built prompt:
        # Qwen-family chat templates reject a system message after the first
        # slot or more than one system message. Empty system messages must be
        # removed too, otherwise they can leave a second system entry behind.
        identity_already_applied = any(
            message.role == Role.SYSTEM
            and message.metadata.get("openjarvis_identity_prompt")
            for message in context_messages
        )
        system_parts = []
        if effective_system_prompt and not identity_already_applied:
            system_parts.append(effective_system_prompt)
        system_parts.extend(
            message.text
            for message in context_messages
            if message.role == Role.SYSTEM and message.text
        )
        context_messages = [
            message for message in context_messages if message.role != Role.SYSTEM
        ]
        if system_parts:
            messages.append(
                Message(role=Role.SYSTEM, content="\n\n".join(system_parts))
            )
        if context_messages:
            messages.extend(context_messages)
        messages.append(Message(role=Role.USER, content=input))
        return self._trim_conversation_to_budget(
            messages,
            max_prompt_tokens=max_prompt_tokens,
        )
    def _generate(self, messages: list[Message], **extra_kwargs: Any) -> dict:
        """Call ``engine.generate()`` with stored defaults.

        Extra kwargs (e.g. ``tools``) are forwarded to the engine.
        Publishes INFERENCE_START/END events on the bus when the engine
        does not publish its own (i.e. non-instrumented engines).
        """
        if self._bus and not getattr(self._engine, "_publishes_events", False):
            engine_id = getattr(self._engine, "engine_id", "")
            self._bus.publish(
                EventType.INFERENCE_START,
                {"model": self._model, "engine": engine_id},
            )

        # Stored engine options originate in CLI/runtime configuration and are
        # intentionally allowlisted. Per-call kwargs originate in the agent
        # implementation itself (for example ``tools`` or ``response_format``)
        # and must reach the engine adapter unchanged. Filtering the merged
        # mapping silently stripped function-calling tools from every agent.
        gen_kwargs = {
            key: value
            for key, value in self._engine_options.items()
            if key in _ALLOWED_ENGINE_OPTION_KEYS
        }
        gen_kwargs.update(extra_kwargs)
        result = self._engine.generate(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **gen_kwargs,
        )

        if self._bus and not getattr(self._engine, "_publishes_events", False):
            usage = result.get("usage", {})
            self._bus.publish(
                EventType.INFERENCE_END,
                {
                    "model": self._model,
                    "usage": usage,
                    "content": result.get("content", ""),
                    "tool_calls": result.get("tool_calls", []),
                    "finish_reason": result.get("finish_reason", ""),
                },
            )

        return result

    def _max_turns_result(
        self,
        tool_results: list[ToolResult],
        turns: int,
        content: str = "",
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Build the standard result for when ``max_turns`` is exceeded."""
        self._emit_turn_end(turns=turns, max_turns_exceeded=True)
        md: Dict[str, Any] = {"max_turns_exceeded": True}
        if metadata:
            md.update(metadata)
        return AgentResult(
            content=content or "Maximum turns reached without a final answer.",
            tool_results=tool_results,
            turns=turns,
            metadata=md,
        )

    def _check_continuation(
        self,
        result: dict,
        messages: list,
        *,
        max_continuations: int = 2,
    ) -> str:
        """Re-prompt on ``finish_reason == "length"`` to get complete output.

        Returns the concatenated content after up to *max_continuations*
        follow-up generate calls.
        """
        content = result.get("content", "")
        finish_reason = result.get("finish_reason", "")

        for _ in range(max_continuations):
            if finish_reason != "length":
                break
            # Append what we have so far and ask the model to continue
            from openjarvis.core.types import Message, Role

            messages.append(Message(role=Role.ASSISTANT, content=content))
            messages.append(
                Message(
                    role=Role.USER,
                    content="Continue from where you left off.",
                ),
            )
            cont = self._generate(messages)
            continuation = cont.get("content", "")
            content += continuation
            finish_reason = cont.get("finish_reason", "")

        return content

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove ``<think>...</think>`` blocks from model output.

        Handles both ``<think>...</think>`` and the common distilled-model
        pattern where the opening ``<think>`` is absent and the response
        begins directly with reasoning text followed by ``</think>``.
        """
        # Full <think>...</think> blocks
        text = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Leading content before a bare </think> (no opening tag)
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @abstractmethod
    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent on *input* and return an ``AgentResult``."""


class ToolUsingAgent(BaseAgent):
    """Intermediate base for agents that accept and use tools.

    Sets ``accepts_tools = True`` for CLI/SDK introspection, and
    initialises a :class:`ToolExecutor` from the provided tools.
    """

    accepts_tools: bool = True

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List["BaseTool"]] = None,  # noqa: F821
        bus: Optional[EventBus] = None,
        max_turns: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        loop_guard_config: Optional[Any] = None,
        capability_policy: Optional[Any] = None,
        agent_id: Optional[str] = None,
        interactive: bool = False,
        confirm_callback: Optional[Any] = None,
        skill_few_shot_examples: Optional[List[str]] = None,
        prompt_builder: Optional[Any] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_builder=prompt_builder,
        )
        from openjarvis.tools._stubs import ToolExecutor

        self._tools = tools or []
        # Plan 2B I3: store optimized few-shot examples for agents to inject
        # into their own system prompt templates as appropriate.
        self._skill_few_shot_examples = list(skill_few_shot_examples or [])
        _aid = agent_id or getattr(self, "agent_id", "")
        self._executor = ToolExecutor(
            self._tools,
            bus=bus,
            capability_policy=capability_policy,
            agent_id=_aid,
            interactive=interactive,
            confirm_callback=confirm_callback,
        )
        # Resolve max_turns: explicit arg > config > class default > 10
        if max_turns is not None:
            self._max_turns = max_turns
        else:
            try:
                cfg = load_config()
                self._max_turns = cfg.agent.max_turns
            except Exception:
                self._max_turns = getattr(self, "_default_max_turns", 10)

        # Loop guard
        self._loop_guard = None
        try:
            from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig

            if loop_guard_config is None:
                loop_guard_config = LoopGuardConfig()
            elif isinstance(loop_guard_config, dict):
                loop_guard_config = LoopGuardConfig(**loop_guard_config)
            if loop_guard_config.enabled:
                self._loop_guard = LoopGuard(loop_guard_config, bus=bus)
        except ImportError:
            pass


__all__ = ["AgentContext", "AgentResult", "BaseAgent", "ToolUsingAgent"]

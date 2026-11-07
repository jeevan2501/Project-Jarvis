"""LLM-backed extraction of durable facts from a conversation turn.

The extractor takes a single (user, assistant) exchange and asks a small
local model to distill any long-term, user-specific facts worth remembering.
It is deliberately defensive: extraction runs on a background thread far from
the request path, so *any* failure — a dropped Ollama connection, a timeout, a
``BrokenPipeError`` when the client went away, or simply unparseable output —
must degrade to "no facts" rather than propagate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from openjarvis.core.types import Message, Role

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You extract only durable, long-term facts explicitly stated by the USER "
    "from a single conversation exchange. A fact must be user-specific, "
    "explicitly supported by the user's message, stable over time, and useful "
    "in future conversations. Good examples include the user's preferences, "
    "identity, goals, ongoing projects, constraints, or relationships.\n\n"

    "IMPORTANT RULES:\n"
    "1. Extract facts ONLY from information explicitly stated by the USER.\n"
    "2. Do NOT extract facts from questions asked by the user.\n"
    "3. Do NOT extract information generated, suggested, or inferred by the assistant.\n"
    "4. Do NOT extract temporary task details, one-time calculations, "
    "instructions, explanations, or conversation context.\n"
    "5. Do NOT turn a question into a fact.\n"
    "6. Do NOT make assumptions or guesses.\n"
    "7. If you are uncertain whether something is a durable fact, do not store it.\n"
    "8. Prefer no fact over an incorrect or weak fact.\n\n"
    "9. If multiple statements express the same underlying fact, output only ONE "
    "canonical fact. Do not create alternate phrasings of the same fact.\n"
    "10. Prefer stable, direct wording that identifies the USER as the subject "
    "when appropriate.\n"
    "11. Do not repeat a fact merely because it was stated in a different way.\n"
    "12. When the USER says an existing fact has changed or gives a new value, "
    "output the updated fact in the same canonical form as the original fact. "
    "Do not preserve update wording such as 'has changed', 'is now', or "
    "'has been updated' in the fact itself.\n\n"

    "Examples:\n"
    'User: "I am building a local-first AI assistant on Windows."\n'
    'Output: ["User is building a local-first AI assistant on Windows."]\n\n'

    'User: "My JARVIS deployment code has changed. It is now SECOND-VALUE-456."\n'
    'Output: ["JARVIS deployment code is SECOND-VALUE-456."]\n\n'

    'User: "My JARVIS deployment code is now THIRD-VALUE-789."\n'
    'Output: ["JARVIS deployment code is THIRD-VALUE-789."]\n\n'

    "Examples:\n"
    'User: "My personal JARVIS project codename is Aegis."\n'
    'Output: ["Personal JARVIS project codename is Aegis."]\n\n'

    'User: "What is my JARVIS project codename?"\n'
    "Output: []\n\n"

    'User: "What kind of AI assistant am I building?"\n'
    "Output: []\n\n"

    'User: "I prefer concise answers."\n'
    'Output: ["User prefers concise answers."]\n\n'

    'User: "I am building a local-first AI assistant on Windows."\n'
    'Output: ["User is building a local-first AI assistant on Windows."]\n\n'

    "Respond with ONLY a JSON array of short fact strings. "
    "Each fact must be under 200 characters. "
    "If there is nothing worth remembering, respond with []."
)


class FactExtractor:
    """Extract memory-worthy facts from a conversation turn via an engine."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_facts_per_turn: int = 10,
        max_fact_chars: int = 200,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_facts_per_turn = max_facts_per_turn
        self._max_fact_chars = max_fact_chars
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def extract(self, user_text: str, assistant_text: str = "") -> List[str]:
        """Return durable facts from the exchange. Never raises."""
        user_text = (user_text or "").strip()
        if not user_text:
            return []

        exchange = f"User: {user_text}"

        messages = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=exchange),
        ]

        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except BrokenPipeError:
            # The classic failure mode: the model call's transport died.
            # Extraction is best-effort, so swallow it.
            logger.debug("Memory extraction aborted: broken pipe", exc_info=True)
            return []
        except Exception:  # noqa: BLE001 — extraction must never crash the worker
            logger.debug("Memory extraction failed", exc_info=True)
            return []

        if isinstance(result, dict):
            content = result.get("content", "") or ""
        else:
            content = str(result)

        facts = self._parse(content)
        return self._validate_facts(user_text, facts)

    # -- parsing ------------------------------------------------------------

    def _validate_facts(self, user_text: str, facts: List[str]) -> List[str]:
        """Keep only facts that are explicitly supported by the user's text."""
        if not facts:
            return []

        text = user_text.strip().lower()

        # Questions do not establish facts.
        question_patterns = (
            "?",
            "what is ",
            "what are ",
            "what do ",
            "what does ",
            "what did ",
            "what can ",
            "what should ",
            "what would ",
            "what kind of ",
            "who is ",
            "who are ",
            "where is ",
            "where are ",
            "when is ",
            "when did ",
            "why is ",
            "why are ",
            "why did ",
            "how is ",
            "how are ",
            "how do ",
            "how does ",
            "how can ",
            "how should ",
            "tell me ",
            "can you ",
            "could you ",
            "would you ",
            "do you ",
            "did you ",
            "is my ",
            "are my ",
        )

        if any(pattern in text for pattern in question_patterns):
            return []

        # Reject common uncertainty/speculation patterns.
        unsupported_patterns = (
            "probably know",
            "you know",
            "i think you know",
            "you likely know",
            "you should know",
            "i assume you know",
        )

        if any(pattern in text for pattern in unsupported_patterns):
            return []

        return facts

    def _parse(self, content: str) -> List[str]:
        """Parse model output into a clean, deduped, capped list of facts."""
        if not content or not content.strip():
            return []

        raw = self._coerce_to_list(content)

        facts: List[str] = []
        seen: set[str] = set()
        for item in raw:
            fact = self._clean_fact(item)
            if not fact:
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= self._max_facts_per_turn:
                break
        return facts

    def _coerce_to_list(self, content: str) -> List[str]:
        """Best-effort conversion of model output to a list of strings."""
        # 1. Try to locate and parse a JSON array anywhere in the output
        #    (models often wrap it in prose or code fences).
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Fall back to line-based parsing (markdown bullets / numbered).
        #    Deliberately permissive: small local models routinely answer with
        #    one bare fact per line, and rejecting those would quietly stop
        #    memory capture. Injected text can trivially add list markers, so
        #    filtering here buys no defence — provenance tagging and the
        #    injection scanner in MemoryService are the real gate.
        items: List[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            items.append(line)
        return items

    def _clean_fact(self, item: str) -> str:
        fact = str(item).strip().strip("\"'").strip()
        # Drop obvious non-facts the model sometimes emits.
        if not fact or fact.lower() in ("[]", "none", "n/a", "null"):
            return ""
        if len(fact) > self._max_fact_chars:
            fact = fact[: self._max_fact_chars].rstrip()
        return fact


__all__ = ["FactExtractor"]

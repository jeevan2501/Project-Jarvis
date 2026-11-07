"""Read-only local process inspection tool for JARVIS."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import psutil

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _safe_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def _process_details(process: psutil.Process) -> dict[str, Any]:
    """Collect safe, read-only information about one process."""
    try:
        with process.oneshot():
            memory_info = process.memory_info()

            try:
                username = process.username()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                username = None

            try:
                executable = process.exe()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                executable = None

            try:
                create_time = process.create_time()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                create_time = None

            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "username": username,
                "executable": executable,
                "memory_mb": round(
                    memory_info.rss / (1024 ** 2),
                    1,
                ),
                "memory_percent": _safe_percent(
                    process.memory_percent()
                ),
                "cpu_percent": _safe_percent(
                    process.cpu_percent(interval=0.05)
                ),
                "create_time": create_time,
            }

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
    ):
        return {}


def _collect_processes(
    *,
    name: str | None = None,
    pid: int | None = None,
    sort_by: str = "memory",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Collect processes using read-only psutil operations."""

    if pid is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return []

        details = _process_details(process)
        return [details] if details else []

    processes: list[dict[str, Any]] = []

    name_filter = name.strip().lower() if name else None

    for process in psutil.process_iter(
        ["pid", "name"],
    ):
        try:
            process_name = process.info.get("name") or ""

            if name_filter and name_filter not in process_name.lower():
                continue

            details = _process_details(process)

            if details:
                processes.append(details)

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    if sort_by == "cpu":
        processes.sort(
            key=lambda item: item.get("cpu_percent") or 0,
            reverse=True,
        )
    elif sort_by == "memory_percent":
        processes.sort(
            key=lambda item: item.get("memory_percent") or 0,
            reverse=True,
        )
    else:
        processes.sort(
            key=lambda item: item.get("memory_mb") or 0,
            reverse=True,
        )

    return processes[:limit]


@ToolRegistry.register("process_inspect")
class ProcessInspectTool(BaseTool):
    """Inspect running processes without modifying them."""

    tool_id = "process_inspect"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="process_inspect",
            description=(
                "Read-only inspection of local Windows/Linux/macOS "
                "processes. Can list the top processes by memory or CPU, "
                "search processes by name, or inspect a specific PID. "
                "Returns PID, name, status, username when available, "
                "executable path when permitted, CPU usage, memory usage, "
                "and process start time. This tool never starts, stops, "
                "terminates, suspends, or modifies processes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional process-name filter. "
                            "For example: python, chrome, code."
                        ),
                    },
                    "pid": {
                        "type": "integer",
                        "description": (
                            "Optional process ID to inspect."
                        ),
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": [
                            "memory",
                            "memory_percent",
                            "cpu",
                        ],
                        "description": (
                            "Sort order for process listings. "
                            "Default is memory."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": (
                            "Maximum number of processes to return. "
                            "Default is 10."
                        ),
                    },
                },
                "required": [],
            },
            category="system",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        try:
            name = params.get("name")
            pid = params.get("pid")
            sort_by = str(
                params.get("sort_by", "memory")
            ).strip().lower()

            limit = int(params.get("limit", 10))
            limit = max(1, min(limit, 25))

            if pid is not None:
                pid = int(pid)

            if sort_by not in {
                "memory",
                "memory_percent",
                "cpu",
            }:
                sort_by = "memory"

            processes = _collect_processes(
                name=name,
                pid=pid,
                sort_by=sort_by,
                limit=limit,
            )

            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(
                    {
                        "query": {
                            "name": name,
                            "pid": pid,
                            "sort_by": sort_by,
                            "limit": limit,
                        },
                        "processes": processes,
                        "timestamp": time.time(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                success=True,
                metadata={
                    "platform": os.name,
                    "read_only": True,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Process inspection failed: {exc}",
                success=False,
            )


__all__ = ["ProcessInspectTool"]
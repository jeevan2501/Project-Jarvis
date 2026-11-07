"""Read-only Windows service inspection tool for JARVIS."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _run_sc(*args: str) -> str:
    """Run a read-only sc.exe command and return stdout."""
    completed = subprocess.run(
        ["sc.exe", *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            error or f"sc.exe exited with code {completed.returncode}"
        )

    return completed.stdout


def _parse_query(output: str) -> list[dict[str, Any]]:
    """Parse `sc query` output into structured service records."""
    services: list[dict[str, Any]] = []

    current: dict[str, Any] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("SERVICE_NAME:"):
            if current:
                services.append(current)

            current = {
                "service_name": line.split(":", 1)[1].strip(),
            }
            continue

        if line.startswith("DISPLAY_NAME:"):
            current["display_name"] = line.split(":", 1)[1].strip()
            continue

        match = re.match(
            r"STATE\s*:\s*(\d+)\s+(\w+)",
            line,
        )
        if match:
            current["state_code"] = int(match.group(1))
            current["state"] = match.group(2)
            continue

        match = re.match(
            r"PID\s*:\s*(\d+)",
            line,
        )
        if match:
            current["pid"] = int(match.group(1))
            continue

    if current:
        services.append(current)

    return services


def _parse_config(output: str) -> dict[str, Any]:
    """Parse `sc qc` output into startup configuration."""
    result: dict[str, Any] = {}

    patterns = {
        "service_name": r"SERVICE_NAME\s*:\s*(.+)",
        "display_name": r"DISPLAY_NAME\s*:\s*(.+)",
        "start_type": r"START_TYPE\s*:\s*\d+\s+(\w+)",
        "error_control": r"ERROR_CONTROL\s*:\s*\d+\s+(\w+)",
        "binary_path": r"BINARY_PATH_NAME\s*:\s*(.+)",
        "load_order_group": r"LOAD_ORDER_GROUP\s*:\s*(.*)",
        "dependencies": r"DEPENDENCIES\s*:\s*(.*)",
        "account_name": r"SERVICE_START_NAME\s*:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1).strip()
            if value:
                result[key] = value

    return result


def _inspect_service(service_name: str) -> dict[str, Any]:
    """Inspect one Windows service."""
    query_output = _run_sc("query", service_name)
    config_output = _run_sc("qc", service_name)

    query = _parse_query(query_output)
    config = _parse_config(config_output)

    if not query:
        raise RuntimeError(
            f"Service '{service_name}' was not found."
        )

    result = dict(query[0])
    result.update(config)

    return result


def _list_services(
    *,
    state: str | None = None,
    name: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """List Windows services without modifying them."""

    output = _run_sc("query", "type=", "service", "state=", "all")

    services = _parse_query(output)

    state_filter = state.strip().upper() if state else None
    name_filter = name.strip().lower() if name else None

    if state_filter:
        services = [
            service
            for service in services
            if service.get("state") == state_filter
        ]

    if name_filter:
        services = [
            service
            for service in services
            if name_filter in (
                str(service.get("service_name", "")).lower()
                + " "
                + str(service.get("display_name", "")).lower()
            )
        ]

    return services[:limit]


@ToolRegistry.register("service_inspect")
class ServiceInspectTool(BaseTool):
    """Inspect Windows services without modifying them."""

    tool_id = "service_inspect"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="service_inspect",
            description=(
                "Read-only inspection of Windows services. "
                "Can inspect a specific service or list services "
                "filtered by state or name. Returns service name, "
                "display name, running state, PID when available, "
                "startup type, executable path, dependencies, and "
                "service account when available. This tool never "
                "starts, stops, restarts, disables, or modifies services."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": (
                            "Optional exact Windows service name to inspect. "
                            "Example: wuauserv."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional text filter for service name or "
                            "display name."
                        ),
                    },
                    "state": {
                        "type": "string",
                        "enum": [
                            "RUNNING",
                            "STOPPED",
                            "PAUSED",
                        ],
                        "description": (
                            "Optional service-state filter when listing."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": (
                            "Maximum number of services to return. "
                            "Default is 25."
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
            service_name = params.get("service_name")
            name = params.get("name")
            state = params.get("state")

            limit = int(params.get("limit", 25))
            limit = max(1, min(limit, 50))

            if service_name:
                service = _inspect_service(
                    str(service_name).strip()
                )

                content = {
                    "mode": "inspect",
                    "service": service,
                    "read_only": True,
                }
            else:
                services = _list_services(
                    state=str(state) if state else None,
                    name=str(name) if name else None,
                    limit=limit,
                )

                content = {
                    "mode": "list",
                    "filters": {
                        "name": name,
                        "state": state,
                        "limit": limit,
                    },
                    "services": services,
                    "read_only": True,
                }

            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(
                    content,
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
                content=f"Service inspection failed: {exc}",
                success=False,
            )


__all__ = ["ServiceInspectTool"]
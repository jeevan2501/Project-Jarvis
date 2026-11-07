"""Local computer diagnostics tool for JARVIS."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from typing import Any
import time
import psutil

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)


def _safe_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def _get_gpu_info() -> list[dict[str, Any]]:
    """Get NVIDIA GPU information when nvidia-smi is available."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []

    query = (
        "name,"
        "driver_version,"
        "utilization.gpu,"
        "utilization.memory,"
        "memory.total,"
        "memory.used,"
        "memory.free,"
        "temperature.gpu"
    )

    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if completed.returncode != 0:
            return []

        gpus: list[dict[str, Any]] = []

        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]

            if len(fields) < 8:
                continue

            def number(value: str) -> float | None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            gpus.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "gpu_utilization_percent": number(fields[2]),
                    "memory_utilization_percent": number(fields[3]),
                    "vram_total_gb": number(fields[4]) / 1024
                    if number(fields[4]) is not None
                    else None,
                    "vram_used_gb": number(fields[5]) / 1024
                    if number(fields[5]) is not None
                    else None,
                    "vram_free_gb": number(fields[6]) / 1024
                    if number(fields[6]) is not None
                    else None,
                    "temperature_c": number(fields[7]),
                }
            )

        return gpus

    except (OSError, subprocess.SubprocessError):
        return []


def _get_disk_info() -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []

    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (OSError, PermissionError):
            continue

        disks.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_gb": _bytes_to_gb(usage.total),
                "used_gb": _bytes_to_gb(usage.used),
                "free_gb": _bytes_to_gb(usage.free),
                "usage_percent": _safe_percent(usage.percent),
            }
        )

    return disks


def _get_process_info() -> list[dict[str, Any]]:
    """Return the top 5 processes by memory usage with CPU usage."""
    processes: list[dict[str, Any]] = []

    process_objects = []

    for process in psutil.process_iter(
        ["pid", "name", "memory_info"],
    ):
        try:
            process.cpu_percent(None)
            process_objects.append(process)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    # Allow psutil to calculate CPU usage over a short interval.
    time.sleep(0.2)

    for process in process_objects:
        try:
            info = process.info
            memory_info = info.get("memory_info")

            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "memory_mb": round(
                        memory_info.rss / (1024 ** 2),
                        1,
                    )
                    if memory_info
                    else None,
                    "cpu_percent": _safe_percent(
                        process.cpu_percent(None)
                    ),
                }
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    # Sort primarily by memory because memory pressure is one
    # of the main causes we want this diagnostic to identify.
    processes.sort(
        key=lambda item: item.get("memory_mb") or 0,
        reverse=True,
    )

    return processes[:5]


def _collect_diagnostics() -> dict[str, Any]:
    """Collect the most relevant computer health information."""
    virtual_memory = psutil.virtual_memory()

    cpu_percent = psutil.cpu_percent(interval=0.5)

    # Generate compact alerts for obvious resource pressure.
    alerts: list[str] = []

    if virtual_memory.percent >= 90:
        alerts.append(
            f"Critical memory pressure: RAM usage is "
            f"{virtual_memory.percent:.1f}%."
        )
    elif virtual_memory.percent >= 85:
        alerts.append(
            f"High memory usage: RAM usage is "
            f"{virtual_memory.percent:.1f}%."
        )

    if cpu_percent >= 90:
        alerts.append(
            f"High CPU usage: CPU usage is {cpu_percent:.1f}%."
        )

    try:
        cpu_frequency = psutil.cpu_freq()

        frequency = (
            {
                "current_mhz": round(cpu_frequency.current, 0),
                "max_mhz": round(cpu_frequency.max, 0),
            }
            if cpu_frequency
            else None
        )

    except (AttributeError, OSError):
        frequency = None

    return {
        "cpu": {
            "model": platform.processor(),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "usage_percent": _safe_percent(cpu_percent),
            "frequency": frequency,
        },
        "memory": {
            "total_gb": _bytes_to_gb(virtual_memory.total),
            "used_gb": _bytes_to_gb(virtual_memory.used),
            "available_gb": _bytes_to_gb(virtual_memory.available),
            "usage_percent": _safe_percent(virtual_memory.percent),
        },
        "gpu": _get_gpu_info(),
        "disk": _get_disk_info(),
        "top_processes": _get_process_info(),
        "alerts": alerts,
    }


@ToolRegistry.register("computer_diagnostics")
class ComputerDiagnosticsTool(BaseTool):
    """Inspect the local computer's current health and hardware."""

    tool_id = "computer_diagnostics"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="computer_diagnostics",
            description=(
                "Inspect the local computer's current health. Returns "
                "CPU usage and cores, RAM usage, NVIDIA GPU utilization "
                "and VRAM when available, disk usage, and the top 5 "
                "processes by memory usage. Use this tool whenever the "
                "user asks JARVIS to check, inspect, diagnose, monitor, "
                "or analyze the computer/system/PC."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            category="system",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        del params

        try:
            diagnostics = _collect_diagnostics()

            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(
                    diagnostics,
                    indent=2,
                    ensure_ascii=False,
                ),
                success=True,
                metadata={
                    "platform": os.name,
                    "diagnostic": True,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Computer diagnostics failed: {exc}",
                success=False,
            )


__all__ = ["ComputerDiagnosticsTool"]

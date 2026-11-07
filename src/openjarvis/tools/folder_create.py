"""Folder creation tool — create directories with verification."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools._stubs import BaseTool, ToolResult, ToolSpec


@ToolRegistry.register("folder_create")
class FolderCreateTool(BaseTool):
    """Create a directory and verify that it exists."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="folder_create",
            description=(
                "Create a folder at the requested path. Creates missing "
                "parent directories and verifies that the folder exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path of the folder to create.",
                    },
                },
                "required": ["path"],
            },
            category="filesystem",
            required_capabilities=["file:write"],
        )

    def execute(self, path: str, **kwargs) -> ToolResult:
        """Create and verify a folder."""
        try:
            folder_path = Path(path).expanduser().resolve()

            # Refuse to replace an existing file with a directory.
            if folder_path.exists() and not folder_path.is_dir():
                return ToolResult(
                    tool_name="folder_create",
                    content=(
                        f"Cannot create folder because a file already exists "
                        f"at {folder_path}"
                    ),
                    success=False,
                )

            folder_path.mkdir(parents=True, exist_ok=True)

            # Verify the directory was actually created.
            if not folder_path.exists() or not folder_path.is_dir():
                return ToolResult(
                    tool_name="folder_create",
                    content=(
                        f"Folder creation failed: "
                        f"{folder_path}"
                    ),
                    success=False,
                )

            return ToolResult(
                tool_name="folder_create",
                content=f"Successfully created folder: {folder_path}",
                success=True,
                metadata={
                    "path": str(folder_path),
                    "created": True,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name="folder_create",
                content=f"Failed to create folder: {exc}",
                success=False,
            )
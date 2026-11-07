"""DOCX creation tool — create real Microsoft Word documents."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools._stubs import BaseTool, ToolResult, ToolSpec


@ToolRegistry.register("docx_create")
class DocxCreateTool(BaseTool):
    """Create a real Microsoft Word .docx document."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="docx_create",
            description=(
                "Create a real Microsoft Word DOCX document from a title "
                "and body text. The file is saved to the requested path "
                "and verified after creation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path where the .docx file should be created."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Main body text of the document.",
                    },
                },
                "required": ["path", "content"],
            },
            category="filesystem",
            required_capabilities=["file:write"],
        )

    def execute(
        self,
        path: str,
        content: str,
        title: str = "",
        **kwargs,
    ) -> ToolResult:
        """Create and verify a DOCX document."""
        try:
            from docx import Document

            file_path = Path(path).expanduser().resolve()

            if file_path.suffix.lower() != ".docx":
                file_path = file_path.with_suffix(".docx")

            file_path.parent.mkdir(parents=True, exist_ok=True)

            document = Document()

            if title.strip():
                document.add_heading(title.strip(), level=1)

            for paragraph in content.split("\n"):
                document.add_paragraph(paragraph)

            document.save(file_path)

            # Verify that a real file was created.
            if not file_path.exists():
                return ToolResult(
                    tool_name="docx_create",
                    content=f"DOCX creation failed: file was not created at {file_path}",
                    success=False,
                )

            size = file_path.stat().st_size

            if size == 0:
                return ToolResult(
                    tool_name="docx_create",
                    content=f"DOCX creation failed: empty file at {file_path}",
                    success=False,
                )

            return ToolResult(
                tool_name="docx_create",
                content=f"Successfully created Word document: {file_path}",
                success=True,
                metadata={
                    "path": str(file_path),
                    "size_bytes": size,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name="docx_create",
                content=f"Failed to create Word document: {exc}",
                success=False,
            )

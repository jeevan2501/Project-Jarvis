"""Excel creation tool — create real .xlsx workbooks."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools._stubs import BaseTool, ToolResult, ToolSpec


@ToolRegistry.register("excel_create")
class ExcelCreateTool(BaseTool):
    """Create a real Microsoft Excel .xlsx workbook."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="excel_create",
            description=(
                "Create a real Microsoft Excel XLSX file on the user's computer. "
                "Use this tool whenever the user asks you to create an Excel "
                "spreadsheet or XLSX file. The path must be the full file path. "
                "The data parameter contains rows of spreadsheet data, using "
                "new lines between rows and | between columns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path of the Excel XLSX file to create."
                        ),
                    },
                    "data": {
                        "type": "string",
                        "description": (
                            "Spreadsheet contents. Use | between columns "
                            "and new lines between rows."
                        ),
                    },
                },
                "required": ["path"],
            },
            category="filesystem",
            required_capabilities=["file:write"],
        )

    def execute(
        self,
        path: str,
        data: str = "",
        **kwargs,
    ) -> ToolResult:
        """Create and verify an XLSX workbook."""
        try:
            from openpyxl import Workbook, load_workbook

            file_path = Path(path).expanduser().resolve()

            if file_path.suffix.lower() != ".xlsx":
                file_path = file_path.with_suffix(".xlsx")

            file_path.parent.mkdir(parents=True, exist_ok=True)

            workbook = Workbook()
            worksheet = workbook.active
            worksheet.title = "Sheet1"

            lines = [
                line.strip()
                for line in str(data).splitlines()
                if line.strip()
            ]

            for line in lines:
                cells = [cell.strip() for cell in line.split("|")]
                worksheet.append(cells)

            workbook.save(file_path)

            if not file_path.exists():
                return ToolResult(
                    tool_name="excel_create",
                    content=f"Excel file was not created: {file_path}",
                    success=False,
                )

            size = file_path.stat().st_size

            if size == 0:
                return ToolResult(
                    tool_name="excel_create",
                    content=f"Excel file is empty: {file_path}",
                    success=False,
                )

            verification_workbook = load_workbook(
                file_path,
                read_only=True,
            )

            sheets = verification_workbook.sheetnames
            verification_workbook.close()

            return ToolResult(
                tool_name="excel_create",
                content=f"Successfully created Excel workbook: {file_path}",
                success=True,
                metadata={
                    "path": str(file_path),
                    "size_bytes": size,
                    "sheets": sheets,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name="excel_create",
                content=f"Failed to create Excel workbook: {exc}",
                success=False,
            )
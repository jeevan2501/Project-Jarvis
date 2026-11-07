from openjarvis.tools.code_interpreter import CodeInterpreterTool

tool = CodeInterpreterTool()

result = tool.execute(
    operation="run_python",
    path=r"C:\Users\JEEVAN\Desktop\calculator\calculator.py",
    arguments=[],
    stdin="1\n10\n20\n",
)

print("SUCCESS:", result.success)
print("CONTENT:")
print(result.content)
print("METADATA:")
print(result.metadata)

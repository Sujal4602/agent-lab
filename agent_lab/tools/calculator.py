import ast
import operator as op
from langchain_core.tools import tool

ops = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv
}

def eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return ops[type(node.op)](
            eval_expr(node.left),
            eval_expr(node.right)
        )
    raise TypeError("Unsupported expression")

@tool
def calculator_tool(expression: str) -> str:
    """Use this to solve math expressions like '2+2' or '20*5'."""
    try:
        return str(eval_expr(ast.parse(expression, mode='eval').body))
    except Exception:
        return "Invalid math expression"
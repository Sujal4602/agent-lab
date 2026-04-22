import ast
import operator as op
import re
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

def extract_expression(text: str) -> str:
    """Extract math expression from natural language."""
    text = text.lower()
    text = text.replace("percent of", "* 0.01 *")
    text = text.replace("difference between", "")
    text = text.replace("sum of", "+")
    text = text.replace("product of", "*")
    text = text.replace("divided by", "/")
    text = text.replace("multiplied by", "*")
    text = text.replace("minus", "-")
    text = text.replace("plus", "+")
    text = text.replace("and", "-")  # "difference between 10 and 5" → "10 - 5"
    
    # extract first valid math expression
    match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
    if match:
        return match.group().strip()
    return text

@tool
def calculator_tool(expression: str) -> str:
    """Use this to solve math problems. Input can be natural language like 'difference between 10 and 5' or a direct expression like '10-5'."""
    try:
        clean = extract_expression(expression)
        return str(eval_expr(ast.parse(clean, mode='eval').body))
    except Exception:
        return "Invalid math expression"
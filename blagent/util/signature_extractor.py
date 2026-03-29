import ast
from typing import Union


class SignatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.tree = []

    def visit_Module(self, node):
        for item in node.body:
            self.tree.append(self._process_node(item))

    def _process_node(self, node) -> Union[str, tuple]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._format_function(node)

        elif isinstance(node, ast.ClassDef):
            # Include base classes for better context
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Handle attribute-based bases like module.Class
                    if hasattr(base.value, "id"):
                        value_id = getattr(base.value, "id")
                        base_name = f"{value_id}.{base.attr}"
                    else:
                        base_name = base.attr
                    bases.append(base_name)

            bases_str = f"({', '.join(bases)})" if bases else ""
            class_name = f"class {node.name}{bases_str}"
            children = []
            for item in node.body:
                processed = self._process_node(item)
                if processed:  # Only add non-None items
                    children.append(processed)
            return (class_name, children)

    def _format_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> str:
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        arg_list = ", ".join(args)

        # Include async prefix if it's an async function
        is_async = isinstance(node, ast.AsyncFunctionDef)
        prefix = "async def" if is_async else "def"
        return f"{prefix} {node.name}({arg_list})"


def str_signature_tree(tree, indent=0):
    signature_tree = ""
    is_first = True
    for node in tree:
        if isinstance(node, str):
            signature_tree += "  " * indent + node + "\n"
        elif isinstance(node, tuple):
            # Add separator before each class (except the first)
            if not is_first and indent == 0:
                signature_tree += "#" * 10 + "\n"
            is_first = False

            class_name, children = node
            signature_tree += "  " * indent + class_name + ":\n"
            signature_tree += str_signature_tree(children, indent + 1)

    return signature_tree


def extract_signature_tree(source: str) -> str:
    tree = ast.parse(source)
    builder = SignatureExtractor()
    builder.visit(tree)
    return str_signature_tree(builder.tree)

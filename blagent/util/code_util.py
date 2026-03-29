import ast
import re


def get_code_text_from_path(node, path_parts):
    if len(path_parts) == 1:
        return "\n".join(node[path_parts[0]]["text"])

    return get_code_text_from_path(node[path_parts[0]], path_parts[1:])


def extract_patch_file_path(patch_str):
    """Extract the file path that was patched from the diff."""
    match = re.search(r"^diff --git a/(.*?) b/\1", patch_str, re.MULTILINE)
    if match:
        return match.group(1)
    fallback = re.search(r"^\+\+\+ b/(.+)", patch_str, re.MULTILINE)
    if fallback:
        return fallback.group(1)
    return None


def view_method_body(source: str, method_name: str) -> str:
    """Return the source code for a function or method.

    Supports finding top-level functions by name, or class methods using
    the notation "ClassName.method_name". Returns the full function
    source (the def line and its body) if found, otherwise an empty
    string.
    """
    tree = ast.parse(source)

    # Helper to get source segment safely
    def _segment(node):
        try:
            return ast.get_source_segment(source, node) or ""
        except (AttributeError, TypeError):
            return ""

    # If caller provided Class.method, split
    if "." in method_name:
        class_name, func_name = method_name.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == func_name:
                            return _segment(item)
        return ""

    # Otherwise search for a top-level function or method inside classes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return _segment(node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == method_name:
                        return _segment(item)

    return ""

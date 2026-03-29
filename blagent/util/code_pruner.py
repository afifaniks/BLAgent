"""
Production-Grade Code Pruner for Python

This module prunes Python source files while preserving:
- File structure (imports, class definitions, method signatures)
- Retrieved code chunks (in full detail)
- Collapses non-retrieved methods/functions to ellipsis (...)

Designed for LLM-based code understanding and cross-file bug localization,
where multiple candidate files with retrieved chunks must fit within
a fixed context window.

Usage:
    from code_pruner import CodePruner

    with open('large_file.py', 'r') as f:
        source = f.read()

    chunks = [
        'def my_function(...):\\n    # code fragment',
        'class MyClass:\\n    # code fragment'
    ]

    pruner = CodePruner(source, chunks)
    pruned = pruner.prune()
"""

import ast
import re
from typing import List, Set, Tuple


class CodePruner:
    """
    Prunes Python source code while preserving file structure and retrieved chunks.

    Strategy:
    1. Parse the full file into an AST
    2. Locate each retrieved chunk using substring matching
    3. Mark all lines belonging to retrieved chunks
    4. Collapse non-retrieved methods/functions to "..."
    5. Preserve imports, class signatures, and file structure

    Attributes:
        full_source (str): The complete Python source code
        lines (List[str]): Source code split into lines
        chunks (List[str]): Code fragments to preserve
        tree (ast.Module): Parsed AST of the source
        chunk_lines (Set[int]): Line numbers that are part of chunks (1-indexed)
    """

    def __init__(self, full_source: str, chunks: List[str]):
        """
        Initialize the pruner.

        Args:
            full_source: Complete Python file content
            chunks: List of code chunk strings to preserve (exact text matches)

        Raises:
            ValueError: If the source code has invalid Python syntax
        """
        self.full_source = full_source
        self.lines = full_source.split("\n")
        self.chunks = [self._clean_chunk(c) for c in chunks]

        # Parse AST
        try:
            self.tree = ast.parse(full_source)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        # Track which line numbers are part of retrieved chunks
        self.chunk_lines: Set[int] = set()
        self._locate_chunks()

    def _clean_chunk(self, chunk: str) -> str:
        """
        Remove metadata markers like [PATH], [CODE] from chunks.

        Args:
            chunk: Raw chunk text potentially containing markers

        Returns:
            Cleaned chunk text
        """
        if "[CODE]" in chunk:
            return chunk.split("[CODE]", 1)[1].strip()
        return chunk.strip()

    def _find_substring_lines(self, substring: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of a substring in the source.

        Uses string matching on the full source to locate chunks,
        then converts character positions to line numbers.

        Args:
            substring: The text to search for

        Returns:
            List of (start_line, end_line) tuples in 1-indexed format.
            Returns empty list if substring not found.
        """
        matches = []
        source_text = self.full_source

        # Find all positions of substring
        pos = 0
        while True:
            pos = source_text.find(substring, pos)
            if pos == -1:
                break

            # Convert character position to line numbers
            line_num = source_text[:pos].count("\n") + 1
            end_pos = pos + len(substring)
            end_line = source_text[:end_pos].count("\n") + 1

            matches.append((line_num, end_line))
            pos += 1

        return matches

    def _locate_chunks(self) -> None:
        """
        Locate each chunk in the source file.

        Finds each chunk by exact substring matching and marks the
        corresponding line ranges as part of chunks.
        """
        for chunk in self.chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue

            # Try to find this chunk as a substring
            matches = self._find_substring_lines(chunk_stripped)

            if matches:
                # Use the first match
                start_line, end_line = matches[0]
                self.chunk_lines.update(range(start_line, end_line + 1))

    def _node_overlaps_chunk(self, node: ast.AST) -> bool:
        """
        Check if an AST node overlaps with any retrieved chunks.

        Args:
            node: An AST node with lineno and end_lineno attributes

        Returns:
            True if the node overlaps with any chunk lines
        """
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return False

        start, end = node.lineno, node.end_lineno
        node_lines = set(range(start, end + 1))
        return bool(node_lines & self.chunk_lines)

    def _extract_lines(self, start: int, end: int) -> str:
        """
        Extract a range of lines from the source.

        Args:
            start: Start line number (1-indexed)
            end: End line number (1-indexed)

        Returns:
            The extracted source code as a string
        """
        if start < 1:
            start = 1
        if end > len(self.lines):
            end = len(self.lines)
        return "\n".join(self.lines[start - 1 : end])

    def _collapse_function(self, node: ast.FunctionDef, indent: str) -> str:
        """
        Collapse a function definition to its signature.

        Preserves:
        - Decorators (@classmethod, @staticmethod, @property, etc.)
        - Function name
        - Parameters
        - Return type annotation

        Args:
            node: The FunctionDef AST node
            indent: Indentation string to use

        Returns:
            A string representation of the collapsed function
        """
        # Get decorators
        decorators = ""
        if node.decorator_list:
            decorators = (
                "\n".join(f"{indent}{ast.unparse(dec)}" for dec in node.decorator_list)
                + "\n"
            )

        # Build signature
        args_str = ast.unparse(node.args)
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        return f"{decorators}{indent}def {node.name}({args_str}){returns}:\n{indent}    ..."

    def _prune_class(self, node: ast.ClassDef, indent: str = "") -> str:
        """
        Prune a class definition.

        Preserves:
        - Class signature and base classes
        - Methods/functions that contain chunk lines
        - Docstrings and class variables

        Collapses:
        - Methods without chunk lines

        Args:
            node: The ClassDef AST node
            indent: Indentation string to use

        Returns:
            A string representation of the pruned class
        """
        output = []

        # Class signature
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        if bases:
            bases = f"({bases})"

        output.append(f"{indent}class {node.name}{bases}:")

        # Process body
        method_indent = indent + "    "
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if self._node_overlaps_chunk(item):
                    # Keep full method
                    start, end = item.lineno, item.end_lineno
                    output.append(self._extract_lines(start, end))
                else:
                    # Collapse
                    output.append(self._collapse_function(item, method_indent))

            elif isinstance(item, ast.ClassDef):
                # Nested class
                if self._node_overlaps_chunk(item):
                    output.append(self._prune_class(item, method_indent))
                else:
                    output.append(
                        f"{method_indent}class {item.name}:\n{method_indent}    ..."
                    )

            elif isinstance(item, (ast.Expr, ast.Assign, ast.AnnAssign)):
                # Keep docstrings and class variables
                start, end = item.lineno, item.end_lineno
                output.append(self._extract_lines(start, end))

        return "\n".join(output)

    def _prune_function(self, node: ast.FunctionDef, indent: str = "") -> str:
        """
        Prune a top-level function.

        Args:
            node: The FunctionDef AST node
            indent: Indentation string to use

        Returns:
            Either the full function (if it contains chunks) or collapsed signature
        """
        if self._node_overlaps_chunk(node):
            start, end = node.lineno, node.end_lineno
            return self._extract_lines(start, end)
        else:
            return self._collapse_function(node, indent)

    def prune(self) -> str:
        """
        Generate the pruned code representation.

        Processes all top-level nodes in the file:
        - Imports: Always kept
        - Classes: Pruned (structure preserved, non-chunk methods collapsed)
        - Functions: Pruned (collapsed if no chunks)
        - Module-level variables: Kept
        - Module docstring: Kept
        - if __name__ == '__main__' blocks: Kept

        Returns:
            Pruned source code as a string
        """
        output = []

        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Always keep imports as-is
                start, end = node.lineno, node.end_lineno
                output.append(self._extract_lines(start, end))

            elif isinstance(node, ast.ClassDef):
                # Prune class (keep structure, selective content)
                output.append(self._prune_class(node))

            elif isinstance(node, ast.FunctionDef):
                # Prune top-level function
                output.append(self._prune_function(node))

            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
                # Keep module-level variables and docstrings
                start, end = node.lineno, node.end_lineno
                output.append(self._extract_lines(start, end))

            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                # Keep other top-level statements (e.g., if __name__ == '__main__')
                start, end = node.lineno, node.end_lineno
                output.append(self._extract_lines(start, end))

        # Clean up result
        result = "\n".join(output)

        # Remove excessive consecutive blank lines (more than 2)
        result = re.sub(r"\n\n\n+", "\n\n", result)

        # Ensure trailing newline
        if result and not result.endswith("\n"):
            result += "\n"

        return result

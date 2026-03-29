import ast
import re
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChunkInfo:
    """Information about a code chunk to preserve"""
    code: str
    start_line: int = -1
    end_line: int = -1
    node: Optional[ast.AST] = None


class CodePruner:
    """
    Prunes Python source code while preserving file structure and retrieved chunks.
    
    Strategy:
    1. Parse the full file into an AST
    2. Locate each retrieved chunk using exact text matching
    3. Mark lines belonging to retrieved chunks
    4. Collapse non-retrieved methods/functions to "..."
    5. Preserve imports, class signatures, and structure
    """
    
    def __init__(self, full_source: str, chunks: List[str]):
        """
        Args:
            full_source: Complete Python file content
            chunks: List of code chunk strings to preserve (from [CODE] markers)
        """
        self.full_source = full_source
        self.lines = full_source.split('\n')
        self.chunks = chunks
        
        # Parse AST
        try:
            self.tree = ast.parse(full_source)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        
        # Track which line numbers are part of retrieved chunks
        self.chunk_lines: Set[int] = set()
        self._locate_chunks()
    
    def _locate_chunks(self) -> None:
        """Find line ranges for each chunk in the source file"""
        for chunk in self.chunks:
            chunk_stripped = chunk.strip()
            
            # Find the chunk in the source
            for i, line in enumerate(self.lines, 1):
                # Check if this line starts a chunk
                if line.strip().startswith(chunk_stripped.split('\n')[0].strip()):
                    # Try to match the entire chunk
                    chunk_lines = chunk_stripped.split('\n')
                    matched = True
                    chunk_start = i
                    
                    for j, chunk_line in enumerate(chunk_lines):
                        source_line_idx = i - 1 + j
                        if source_line_idx >= len(self.lines):
                            matched = False
                            break
                        
                        # Normalize whitespace for comparison
                        src_normalized = self.lines[source_line_idx].rstrip()
                        chunk_normalized = chunk_line.rstrip()
                        
                        if src_normalized != chunk_normalized:
                            matched = False
                            break
                    
                    if matched:
                        chunk_end = i - 1 + len(chunk_lines)
                        self.chunk_lines.update(range(chunk_start, chunk_end + 1))
                        break
    
    def _is_line_in_chunk(self, line_num: int) -> bool:
        """Check if a line number is part of a retrieved chunk"""
        return line_num in self.chunk_lines
    
    def _get_node_lines(self, node: ast.AST) -> Tuple[int, int]:
        """Get start and end line numbers for an AST node"""
        start = getattr(node, 'lineno', 1)
        end = getattr(node, 'end_lineno', 1)
        return start, end
    
    def _node_overlaps_chunk(self, node: ast.AST) -> bool:
        """Check if an AST node overlaps with retrieved chunks"""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return False
        
        start, end = self._get_node_lines(node)
        node_lines = set(range(start, end + 1))
        return bool(node_lines & self.chunk_lines)
    
    def _get_indentation(self, line_num: int) -> str:
        """Get the indentation of a specific line"""
        if line_num < 1 or line_num > len(self.lines):
            return ""
        line = self.lines[line_num - 1]
        return line[:len(line) - len(line.lstrip())]
    
    def _extract_lines(self, start: int, end: int) -> str:
        """Extract lines from the source"""
        if start < 1:
            start = 1
        if end > len(self.lines):
            end = len(self.lines)
        return '\n'.join(self.lines[start - 1:end])
    
    def _collapse_function(self, node: ast.FunctionDef, indent: str) -> str:
        """Collapse a function to ellipsis"""
        # Reconstruct function signature
        args_str = ast.unparse(node.args)
        
        # Add return annotation if present
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        signature = f"{indent}def {node.name}({args_str}){returns}:\n{indent}    ..."
        return signature
    
    def _collapse_method(self, node: ast.FunctionDef, indent: str) -> str:
        """Collapse a method to ellipsis"""
        return self._collapse_function(node, indent)
    
    def _prune_class(self, node: ast.ClassDef, indent: str = "") -> str:
        """Prune a class definition"""
        output_lines = []
        
        # Class definition line
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        if bases:
            bases = f"({bases})"
        else:
            bases = ""
        
        output_lines.append(f"{indent}class {node.name}{bases}:")
        
        # Add class body items
        method_indent = indent + "    "
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if self._node_overlaps_chunk(item):
                    # Keep full method - extract from source
                    start, end = self._get_node_lines(item)
                    method_code = self._extract_lines(start, end)
                    output_lines.append(method_code)
                else:
                    # Collapse method
                    output_lines.append(self._collapse_method(item, method_indent))
            
            elif isinstance(item, ast.ClassDef):
                # Nested class
                if self._node_overlaps_chunk(item):
                    nested = self._prune_class(item, method_indent)
                    output_lines.append(nested)
                else:
                    # Collapse nested class
                    output_lines.append(f"{method_indent}class {item.name}:\n{method_indent}    ...")
            
            elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                # Docstring - keep it
                output_lines.append(self._extract_lines(*self._get_node_lines(item)))
            
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                # Class variables - keep them
                output_lines.append(self._extract_lines(*self._get_node_lines(item)))
        
        return '\n'.join(output_lines)
    
    def _prune_function(self, node: ast.FunctionDef, indent: str = "") -> str:
        """Prune a top-level function"""
        if self._node_overlaps_chunk(node):
            # Keep full function
            start, end = self._get_node_lines(node)
            return self._extract_lines(start, end)
        else:
            # Collapse function
            return self._collapse_function(node, indent)
    
    def _should_keep_node(self, node: ast.AST) -> bool:
        """Determine if a top-level node should be kept"""
        # Always keep imports, type definitions, constants at module level
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        if isinstance(node, ast.ClassDef):
            # Will prune class body, but keep structure
            return True
        if isinstance(node, ast.FunctionDef):
            # Will decide based on chunk overlap
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module-level docstring
            return True
        return False
    
    def prune(self) -> str:
        """
        Generate the pruned code representation.
        
        Returns:
            Pruned source code as a string
        """
        output_lines = []
        
        # Process each top-level node in the AST
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Always keep imports as-is
                start, end = self._get_node_lines(node)
                output_lines.append(self._extract_lines(start, end))
            
            elif isinstance(node, ast.ClassDef):
                # Prune class (keep structure, selective content)
                pruned_class = self._prune_class(node)
                output_lines.append(pruned_class)
            
            elif isinstance(node, ast.FunctionDef):
                # Prune top-level function
                pruned_func = self._prune_function(node)
                output_lines.append(pruned_func)
            
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                # Keep module-level variables
                start, end = self._get_node_lines(node)
                output_lines.append(self._extract_lines(start, end))
            
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                # Module docstring
                start, end = self._get_node_lines(node)
                output_lines.append(self._extract_lines(start, end))
            
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                # Keep other top-level statements (e.g., if __name__ == '__main__')
                start, end = self._get_node_lines(node)
                output_lines.append(self._extract_lines(start, end))
        
        # Join and clean up excessive blank lines
        result = '\n'.join(output_lines)
        
        # Remove excessive consecutive blank lines (more than 2)
        result = re.sub(r'\n\n\n+', '\n\n', result)
        
        # Ensure trailing newline
        if result and not result.endswith('\n'):
            result += '\n'
        
        return result


def parse_chunks_from_text(chunks_text: List[str]) -> List[str]:
    """
    Parse chunks from raw text strings with [PATH] and [CODE] markers.
    
    Args:
        chunks_text: List of strings in format "[PATH] path\n[CODE]\ncode content"
    
    Returns:
        List of extracted code strings
    """
    chunks = []
    
    for chunk in chunks_text:
        # Extract code between [CODE] and the next [PATH] or end
        code_match = re.search(r'\[CODE\](.*?)(?:\[PATH\]|$)', chunk, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            chunks.append(code)
    
    return chunks


# Example usage
if __name__ == "__main__":
    # Example: Django model validation
    example_source = '''
import os
import json
from django.core.exceptions import ValidationError
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _


class Model:
    """Base Django model class"""
    
    def __init__(self):
        self._meta = None
    
    def clean(self):
        """Validate the model instance"""
        pass
    
    def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )
    
    def validate_unique(self, exclude=None):
        """Validate unique constraints"""
        errors = {}
        return errors
    
    def full_clean(self, exclude=None):
        """Run full validation"""
        self.clean()
        self.validate_unique(exclude)
'''
    
    example_chunks = [
        '''def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )'''
    ]
    
    pruner = CodePruner(example_source, example_chunks)
    pruned = pruner.prune()
    print(pruned)

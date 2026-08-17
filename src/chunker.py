"""AST-based code chunker.

Splits Python source into one chunk per FunctionDef / AsyncFunctionDef /
ClassDef, preserving qualified name, source file, and line range as
metadata. Files that fail to parse fall back to fixed-size line chunking,
and the fallback is logged rather than failing silently.
"""

import ast
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FALLBACK_CHUNK_LINES = 50


@dataclass
class Chunk:
    qualified_name: str
    kind: str  # function | async_function | method | async_method | class | line_fallback
    file_path: str
    start_line: int
    end_line: int
    source: str


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, source_lines, file_path):
        self._source_lines = source_lines
        self._file_path = file_path
        self._class_stack = []
        self.chunks = []

    def visit_ClassDef(self, node):
        qualified_name = ".".join(self._class_stack + [node.name])
        self.chunks.append(self._chunk_for(node, qualified_name, "class"))
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node, is_async=True)

    def _visit_function(self, node, is_async):
        qualified_name = ".".join(self._class_stack + [node.name])
        in_class = bool(self._class_stack)
        if in_class:
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"
        self.chunks.append(self._chunk_for(node, qualified_name, kind))
        self.generic_visit(node)

    def _chunk_for(self, node, qualified_name, kind):
        start, end = node.lineno, node.end_lineno
        source = "\n".join(self._source_lines[start - 1:end])
        return Chunk(qualified_name, kind, self._file_path, start, end, source)


def chunk_source(source, file_path):
    """Chunk a single source string. Falls back to line chunking on SyntaxError."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        logger.warning(
            "failed to parse %s (%s); falling back to line chunking", file_path, exc
        )
        return _line_fallback_chunks(source, file_path)

    lines = source.splitlines()
    visitor = _DefinitionVisitor(lines, file_path)
    visitor.visit(tree)
    return visitor.chunks


def _line_fallback_chunks(source, file_path, chunk_size=FALLBACK_CHUNK_LINES):
    lines = source.splitlines()
    chunks = []
    for start in range(0, len(lines), chunk_size):
        end = min(start + chunk_size, len(lines))
        chunks.append(
            Chunk(
                qualified_name=f"{os.path.basename(file_path)}:{start + 1}-{end}",
                kind="line_fallback",
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                source="\n".join(lines[start:end]),
            )
        )
    return chunks


def chunk_directory(directory):
    """Walk a directory and chunk every .py file found."""
    all_chunks = []
    for root, _dirs, files in os.walk(directory):
        for file_name in sorted(files):
            if not file_name.endswith(".py"):
                continue
            path = os.path.join(root, file_name)
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            all_chunks.extend(chunk_source(source, path))
    return all_chunks

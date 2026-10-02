"""Robust Java method recovery from corpus fragments.

The ARPA corpus ships source-file *windows*, not compilation units or clean
methods: a fragment routinely starts mid-method, contains unrelated neighbour
methods, and ends mid-method. ROTA needs a real AST per program version, so a
whole-fragment parse is not viable - one broken neighbour poisons the tree.

This module recovers individual methods by scanning for method headers,
brace-matching each body directly out of the raw text, and validating each
candidate by parsing it standalone. Broken neighbours are simply dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tree_sitter_java
from tree_sitter import Language, Parser

JAVA = Language(tree_sitter_java.language())
_PARSER = Parser(JAVA)

MODIFIERS = (
    "public", "private", "protected", "static", "final", "synchronized",
    "abstract", "native", "default", "strictfp", "transient", "volatile",
)

#: Statement keywords that take a parenthesised head and a brace body, and so
#: look exactly like a method header to a permissive scanner.
_STMT_KEYWORDS = frozenset({
    "if", "while", "for", "switch", "catch", "try", "do", "else", "finally",
    "synchronized", "return", "new", "assert", "throw", "case", "default",
})

_HEADER = re.compile(
    r"(?:(?:@\w+(?:\([^()]*\))?\s*)*)"            # annotations
    r"(?:(?:" + "|".join(MODIFIERS) + r")\s+)*"    # modifiers
    r"(?:<[^<>;{}]*(?:<[^<>;{}]*>)?[^<>;{}]*>\s*)?"  # generic parameters
    r"(?:[\w.$]+(?:\s*<[^;{}]*?>)?(?:\s*\[\s*\])*\s+)?"  # return type
    r"(\w+)\s*"                                    # (1) method name
    r"\([^;{}]*\)\s*"                              # parameter list
    r"(?:throws\s+[\w.$,\s]+?)?"                   # throws clause
    r"\{"
)


@dataclass
class JavaMethod:
    """A brace-complete method recovered from a fragment."""

    name: str
    text: str
    start: int          # offset of the header within the fragment
    end: int            # offset just past the closing brace
    tree: object        # tree_sitter.Tree of `class W { <text> }`
    body_node: object   # tree_sitter.Node for the method_declaration

    @property
    def loc(self) -> int:
        return self.text.count("\n") + 1


def blank_noncode(src: str) -> str:
    """Replace comment and literal content with spaces, preserving offsets.

    Scanning on the blanked text means brace matching, header detection, and
    statement-boundary checks never trip over a brace in a string or a method
    signature quoted inside a Javadoc, while every offset still indexes the
    original fragment.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if c in ("'", '"'):
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                if src[j] == "\n":
                    break
                j += 1
            for k in range(i + 1, min(j - 1, n) + 1):
                if k < n and out[k] not in ("\n", quote):
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def match_brace(blank: str, open_idx: int) -> int:
    """Index just past the brace closing ``blank[open_idx]``, or -1.

    Expects comment- and literal-blanked text from :func:`blank_noncode`.
    """
    depth, i, n = 0, open_idx, len(blank)
    while i < n:
        c = blank[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _at_statement_start(blank: str, idx: int) -> bool:
    """True when idx begins a declaration rather than continuing an expression.

    Comments are already blanked, so a Javadoc above a method does not hide the
    statement boundary.
    """
    j = idx - 1
    while j >= 0 and blank[j] in " \t\r\n":
        j -= 1
    return j < 0 or blank[j] in ";{}"


def _find_method_node(tree) -> Optional[object]:
    for child in tree.root_node.children:
        if child.type != "class_declaration":
            continue
        body = child.child_by_field_name("body")
        if body is None:
            continue
        for node in body.children:
            if node.type in ("method_declaration", "constructor_declaration"):
                return node
    return None


def extract_methods(src: str, *, max_candidates: int = 400) -> List[JavaMethod]:
    """Recover every brace-complete, independently parseable method in ``src``.

    Candidates are validated by parsing, so the permissive header regex only
    needs to over-approximate. Nested methods (anonymous classes, local classes)
    are reported as part of their enclosing method rather than separately.
    """
    if not src or "{" not in src:
        return []

    blank = blank_noncode(src)
    found: List[JavaMethod] = []
    claimed: List[Tuple[int, int]] = []
    seen = 0

    for m in _HEADER.finditer(blank):
        seen += 1
        if seen > max_candidates:
            break
        name = m.group(1)
        if name in _STMT_KEYWORDS:
            continue
        if not _at_statement_start(blank, m.start()):
            continue
        if any(lo <= m.start() < hi for lo, hi in claimed):
            continue  # nested inside a method already recovered

        end = match_brace(blank, m.end() - 1)
        if end < 0:
            continue
        text = src[m.start():end]
        tree = _PARSER.parse(("class W {\n" + text + "\n}").encode("utf-8"))
        if tree.root_node.has_error:
            continue
        node = _find_method_node(tree)
        if node is None:
            continue
        found.append(JavaMethod(name, text, m.start(), end, tree, node))
        claimed.append((m.start(), end))

    return found


def key_symbols(key_var: str) -> List[str]:
    """Candidate identifiers naming the reported resource kappa."""
    out: List[str] = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\b", key_var or ""):
        w = m.group(1)
        if w in _STMT_KEYWORDS or w in ("null", "true", "false"):
            continue
        if w not in out:
            out.append(w)
    return out


def select_target_method(
    methods: List[JavaMethod],
    key_var: str,
    resource_type: str = "",
) -> Optional[JavaMethod]:
    """Pick the recovered method that owns the reported resource.

    Scoring favours a method that mentions the key variable, then one that
    mentions the simple resource type. Ties break toward the smaller method,
    which keeps the analysis scoped to the leaking method rather than an
    enclosing wrapper.
    """
    syms = key_symbols(key_var)
    simple = (resource_type or "").split(".")[-1].split("<")[0]
    best, best_key = None, None
    for meth in methods:
        score = sum(3 for s in syms if re.search(rf"\b{re.escape(s)}\b", meth.text))
        if simple and re.search(rf"\b{re.escape(simple)}\b", meth.text):
            score += 2
        if score == 0:
            continue
        key = (score, -meth.loc)
        if best_key is None or key > best_key:
            best, best_key = meth, key
    return best


def parse_method(text: str):
    """Parse a standalone method body; returns (tree, node) or (None, None)."""
    tree = _PARSER.parse(("class W {\n" + text + "\n}").encode("utf-8"))
    if tree.root_node.has_error:
        return None, None
    return tree, _find_method_node(tree)

"""
cfg_lifecycle.py — CFG-based Resource Lifecycle Analyzer
=========================================================
Implements the approach described in §3.1-§3.2 of the paper:
  1. Build scope tree (try/catch/finally/twr nesting)
  2. Identify resource allocations, releases, dependencies
  3. Post-dominance-style analysis: is release guaranteed on all paths?
  4. Compute RLDs with precise σ (Safe/Leaked/Unsafe)
  5. Compute ΔL from (M, M') RLD comparison
"""
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ════════════════════════════════════════════════════════════
#  RESOURCE TYPE DEFINITIONS
# ════════════════════════════════════════════════════════════

RESOURCE_SUFFIXES = ["Stream", "Reader", "Writer", "Connection",
                     "Statement", "ResultSet", "Channel", "Socket",
                     "Cursor", "Closeable"]

SAFE_WITHOUT_CLOSE = {
    "ByteArrayOutputStream", "ByteArrayInputStream",
    "StringWriter", "StringReader", "CharArrayWriter", "CharArrayReader",
    "URLConnection",
}

RELEASE_METHODS = {"close", "disconnect", "shutdown", "destroy", "dispose",
                   "release", "finish", "end", "recycle", "abort"}

RELEASE_PATTERNS = re.compile(
    r'(\w+)\s*\.\s*(close|disconnect|shutdown|destroy|dispose|release'
    r'|releaseConnection|finish|abort|recycle)\s*\(', re.IGNORECASE
)

CLOSE_UTILITY_PATTERNS = [
    re.compile(r'(?:IOUtils|CloseUtils|Closer|StreamUtils|DbUtils|JdbcUtils)\s*\.\s*\w*[Cc]lose\w*\s*\(\s*(\w+)'),
    re.compile(r'(?:close|release|disconnect)\w*\s*\(\s*(\w+)\s*\)'),
]


def _is_resource_type(t: str) -> bool:
    if not t:
        return False
    base = t.split('<')[0].split('.')[-1]
    if base in SAFE_WITHOUT_CLOSE:
        return True
    return any(base.endswith(s) for s in RESOURCE_SUFFIXES) or base in {
        "Connection", "Process", "HttpURLConnection", "URLConnection",
        "Selector", "RandomAccessFile", "Formatter", "Scanner",
        "CloseableHttpClient", "CloseableHttpResponse",
        "LazyFileOutputStream", "CheckedInputStream", "JarOutputStream",
        "ObjectWriter", "ObjectReader", "ObjectMapper",
        "Cipher", "ZipFile", "JarFile",
    }


# ════════════════════════════════════════════════════════════
#  SCOPE TREE (try/catch/finally nesting)
# ════════════════════════════════════════════════════════════

class ScopeType(Enum):
    METHOD = "method"
    TRY = "try"
    CATCH = "catch"
    FINALLY = "finally"
    TWR = "twr"
    IF = "if"
    ELSE = "else"
    LOOP = "loop"
    BLOCK = "block"


@dataclass
class Scope:
    type: ScopeType
    start: int
    end: int = -1
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    twr_resources: Set[str] = field(default_factory=set)

    def contains_line(self, line: int) -> bool:
        return self.start <= line <= self.end

    def find_child(self, stype: ScopeType) -> Optional['Scope']:
        for c in self.children:
            if c.type == stype:
                return c
        return None

    def find_enclosing(self, stype: ScopeType) -> Optional['Scope']:
        p = self.parent
        while p:
            if p.type == stype:
                return p
            p = p.parent
        return None


def _find_brace_end(code: str, start: int) -> int:
    depth = 0
    for i in range(start, len(code)):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(code) - 1


def _char_to_line(code: str, pos: int) -> int:
    return code[:pos].count('\n')


def build_scope_tree(code: str) -> Scope:
    """Build hierarchical scope tree from Java code using brace matching."""
    lines = code.split('\n')
    root = Scope(type=ScopeType.METHOD, start=0, end=len(lines)-1)

    twr_pattern = re.compile(r'\btry\s*\(')
    try_pattern = re.compile(r'\btry\s*\{')
    catch_pattern = re.compile(r'\bcatch\s*\(')
    finally_pattern = re.compile(r'\bfinally\s*\{')

    processed = set()

    def find_blocks(text: str, offset: int, parent: Scope):
        pos = 0
        while pos < len(text):
            abs_pos = offset + pos

            if abs_pos in processed:
                pos += 1
                continue

            # Try-with-resources
            m = twr_pattern.search(text, pos)
            mt = try_pattern.search(text, pos)
            mc = catch_pattern.search(text, pos)
            mf = finally_pattern.search(text, pos)

            candidates = []
            if m: candidates.append(('twr', m.start(), m))
            if mt: candidates.append(('try', mt.start(), mt))
            if mc: candidates.append(('catch', mc.start(), mc))
            if mf: candidates.append(('finally', mf.start(), mf))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1])
            kind, rel_pos, match_obj = candidates[0]

            abs_match = offset + match_obj.start()
            if abs_match in processed:
                pos = match_obj.end()
                continue

            processed.add(abs_match)

            if kind == 'twr':
                paren_start = offset + match_obj.end() - 1
                paren_depth = 1
                paren_end = paren_start + 1
                while paren_end < len(code) and paren_depth > 0:
                    if code[paren_end] == '(':
                        paren_depth += 1
                    elif code[paren_end] == ')':
                        paren_depth -= 1
                    paren_end += 1

                twr_decl = code[paren_start+1:paren_end-1]
                twr_vars = set()
                for decl in twr_decl.split(';'):
                    stripped = decl.strip()
                    if not stripped:
                        continue
                    if '=' in stripped:
                        before_eq = stripped.split('=')[0].strip()
                        parts = before_eq.split()
                        if len(parts) >= 2:
                            twr_vars.add(parts[-1].strip())
                    else:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            twr_vars.add(parts[-1].strip())

                brace_pos = code.find('{', paren_end)
                if brace_pos < 0:
                    pos = match_obj.end()
                    continue
                brace_end = _find_brace_end(code, brace_pos)
                scope = Scope(
                    type=ScopeType.TWR,
                    start=_char_to_line(code, abs_match),
                    end=_char_to_line(code, brace_end),
                    parent=parent,
                    twr_resources=twr_vars,
                )
                parent.children.append(scope)
                inner = code[brace_pos+1:brace_end]
                find_blocks(inner, brace_pos+1, scope)
                pos = brace_end - offset + 1

            elif kind == 'try':
                brace_pos = offset + match_obj.end() - 1
                brace_end = _find_brace_end(code, brace_pos)
                scope = Scope(
                    type=ScopeType.TRY,
                    start=_char_to_line(code, abs_match),
                    end=_char_to_line(code, brace_end),
                    parent=parent,
                )
                parent.children.append(scope)
                inner = code[brace_pos+1:brace_end]
                find_blocks(inner, brace_pos+1, scope)
                pos = brace_end - offset + 1

            elif kind == 'catch':
                after_catch = code.find('{', abs_match)
                if after_catch < 0:
                    pos = match_obj.end()
                    continue
                brace_end = _find_brace_end(code, after_catch)
                scope = Scope(
                    type=ScopeType.CATCH,
                    start=_char_to_line(code, abs_match),
                    end=_char_to_line(code, brace_end),
                    parent=parent,
                )
                parent.children.append(scope)
                inner = code[after_catch+1:brace_end]
                find_blocks(inner, after_catch+1, scope)
                pos = brace_end - offset + 1

            elif kind == 'finally':
                brace_pos = offset + match_obj.end() - 1
                brace_end = _find_brace_end(code, brace_pos)
                scope = Scope(
                    type=ScopeType.FINALLY,
                    start=_char_to_line(code, abs_match),
                    end=_char_to_line(code, brace_end),
                    parent=parent,
                )
                parent.children.append(scope)
                inner = code[brace_pos+1:brace_end]
                find_blocks(inner, brace_pos+1, scope)
                pos = brace_end - offset + 1

            else:
                pos = match_obj.end()

    find_blocks(code, 0, root)
    return root


def scope_at_line(root: Scope, line: int) -> Scope:
    """Find the innermost scope containing the given line."""
    best = root
    def walk(scope):
        nonlocal best
        if scope.contains_line(line):
            if scope.start >= best.start:
                best = scope
            for c in scope.children:
                walk(c)
    walk(root)
    return best


# ════════════════════════════════════════════════════════════
#  RESOURCE OPERATION DETECTION
# ════════════════════════════════════════════════════════════

@dataclass
class ResourceInfo:
    name: str
    rtype: str
    alloc_line: int
    is_parameter: bool = False
    wraps: Optional[str] = None
    safe_without_close: bool = False


def detect_resources(code: str) -> Tuple[Dict[str, ResourceInfo], Dict[str, str]]:
    """Detect resource allocations and parameters."""
    lines = code.split('\n')
    resources = {}
    params = {}

    sig = re.search(r'(?:public|private|protected|static|final|synchronized)\s+.*?\(([^)]*)\)', code)
    if not sig:
        sig = re.search(r'\w+\s+\w+\s*\(([^)]*)\)', code)
    if sig:
        for p in sig.group(1).split(','):
            parts = p.strip().split()
            if len(parts) >= 2:
                ptype = parts[-2].rstrip('>')
                pname = parts[-1]
                if _is_resource_type(ptype):
                    params[pname] = ptype
                    resources[pname] = ResourceInfo(
                        name=pname, rtype=ptype, alloc_line=0,
                        is_parameter=True, safe_without_close=ptype.split('.')[-1] in SAFE_WITHOUT_CLOSE
                    )

    alloc_patterns = [
        re.compile(r'(\w+)\s+(\w+)\s*=\s*new\s+(\w+)\s*\(([^)]*)\)'),
        re.compile(r'(\w+)\s+(\w+)\s*=\s*null\s*;'),
    ]

    for i, line in enumerate(lines):
        for pat in alloc_patterns:
            for m in pat.finditer(line):
                vtype = m.group(1)
                vname = m.group(2)
                ctype = m.group(3) if pat.groups >= 3 else vtype

                if not (_is_resource_type(vtype) or _is_resource_type(ctype)):
                    continue
                if vname in resources and resources[vname].alloc_line > 0:
                    continue

                actual_type = ctype if len(m.groups()) >= 3 else vtype
                args = m.group(4) if len(m.groups()) >= 4 else ""
                wraps = None
                if args:
                    for other in list(resources.keys()) + list(params.keys()):
                        if other != vname and re.search(rf'\b{re.escape(other)}\b', args):
                            wraps = other
                            break

                resources[vname] = ResourceInfo(
                    name=vname, rtype=actual_type, alloc_line=i,
                    is_parameter=False, wraps=wraps,
                    safe_without_close=actual_type.split('.')[-1] in SAFE_WITHOUT_CLOSE
                )

        late_assign = re.search(r'(\w+)\s*=\s*new\s+(\w+)\s*\(([^)]*)\)', line)
        if late_assign:
            vname = late_assign.group(1)
            ctype = late_assign.group(2)
            args = late_assign.group(3)
            if vname in resources:
                if args and resources[vname].wraps is None:
                    for other in list(resources.keys()) + list(params.keys()):
                        if other != vname and re.search(rf'\b{re.escape(other)}\b', args):
                            resources[vname].wraps = other
                            break
                if resources[vname].rtype in (ctype, vname) or not resources[vname].rtype:
                    resources[vname].rtype = ctype
                    resources[vname].safe_without_close = ctype.split('.')[-1] in SAFE_WITHOUT_CLOSE

    # Factory method allocations: conn.prepareStatement(), conn.createStatement(),
    # ps.executeQuery(), factory.getConnection(), etc.
    FACTORY_PATTERNS = [
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(prepareStatement|prepareCall)\s*\('), "PreparedStatement"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(createStatement)\s*\('), "Statement"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(executeQuery|getResultSet)\s*\('), "ResultSet"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(getConnection|borrowObject)\s*\('), "Connection"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(getInputStream|getOutputStream|getErrorStream)\s*\('), "InputStream"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(openConnection)\s*\('), "URLConnection"),
        (re.compile(r'(\w+)\s+(\w+)\s*=\s*(\w+)\.(openStream)\s*\('), "InputStream"),
    ]
    FACTORY_LATE = [
        (re.compile(r'(\w+)\s*=\s*(\w+)\.(prepareStatement|prepareCall)\s*\('), "PreparedStatement"),
        (re.compile(r'(\w+)\s*=\s*(\w+)\.(createStatement)\s*\('), "Statement"),
        (re.compile(r'(\w+)\s*=\s*(\w+)\.(executeQuery|getResultSet)\s*\('), "ResultSet"),
        (re.compile(r'(\w+)\s*=\s*(\w+)\.(getConnection|borrowObject)\s*\('), "Connection"),
        (re.compile(r'(\w+)\s*=\s*(\w+)\.(getInputStream|getOutputStream|getErrorStream)\s*\('), "InputStream"),
    ]

    for i, line in enumerate(lines):
        for pat, rtype in FACTORY_PATTERNS:
            m = pat.search(line)
            if m:
                vtype_decl = m.group(1)
                vname = m.group(2)
                if vname not in resources:
                    resources[vname] = ResourceInfo(
                        name=vname, rtype=rtype if _is_resource_type(rtype) else vtype_decl,
                        alloc_line=i, is_parameter=False,
                        wraps=None,
                        safe_without_close=(rtype.split('.')[-1] in SAFE_WITHOUT_CLOSE),
                    )

        for pat, rtype in FACTORY_LATE:
            m = pat.search(line)
            if m:
                vname = m.group(1)
                if vname in resources:
                    if not resources[vname].rtype or resources[vname].rtype == vname:
                        resources[vname].rtype = rtype

    return resources, params


def _in_mutually_exclusive_branches(s1: Scope, s2: Scope) -> bool:
    """Check if two scopes are in mutually exclusive branches (if/else, try/catch)."""
    branch_types = {ScopeType.IF, ScopeType.ELSE, ScopeType.CATCH}

    def ancestors(s):
        chain = []
        while s:
            chain.append(s)
            s = s.parent
        return chain

    a1, a2 = ancestors(s1), ancestors(s2)
    a2_set = set(id(s) for s in a2)

    for anc in a1:
        if anc.type in branch_types and anc.parent:
            for sib in anc.parent.children:
                if sib is not anc and sib.type in branch_types:
                    if id(sib) in a2_set or any(id(x) in a2_set for x in _all_descendants(sib)):
                        return True
    return False


def _all_descendants(scope: Scope) -> List[Scope]:
    result = []
    for c in scope.children:
        result.append(c)
        result.extend(_all_descendants(c))
    return result


def detect_code_warnings(code: str, resources: Dict[str, ResourceInfo]) -> List[str]:
    """Detect code-level issues that complement lifecycle analysis."""
    lines = code.split('\n')
    warnings = []

    # 1. System stream closing: resource initialized to System.out/in/err
    for i, line in enumerate(lines):
        m = re.search(r'(\w+)\s*=\s*System\.(out|in|err)\b', line)
        if m:
            vname = m.group(1)
            stream = m.group(2)
            for j, line2 in enumerate(lines):
                if re.search(rf'\b{re.escape(vname)}\s*\.\s*close\s*\(', line2):
                    warnings.append(
                        f"'{vname}' is initialized to System.{stream} and later closed — "
                        f"this would close System.{stream}, which is shared and should not be closed")
                    break

    # 2. Variable reassignment: detect when a resource variable receives a real value
    #    (not null) multiple times without closing the previous allocation
    scope_tree = build_scope_tree(code)
    real_assigns = {}
    null_assign_pat = re.compile(r'=\s*null\s*;')
    for i, line in enumerate(lines):
        for rname in resources:
            if resources[rname].is_parameter:
                continue
            assign_pat = re.compile(rf'(?<![!=<>])\b{re.escape(rname)}\s*=(?!=)\s*')
            if assign_pat.search(line) and not null_assign_pat.search(line):
                s = scope_at_line(scope_tree, i)
                is_twr_decl = False
                while s:
                    if s.type == ScopeType.TWR and rname in s.twr_resources:
                        is_twr_decl = True
                        break
                    s = s.parent
                if not is_twr_decl:
                    real_assigns.setdefault(rname, []).append(i)

    releases_cache = detect_releases(code)
    for rname, alines in real_assigns.items():
        if len(alines) >= 2:
            rels = releases_cache.get(rname, [])
            for k in range(1, len(alines)):
                prev_assign = alines[k-1]
                curr_assign = alines[k]
                prev_scope = scope_at_line(scope_tree, prev_assign)
                curr_scope = scope_at_line(scope_tree, curr_assign)
                if _in_mutually_exclusive_branches(prev_scope, curr_scope):
                    continue
                close_between = any(prev_assign < r < curr_assign for r in rels)
                if not close_between:
                    warnings.append(
                        f"'{rname}' is reassigned at line {curr_assign+1} without closing "
                        f"the previous value (assigned at line {prev_assign+1}) — potential leak")
                    break

    # 3. Double-release: resource returned to pool via releaseConnection/release AND also closed
    POOL_RELEASE_PAT = re.compile(
        r'\b(?:releaseConnection|returnConnection|returnObject|release)\s*\(\s*(\w+)\s*\)')
    pool_released = set()
    for i, line in enumerate(lines):
        m = POOL_RELEASE_PAT.search(line)
        if m:
            pool_released.add(m.group(1))
    for vname in pool_released:
        if vname in resources:
            for i, line in enumerate(lines):
                if re.search(rf'\b{re.escape(vname)}\s*\.\s*close\s*\(', line):
                    warnings.append(
                        f"'{vname}' is returned to a connection pool via releaseConnection() "
                        f"and also explicitly closed — double-release or use-after-return")
                    break

    return warnings


def detect_releases(code: str) -> Dict[str, List[int]]:
    """Detect all release operations, mapping resource name → list of line numbers."""
    lines = code.split('\n')
    releases = {}

    for i, line in enumerate(lines):
        for m in RELEASE_PATTERNS.finditer(line):
            var = m.group(1)
            releases.setdefault(var, []).append(i)

        for pat in CLOSE_UTILITY_PATTERNS:
            for m in pat.finditer(line):
                var = m.group(1)
                releases.setdefault(var, []).append(i)

    return releases


# ════════════════════════════════════════════════════════════
#  CFG-BASED LIFECYCLE STATE ANALYSIS
# ════════════════════════════════════════════════════════════

def _is_in_scope_type(root: Scope, line: int, stype: ScopeType) -> bool:
    s = scope_at_line(root, line)
    while s:
        if s.type == stype:
            return True
        s = s.parent
    return False


def _find_finally_for_try(root: Scope, try_scope: Scope) -> Optional[Scope]:
    """Find the finally block that pairs with a given try scope."""
    if try_scope.parent is None:
        return None
    siblings = try_scope.parent.children
    for i, sib in enumerate(siblings):
        if sib is try_scope:
            for j in range(i+1, len(siblings)):
                if siblings[j].type == ScopeType.FINALLY:
                    return siblings[j]
            break
    return None


def _find_enclosing_try_or_twr(root: Scope, line: int) -> Optional[Scope]:
    s = scope_at_line(root, line)
    while s:
        if s.type in (ScopeType.TRY, ScopeType.TWR):
            return s
        s = s.parent
    return None


def analyze_lifecycle_state(
    resource: ResourceInfo,
    release_lines: List[int],
    scope_tree: Scope,
    all_resources: Dict[str, ResourceInfo],
    all_releases: Dict[str, List[int]],
) -> Tuple[str, str]:
    """
    Determine lifecycle state (σ) and guard mechanism (γ) for a resource.

    Returns: (state, guard)
        state: "Safe" | "Leaked" | "Unsafe"
        guard: "try-with-resources" | "finally" | "normal-path" | "external" | "wrapper-closed(X)" | "none"
    """
    if resource.is_parameter:
        return "Safe", "external"

    if resource.safe_without_close:
        return "Safe", "memory-backed"

    # Resource wraps a parameter → closing it would cascade-close the caller's resource
    # Not closing is the correct pattern (dependency safety)
    if resource.wraps and resource.wraps in {r.name for r in all_resources.values() if r.is_parameter}:
        return "Safe", f"wraps-parameter({resource.wraps})"

    # Check if in try-with-resources
    alloc_scope = scope_at_line(scope_tree, resource.alloc_line)
    s = alloc_scope
    while s:
        if s.type == ScopeType.TWR and resource.name in s.twr_resources:
            return "Safe", "try-with-resources"
        s = s.parent

    # Check explicit release
    if release_lines:
        for rl in release_lines:
            rel_scope = scope_at_line(scope_tree, rl)

            # Release in finally block?
            if rel_scope.type == ScopeType.FINALLY:
                enclosing_try = rel_scope.parent
                if enclosing_try and enclosing_try.contains_line(resource.alloc_line):
                    return "Safe", "finally"
                fin_parent = rel_scope.find_enclosing(ScopeType.FINALLY)
                if fin_parent and fin_parent.parent and fin_parent.parent.contains_line(resource.alloc_line):
                    return "Safe", "finally"

                if resource.alloc_line < rel_scope.start:
                    return "Safe", "finally"

            # Release in a finally that's a sibling of the enclosing try
            s = rel_scope
            while s:
                if s.type == ScopeType.FINALLY:
                    return "Safe", "finally"
                s = s.parent

        for rl in release_lines:
            if rl > resource.alloc_line:
                rel_scope = scope_at_line(scope_tree, rl)

                enclosing_try = _find_enclosing_try_or_twr(scope_tree, resource.alloc_line)
                if enclosing_try:
                    fin = _find_finally_for_try(scope_tree, enclosing_try)
                    if fin:
                        for frl in release_lines:
                            if fin.contains_line(frl):
                                return "Safe", "finally"

                    if enclosing_try.contains_line(rl):
                        return "Safe", "normal-path"

                return "Safe", "normal-path"

    # If this resource wraps a base that IS already safely closed,
    # its underlying resource is managed even if this wrapper isn't explicitly closed
    if resource.wraps and resource.wraps in all_resources:
        base = all_resources[resource.wraps]
        base_releases = all_releases.get(resource.wraps, [])
        if base_releases:
            base_state, base_guard = analyze_lifecycle_state(
                base, base_releases, scope_tree, all_resources, all_releases
            )
            if base_state == "Safe":
                return "Safe", f"underlying-closed({resource.wraps})"

    # Transitive wrapper closure: walk up the wrapper chain
    # fis ← isr ← br (br wraps isr wraps fis)
    # if br is Safe → isr is Safe → fis is Safe
    visited = {resource.name}
    current = resource.name
    for _ in range(10):  # max depth guard
        wrapper_name = None
        for rn, ri in all_resources.items():
            if rn not in visited and ri.wraps == current:
                wrapper_name = rn
                break
        if not wrapper_name:
            break
        visited.add(wrapper_name)
        wr_info = all_resources[wrapper_name]
        wr_releases = all_releases.get(wrapper_name, [])
        if wr_releases:
            wr_state, wr_guard = analyze_lifecycle_state(
                wr_info, wr_releases, scope_tree,
                all_resources, all_releases
            )
            if wr_state == "Safe":
                return "Safe", f"wrapper-closed({wrapper_name})"
        current = wrapper_name

    return "Leaked", "none"


def _find_wrapper_of(name: str, resources: Dict[str, ResourceInfo]) -> Optional[str]:
    """Find a resource that wraps the given resource."""
    for rname, rinfo in resources.items():
        if rinfo.wraps == name:
            return rname
    return None


def _check_wrapper_of(name: str, resources: Dict[str, ResourceInfo],
                       visited: Set[str]) -> Optional[str]:
    """Transitively find if any wrapper chain leads to a closed resource."""
    for rname, rinfo in resources.items():
        if rname in visited:
            continue
        if rinfo.wraps == name:
            visited.add(rname)
            return rname
    return None


# ════════════════════════════════════════════════════════════
#  RLD COMPUTATION
# ════════════════════════════════════════════════════════════

@dataclass
class RLD:
    name: str
    rtype: str
    state: str
    guard: str
    dependency: Optional[str]
    is_parameter: bool
    wraps_parameter: bool


def extract_rlds(code: str) -> List[RLD]:
    """Extract Resource Lifecycle Descriptors using CFG-based analysis."""
    if not code or len(code.strip()) < 10:
        return []

    scope_tree = build_scope_tree(code)
    resources, params = detect_resources(code)
    releases = detect_releases(code)

    rlds = []
    for rname, rinfo in resources.items():
        rel_lines = releases.get(rname, [])
        state, guard = analyze_lifecycle_state(
            rinfo, rel_lines, scope_tree, resources, releases
        )

        wraps_param = False
        if rinfo.wraps and rinfo.wraps in params:
            wraps_param = True

        rlds.append(RLD(
            name=rname, rtype=rinfo.rtype,
            state=state, guard=guard,
            dependency=rinfo.wraps,
            is_parameter=rinfo.is_parameter,
            wraps_parameter=wraps_param,
        ))

    return rlds


def get_code_warnings(code: str) -> List[str]:
    """Get all code-level warnings for a method."""
    if not code or len(code.strip()) < 10:
        return []
    resources, params = detect_resources(code)
    return detect_code_warnings(code, resources)


# ════════════════════════════════════════════════════════════
#  ΔL COMPUTATION
# ════════════════════════════════════════════════════════════

def compute_delta_l(orig_rlds: List[RLD], patch_rlds: List[RLD]) -> dict:
    """Compute lifecycle transformation ΔL(M, M')."""
    orig_map = {r.name: r for r in orig_rlds}
    patch_map = {r.name: r for r in patch_rlds}

    fixed, remaining, new_violations = [], [], []

    for name, r in orig_map.items():
        pr = patch_map.get(name)
        if r.state != "Safe":
            if pr and pr.state == "Safe":
                fixed.append(name)
            elif pr:
                remaining.append(name)

    for name, r in patch_map.items():
        if name not in orig_map and r.state != "Safe":
            new_violations.append(name)
        elif name in orig_map and orig_map[name].state == "Safe" and r.state != "Safe":
            new_violations.append(name)

    dep_issues = []
    for r in patch_rlds:
        if r.wraps_parameter and r.guard in ("finally", "try-with-resources"):
            dep_issues.append(
                f"closing {r.name} may cascade-close caller parameter {r.dependency}")

    return {
        "fixed": fixed, "remaining": remaining,
        "new_violations": new_violations,
        "dependency_issues": dep_issues,
    }


# ════════════════════════════════════════════════════════════
#  CONVENIENCE: dict format for prompt construction
# ════════════════════════════════════════════════════════════

def detect_scope_errors(code: str) -> List[str]:
    """Detect variables declared inside try but used in finally (compile error)."""
    scope_tree = build_scope_tree(code)
    lines = code.split('\n')
    errors = []

    try_scopes = []
    def collect_tries(s):
        if s.type in (ScopeType.TRY, ScopeType.TWR):
            try_scopes.append(s)
        for c in s.children:
            collect_tries(c)
    collect_tries(scope_tree)

    for ts in try_scopes:
        finally_scope = None
        if ts.parent:
            for i, sib in enumerate(ts.parent.children):
                if sib is ts:
                    for j in range(i+1, len(ts.parent.children)):
                        if ts.parent.children[j].type == ScopeType.FINALLY:
                            finally_scope = ts.parent.children[j]
                            break
                    break

        if not finally_scope:
            continue

        decls_in_try = set()
        for li in range(ts.start, min(ts.end+1, len(lines))):
            for m in re.finditer(r'\b(\w+)\s+(\w+)\s*=', lines[li]):
                vtype, vname = m.group(1), m.group(2)
                if vtype[0].isupper() or vtype in ('int','long','boolean','byte','char','float','double'):
                    inner_scope = scope_at_line(scope_tree, li)
                    if inner_scope is ts or (inner_scope.find_enclosing(ScopeType.TRY) is ts):
                        decls_in_try.add(vname)

        decls_before_try = set()
        for li in range(0, ts.start):
            for m in re.finditer(r'\b(\w+)\s+(\w+)\s*=\s*null\s*;', lines[li]):
                decls_before_try.add(m.group(2))

        for li in range(finally_scope.start, min(finally_scope.end+1, len(lines))):
            line = lines[li]
            for vname in decls_in_try:
                if vname in decls_before_try:
                    continue
                if re.search(rf'\b{re.escape(vname)}\b', line):
                    errors.append(
                        f"scope error: '{vname}' declared inside try (line {ts.start+1}-{ts.end+1}) "
                        f"but used in finally (line {li+1})")

    return errors


def rld_to_dict(r: RLD) -> dict:
    return {
        "name": r.name, "type": r.rtype,
        "state": r.state, "guard": r.guard,
        "dependency": r.dependency,
        "is_parameter": r.is_parameter,
        "wraps_parameter": r.wraps_parameter,
    }

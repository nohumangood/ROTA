"""Per-version substrate for ROTA: CFG plus abstract resource facts.

Nothing in this module is novel on its own. Control-flow graphs, alias
relations, wrapper summaries, ownership and escape analysis are all standard,
and ROTA treats them purely as substrate. What ROTA builds on top of them -
cross-version lineage, paired obligation states, and transformation semantics -
lives in :mod:`rota.lineage` and :mod:`rota.relational`.

The one design choice that matters here is that resources are modelled as
*abstract resource objects* keyed by allocation site, not as variables. A
variable may be reassigned to several resources and a resource may be reachable
through several variables, so variable-keyed tracking mis-attributes obligations
in exactly the reassignment and alias cases ROTA needs to get right.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rota.javasrc import parse_method

# --------------------------------------------------------------------------- #
# resource API tables (substrate; frozen before evaluation)
# --------------------------------------------------------------------------- #

RELEASE_METHODS = frozenset({
    "close", "disconnect", "shutdown", "shutdownNow", "dispose", "release",
    "free", "destroy", "stop", "abort", "cleanup", "closeQuietly",
})

#: Helpers that release the resources handed to them as arguments. Without
#: these, the very common ``IOUtils.closeQuietly(in)`` idiom reads as a leak.
STATIC_RELEASERS = frozenset({
    "closeQuietly", "closeAll", "closeWhileHandlingException", "closeSilently",
    "close", "shutdownAndAwaitTermination", "cleanup", "dispose", "release",
    "safeClose", "quietlyClose", "closeStream", "closeResources",
})

#: Receivers whose static methods are known to release their arguments.
RELEASER_HOLDERS = frozenset({
    "IOUtils", "Closeables", "Closer", "DbUtils", "JdbcUtils", "StreamUtils",
    "FileUtils", "Util", "Utils", "IOUtil", "CloseableUtils", "ResourceUtils",
    "QuietClose", "Streams", "Resources", "IOHelper", "CloseUtil", "CloseUtils",
})

#: Types whose close() is known to propagate to the wrapped delegate.
PROPAGATING_WRAPPERS = frozenset({
    "BufferedInputStream", "BufferedOutputStream", "BufferedReader", "BufferedWriter",
    "DataInputStream", "DataOutputStream", "InputStreamReader", "OutputStreamWriter",
    "ObjectInputStream", "ObjectOutputStream", "GZIPInputStream", "GZIPOutputStream",
    "ZipInputStream", "ZipOutputStream", "PrintWriter", "PrintStream",
    "FilterInputStream", "FilterOutputStream", "PushbackInputStream", "LineNumberReader",
    "InflaterInputStream", "DeflaterOutputStream", "CheckedInputStream", "CheckedOutputStream",
    "Scanner", "BufferedRandomAccessFile", "CipherInputStream", "CipherOutputStream",
})

#: Types whose close() is known *not* to reach an underlying resource.
NON_PROPAGATING = frozenset({
    "ByteArrayInputStream", "ByteArrayOutputStream", "StringWriter", "StringReader",
    "CharArrayWriter", "CharArrayReader", "StringBufferInputStream",
})

#: Types that hold no OS resource, so an unreleased instance is not a leak.
NO_OBLIGATION = frozenset({
    "ByteArrayOutputStream", "ByteArrayInputStream", "StringWriter", "StringReader",
    "CharArrayWriter", "CharArrayReader", "StringBuilder", "StringBuffer",
})

#: Suffixes of types that own a system resource. Deliberately a frozen table
#: rather than an inference: ROTA's contribution is the relational layer, and
#: the resource-type substrate is held fixed so it cannot absorb tuning.
RESOURCE_SUFFIXES = (
    "InputStream", "OutputStream", "Reader", "Writer", "Stream",
    "Connection", "Statement", "ResultSet", "Socket", "ServerSocket",
    "Channel", "FileChannel", "Selector", "Scanner", "RandomAccessFile",
    "Session", "Cursor", "Formatter", "ZipFile", "JarFile", "Process",
    "Watcher", "WatchService", "ExecutorService", "Pool", "Transaction",
    "Client", "Response", "Subscription", "Registry", "Recorder",
)

#: Exact type names that are never resources despite matching a suffix rule or
#: appearing as a constructor result.
NON_RESOURCE_TYPES = frozenset({
    "String", "StringBuilder", "StringBuffer", "File", "Path", "URI", "URL",
    "Properties", "Object", "Integer", "Long", "Double", "Float", "Boolean",
    "Byte", "Short", "Character", "BigDecimal", "BigInteger", "Date", "Calendar",
    "Exception", "RuntimeException", "IOException", "Throwable", "Error",
    "Class", "Thread", "Runnable", "Comparator", "Pattern", "Matcher",
    "List", "ArrayList", "LinkedList", "Map", "HashMap", "TreeMap", "Set",
    "HashSet", "TreeSet", "Collection", "Iterator", "Optional", "Stream",
    "byte", "char", "int", "long", "double", "float", "boolean", "short", "void",
})


#: Calls that hand back a freshly acquired resource. Needed because an
#: allocation is often an expression rather than a ``new``, and because such a
#: call may never be bound to a variable at all.
ACQUIRING_CALLS = {
    "getResourceAsStream": "InputStream", "getInputStream": "InputStream",
    "getOutputStream": "OutputStream", "openStream": "InputStream",
    "newInputStream": "InputStream", "newOutputStream": "OutputStream",
    "newBufferedReader": "BufferedReader", "newBufferedWriter": "BufferedWriter",
    "getConnection": "Connection", "createStatement": "Statement",
    "prepareStatement": "PreparedStatement", "prepareCall": "CallableStatement",
    "executeQuery": "ResultSet", "getResultSet": "ResultSet",
    "getGeneratedKeys": "ResultSet", "getMetaData": "ResultSet",
    "accept": "Socket", "openChannel": "Channel", "getChannel": "FileChannel",
    "newChannel": "Channel", "openSelector": "Selector", "open": "Channel",
    "newDirectoryStream": "DirectoryStream", "list": "Stream", "walk": "Stream",
    "lines": "Stream", "newWatchService": "WatchService",
    "newFixedThreadPool": "ExecutorService", "newCachedThreadPool": "ExecutorService",
    "newSingleThreadExecutor": "ExecutorService", "newScheduledThreadPool": "ExecutorService",
    "newWorkStealingPool": "ExecutorService", "createSocket": "Socket",
    "getSession": "Session", "openSession": "Session", "exec": "Process",
    "start": "Process",
}


def is_resource_type(rtype: str) -> bool:
    """Whether a declared type owns a system resource that must be released."""
    simple = (rtype or "").split(".")[-1].split("<")[0].replace("[]", "").strip()
    if not simple or simple in NON_RESOURCE_TYPES:
        return False
    if simple in PROPAGATING_WRAPPERS or simple in NON_PROPAGATING:
        return True
    return simple.endswith(RESOURCE_SUFFIXES)


class Ownership(Enum):
    """Who is responsible for releasing a resource."""

    LOCAL = "local"            # created here, must be released here
    CALLER = "caller"          # a parameter; releasing it here is a hazard
    ESCAPED = "escaped"        # returned to the caller; obligation transfers out
    FIELD = "field"            # stored into object state; obligation transfers out
    UNKNOWN = "unknown"


class EventKind(Enum):
    ACQUIRE = "acquire"
    USE = "use"
    RELEASE = "release"
    WRAP = "wrap"
    ASSIGN = "assign"
    ESCAPE = "escape"
    STORE_FIELD = "store_field"


@dataclass
class ResourceEvent:
    kind: EventKind
    resource: str                  # abstract resource id
    node: int                      # CFG node index
    line: int
    text: str = ""
    target: Optional[str] = None   # wrap/alias counterpart
    guarded: str = "none"          # none | finally | try-with-resources


@dataclass
class AbstractResource:
    """A resource object identified by its allocation site, not its variable."""

    rid: str
    rtype: str
    alloc_line: int
    ownership: Ownership = Ownership.LOCAL
    symbols: Set[str] = field(default_factory=set)
    wraps: Optional[str] = None            # rid of the delegate it wraps
    wrapped_by: Optional[str] = None
    is_parameter: bool = False
    from_call: str = ""                    # producing expression, e.g. "new FileInputStream"
    #: Enclosing conditional arms, as (branching node, arm index) pairs. Two
    #: allocations whose paths disagree on some branch cannot both happen.
    branch_path: Tuple[Tuple[int, int], ...] = ()
    #: Byte range of the allocating expression, used to recover the nesting
    #: between allocations written inline inside one another.
    alloc_span: Tuple[int, int] = (0, 0)

    @property
    def simple_type(self) -> str:
        return self.rtype.split(".")[-1].split("<")[0].strip()

    @property
    def carries_obligation(self) -> bool:
        if self.is_parameter:
            return False
        return self.simple_type not in NO_OBLIGATION


@dataclass
class CFGNode:
    index: int
    kind: str                      # statement | branch | try | catch | finally | exit
    line: int
    end_line: int
    text: str
    succ: List[int] = field(default_factory=list)
    exc_succ: List[int] = field(default_factory=list)   # exceptional edges
    in_finally: bool = False
    in_try: bool = False
    twr_resources: Set[str] = field(default_factory=set)


@dataclass
class MethodIR:
    """CFG plus resource facts for one program version."""

    name: str
    source: str
    nodes: List[CFGNode] = field(default_factory=list)
    resources: Dict[str, AbstractResource] = field(default_factory=dict)
    events: List[ResourceEvent] = field(default_factory=list)
    symbol_map: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    parameters: Set[str] = field(default_factory=set)
    parse_ok: bool = True
    #: When off, a symbol denotes only its last binding, collapsing the arms of
    #: a conditional. Exposed so the component can be ablated.
    path_correspondence: bool = True
    #: Line ranges the CFG proves unreachable. A release here is dead code.
    dead_lines: Set[Tuple[int, int]] = field(default_factory=set)

    def is_dead(self, line: int) -> bool:
        return any(lo <= line <= hi for lo, hi in self.dead_lines)

    def events_of(self, rid: str) -> List[ResourceEvent]:
        return [e for e in self.events if e.resource == rid or e.target == rid]

    def resource_of_symbol(self, sym: str, line: int) -> Optional[str]:
        """Resource a symbol refers to at a given line (latest binding at or before)."""
        bindings = self.symbol_map.get(sym, [])
        best = None
        for bline, rid in bindings:
            if bline <= line:
                best = rid
        return best if best is not None else (bindings[0][1] if bindings else None)

    def resources_of_symbol(self, sym: str, line: int) -> List[str]:
        """Every resource the symbol may denote at ``line``.

        A symbol assigned in both arms of a conditional denotes either
        allocation depending on the path taken, so a close on that symbol
        discharges both. Keeping only the textually last binding would report
        the other arm as leaking on every such method.
        """
        last = self.resource_of_symbol(sym, line)
        if last is None:
            return []
        out = [last]
        if not self.path_correspondence:
            return out
        last_res = self.resources.get(last)
        for bline, rid in self.symbol_map.get(sym, []):
            if rid == last or bline > line or rid not in self.resources:
                continue
            if last_res is not None and _branch_exclusive(
                self.resources[rid].branch_path, last_res.branch_path
            ):
                out.append(rid)
        return out


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #

def _txt(node) -> str:
    return node.text.decode("utf-8", "replace") if node is not None else ""


def _walk(node, types: Set[str]):
    if node.type in types:
        yield node
    for c in node.children:
        yield from _walk(c, types)


def _enclosing(node, types: Set[str]):
    cur = node.parent
    while cur is not None:
        if cur.type in types:
            return cur
        cur = cur.parent
    return None


def _ctor_type(expr) -> str:
    t = expr.child_by_field_name("type")
    return _txt(t).strip() if t is not None else ""


def _first_resource_arg(expr, known: Set[str]) -> Optional[str]:
    """A plain variable passed as a constructor argument, i.e. the delegate.

    Restricted to arguments that *are* an identifier. A ``connection`` buried in
    ``connection.getInputStream()`` is the receiver of the call, not the
    delegate; the delegate is the call's result, which span containment links.
    """
    args = expr.child_by_field_name("arguments")
    if args is None:
        return None
    for arg in args.named_children:
        if arg.type == "identifier" and _txt(arg) in known:
            return _txt(arg)
    return None


# --------------------------------------------------------------------------- #
# IR construction
# --------------------------------------------------------------------------- #

_STMT_TYPES = {
    "local_variable_declaration", "expression_statement", "return_statement",
    "throw_statement", "if_statement", "while_statement", "for_statement",
    "enhanced_for_statement", "try_statement", "try_with_resources_statement",
    "switch_expression", "synchronized_statement", "do_statement", "break_statement",
    "continue_statement",
}


def build_ir(
    method_text: str, *, key_var: str = "", resource_type: str = "",
    path_correspondence: bool = True,
) -> MethodIR:
    """Build the per-version substrate for one method."""
    tree, node = parse_method(method_text)
    ir = MethodIR(name="", source=method_text, path_correspondence=path_correspondence)
    if node is None:
        ir.parse_ok = False
        return ir

    nm = node.child_by_field_name("name")
    ir.name = _txt(nm)
    ir.parameters = _parameters(node)

    body = node.child_by_field_name("body")
    if body is None:
        ir.parse_ok = False
        return ir

    _build_cfg(ir, body)
    ir.dead_lines = _dead_lines(body)
    _build_resources(ir, body, _resource_evidence(body, key_var, resource_type))
    _build_events(ir, body)
    _infer_ownership(ir, body)
    return ir


def _terminates(node) -> bool:
    """Whether control is guaranteed to leave the method or loop at ``node``.

    A statement-level CFG cannot tell that an if whose arms both return is a
    dead end, yet that is exactly the shape of a patch that appends a close()
    after the method has already returned. Recovering it is what lets a release
    be recognised as unreachable rather than as a discharge.
    """
    if node is None:
        return False
    t = node.type
    if t in ("return_statement", "throw_statement", "break_statement",
             "continue_statement"):
        return True
    if t == "block":
        last = [c for c in node.children if c.type in _STMT_TYPES]
        return _block_terminates(last)
    if t == "if_statement":
        cons = node.child_by_field_name("consequence")
        alt = node.child_by_field_name("alternative")
        return alt is not None and _terminates(cons) and _terminates(alt)
    if t in ("try_statement", "try_with_resources_statement"):
        for cc in node.children:
            if cc.type == "finally_clause":
                fb = cc.child_by_field_name("body") or (
                    cc.children[-1] if cc.children else None)
                if _terminates(fb):
                    return True
        body = node.child_by_field_name("body")
        if not _terminates(body):
            return False
        return all(
            _terminates(cc.child_by_field_name("body"))
            for cc in node.children if cc.type == "catch_clause"
        )
    if t in ("expression_statement",) and node.children:
        return False
    return False


def _block_terminates(stmts) -> bool:
    for i, s in enumerate(stmts):
        if _terminates(s):
            return True
        if s.type == "if_statement" and i == len(stmts) - 1:
            return _terminates(s)
    return False


def _dead_lines(body) -> Set[Tuple[int, int]]:
    """Line ranges unreachable because an earlier sibling always transfers out."""
    dead: Set[Tuple[int, int]] = set()

    def visit(block) -> None:
        stmts = [c for c in block.children if c.type in _STMT_TYPES]
        terminated = False
        for s in stmts:
            if terminated:
                dead.add((s.start_point[0], s.end_point[0]))
            elif _terminates(s):
                terminated = True
            for sub in _walk(s, {"block"}):
                if sub is not s:
                    visit(sub)
    visit(body)
    return dead


@dataclass
class _Evidence:
    """Per-method evidence for deciding what counts as a resource.

    Type tables alone miss project-specific resource classes, so two local
    signals are admitted: a symbol the method actually releases, and the
    reported type tau. Both are properties of the inputs ROTA is allowed to see.
    """

    released_symbols: Set[str] = field(default_factory=set)
    kappa_symbols: Set[str] = field(default_factory=set)
    kappa_calls: Set[str] = field(default_factory=set)
    tau: str = ""

    def accepts(self, sym: str, rtype: str, init_text: str = "") -> bool:
        if is_resource_type(rtype):
            return True
        if sym in self.released_symbols or sym in self.kappa_symbols:
            return True
        if self.tau and rtype and self.tau.split(".")[-1] == rtype.split(".")[-1]:
            return True
        # kappa given as an allocation expression names the acquiring call, not
        # a variable. A binding whose initializer invokes that call is the
        # reported resource even when its declared type is something innocuous
        # like Iterator - which tau confirms is actually a resource.
        if self.tau and init_text and self.kappa_calls:
            if any(f"{c}(" in init_text for c in self.kappa_calls):
                return True
        return False

    def typed(self, rtype: str, init_text: str) -> str:
        """Resource type to record, preferring tau when kappa located it."""
        if not is_resource_type(rtype) and self.tau and init_text and self.kappa_calls:
            if any(f"{c}(" in init_text for c in self.kappa_calls):
                return self.tau
        return rtype


def _resource_evidence(body, key_var: str, resource_type: str) -> "_Evidence":
    released: Set[str] = set()
    for call in _walk(body, {"method_invocation"}):
        obj = call.child_by_field_name("object")
        name = _txt(call.child_by_field_name("name"))
        if obj is not None and obj.type == "identifier" and name in RELEASE_METHODS:
            released.add(_txt(obj))
    # kappa is sometimes an allocation expression rather than a variable. Its
    # arguments name the inputs to the resource, not the resource, so taking
    # every identifier would register the arguments as resources too.
    stripped = re.sub(r"\([^()]*\)", "()", key_var or "")
    kappa = {
        m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\b", stripped)
        if m.group(1) not in ("new", "null", "return", "try", "finally", "this")
    }
    # Call names appearing as ``name(`` in kappa: the acquisition the report
    # points at when kappa is an expression rather than a plain variable.
    calls = {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", key_var or "")}
    return _Evidence(released, kappa, calls, resource_type or "")


def _parameters(method_node) -> Set[str]:
    out: Set[str] = set()
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return out
    for p in params.children:
        if p.type in ("formal_parameter", "spread_parameter"):
            nmn = p.child_by_field_name("name")
            if nmn is not None:
                out.add(_txt(nmn))
    return out


def _line_of(node) -> int:
    return node.start_point[0]


def _build_cfg(ir: MethodIR, body) -> None:
    """Statement-level CFG with normal and exceptional successors.

    Exceptional edges are what make release-guard reasoning possible: a release
    reachable only along normal edges does not discharge an obligation that can
    also be abandoned on a throwing path.
    """
    def collect(block, in_try: bool, in_finally: bool, twr: Set[str]) -> List[int]:
        made: List[int] = []
        for child in block.children:
            if child.type not in _STMT_TYPES:
                continue
            idx = len(ir.nodes)
            kind = (
                "try" if child.type in ("try_statement", "try_with_resources_statement")
                else "branch" if child.type in (
                    "if_statement", "while_statement", "for_statement",
                    "enhanced_for_statement", "do_statement", "switch_expression")
                else "statement"
            )
            ir.nodes.append(CFGNode(
                index=idx, kind=kind, line=_line_of(child),
                end_line=child.end_point[0], text=_txt(child)[:200],
                in_try=in_try, in_finally=in_finally, twr_resources=set(twr),
            ))
            made.append(idx)

            if child.type in ("try_statement", "try_with_resources_statement"):
                local_twr = set(twr)
                if child.type == "try_with_resources_statement":
                    res = child.child_by_field_name("resources")
                    if res is not None:
                        for d in _walk(res, {"resource"}):
                            n2 = d.child_by_field_name("name")
                            if n2 is not None:
                                local_twr.add(_txt(n2))
                tblock = child.child_by_field_name("body")
                tbody = collect(tblock, True, in_finally, local_twr) if tblock else []
                cbodies: List[int] = []
                for cc in child.children:
                    if cc.type == "catch_clause":
                        cb = cc.child_by_field_name("body")
                        if cb is not None:
                            cbodies += collect(cb, in_try, in_finally, twr)
                    elif cc.type == "finally_clause":
                        fb = cc.child_by_field_name("body") or (
                            cc.children[-1] if cc.children else None
                        )
                        if fb is not None:
                            cbodies += collect(fb, in_try, True, twr)
                # Any statement in the try can abandon the block and reach the
                # handlers, so give each one an exceptional successor.
                for t in tbody:
                    ir.nodes[t].exc_succ.extend(cbodies[:1] or [])
            else:
                for sub in child.children:
                    if sub.type == "block":
                        collect(sub, in_try, in_finally, twr)
        # Normal fallthrough between sibling statements.
        for a, b in zip(made, made[1:]):
            ir.nodes[a].succ.append(b)
        return made

    collect(body, False, False, set())


def _build_resources(ir: MethodIR, body, ev: "_Evidence") -> None:
    """Create one abstract resource per *resource* allocation site.

    Filtering here rather than later matters: a method body is mostly non-
    resource locals, and admitting them would flood the relational layer with
    obligations that never existed.
    """
    known_syms: Set[str] = set(ir.parameters)
    counter = 0
    decl_types: Dict[str, str] = {}

    param_types = _parameter_types(body)
    for param in sorted(ir.parameters):
        ptype = param_types.get(param, "")
        if not ev.accepts(param, ptype):
            continue
        rid = f"p:{param}"
        ir.resources[rid] = AbstractResource(
            rid=rid, rtype=ptype, alloc_line=0, ownership=Ownership.CALLER,
            symbols={param}, is_parameter=True,
        )
        ir.symbol_map.setdefault(param, []).append((0, rid))

    # try-with-resources declares its resources in the statement header, not in
    # a local_variable_declaration. Missing these makes a patch that converts a
    # manual close into try-with-resources look like it deleted the resource.
    for res_node in _walk(body, {"resource"}):
        nmn = res_node.child_by_field_name("name")
        tn = res_node.child_by_field_name("type")
        val = res_node.child_by_field_name("value")
        if nmn is None or val is None:
            continue
        sym, dtype = _txt(nmn), _txt(tn)
        known_syms.add(sym)
        decl_types[sym] = dtype
        if not ev.accepts(sym, dtype):
            continue
        rid, counter = _resource_from_expr(
            ir, val, dtype, sym, known_syms, counter, _line_of(res_node)
        )
        if rid:
            ir.symbol_map.setdefault(sym, []).append((_line_of(res_node), rid))

    for decl in _walk(body, {"local_variable_declaration"}):
        dtype = _txt(decl.child_by_field_name("type"))
        for d in decl.children:
            if d.type != "variable_declarator":
                continue
            nmn = d.child_by_field_name("name")
            val = d.child_by_field_name("value")
            if nmn is None:
                continue
            sym = _txt(nmn)
            known_syms.add(sym)
            decl_types[sym] = dtype
            init_text = _txt(val) if val is not None else ""
            if val is None or not ev.accepts(sym, dtype, init_text):
                continue
            rid, counter = _resource_from_expr(
                ir, val, ev.typed(dtype, init_text), sym, known_syms, counter, _line_of(d)
            )
            if rid:
                ir.symbol_map.setdefault(sym, []).append((_line_of(d), rid))

    for asg in _walk(body, {"assignment_expression"}):
        left = asg.child_by_field_name("left")
        right = asg.child_by_field_name("right")
        if left is None or right is None:
            continue
        if left.type == "field_access":
            # Storing into a field hands the obligation to the enclosing object,
            # so the resource exists but this method no longer has to close it.
            _field_store(ir, left, right, decl_types, known_syms, _line_of(asg))
            continue
        if left.type != "identifier":
            continue
        sym = _txt(left)
        known_syms.add(sym)
        if not ev.accepts(sym, decl_types.get(sym, "")):
            continue
        rid, counter = _resource_from_expr(
            ir, right, decl_types.get(sym, ""), sym, known_syms, counter, _line_of(asg)
        )
        if rid:
            # Reassignment rebinds the symbol; the previous resource keeps its
            # own obligation, which is exactly the reassignment-leak case.
            ir.symbol_map.setdefault(sym, []).append((_line_of(asg), rid))

    counter = _unbound_resources(ir, body, counter)


def _field_store(ir: MethodIR, left, right, decl_types, known, line: int) -> None:
    """Record ``this.f = expr`` as an escape of the allocated obligation."""
    fname = _txt(left.child_by_field_name("field")) or _txt(left)
    rid = None
    if right.type == "identifier":
        rid = ir.resource_of_symbol(_txt(right), line)
    else:
        name = _txt(right.child_by_field_name("name")) if right.type == "method_invocation" else ""
        rtype = _ctor_type(right) if right.type == "object_creation_expression" else \
            ACQUIRING_CALLS.get(name, "")
        if rtype and is_resource_type(rtype):
            rid = f"f{len(ir.resources)}:{rtype.split('.')[-1]}"
            ir.resources[rid] = AbstractResource(
                rid=rid, rtype=rtype, alloc_line=line, symbols={fname},
                from_call=_txt(right)[:60], ownership=Ownership.FIELD,
            )
            ir.symbol_map.setdefault(fname, []).append((line, rid))
    if rid and rid in ir.resources:
        ir.resources[rid].ownership = Ownership.FIELD
        ir.resources[rid].symbols.add(fname)
        ir.events.append(ResourceEvent(
            EventKind.STORE_FIELD, rid, 0, line, _txt(left)[:60], target=fname,
        ))


def _unbound_resources(ir: MethodIR, body, counter: int) -> int:
    """Resources allocated inline, never bound to a symbol.

    ``new Icon(..., getClass().getResourceAsStream(p))`` acquires a stream that
    no symbol names, so nothing in the method can close it. Such an obligation
    is unreachable by any symbol-driven analysis, yet it is exactly what a
    repair must address, so it gets an abstract resource of its own.
    """
    # Identified by byte range: the bindings hand back fresh node objects for
    # the same syntax, so object identity does not survive a second traversal.
    bound = {(n.start_byte, n.end_byte) for n in _bound_value_nodes(body)}
    for expr in _walk(body, {"object_creation_expression", "method_invocation"}):
        if (expr.start_byte, expr.end_byte) in bound:
            continue
        if expr.type == "object_creation_expression":
            rtype = _ctor_type(expr)
        else:
            rtype = ACQUIRING_CALLS.get(_txt(expr.child_by_field_name("name")), "")
        if not rtype or not is_resource_type(rtype):
            continue
        line = _line_of(expr)
        sym = f"<anon@{line}>"
        rid = f"u{counter}:{rtype.split('.')[-1]}"
        counter += 1
        res = AbstractResource(
            rid=rid, rtype=rtype, alloc_line=line, symbols={sym},
            from_call=_txt(expr)[:60], ownership=Ownership.LOCAL,
            branch_path=_branch_path(expr),
            alloc_span=(expr.start_byte, expr.end_byte),
        )
        ir.resources[rid] = res
        ir.symbol_map.setdefault(sym, []).append((line, rid))
    _link_nested(ir)
    return counter


def _link_nested(ir: MethodIR) -> None:
    """Connect allocations written inside one another into a wrapper chain.

    ``new DataInputStream(new FileInputStream(f))`` is one obligation, not two:
    the inner stream is released when the outer one is. Recovering the nesting
    here, rather than dropping the inner allocation, keeps it available for
    lineage, so a patch that inlines a previously named stream reads as a merge
    instead of a disappearance.
    """
    spans = [
        (r.alloc_span, rid) for rid, r in ir.resources.items()
        if r.alloc_span != (0, 0)
    ]
    for (lo, hi), rid in spans:
        best: Optional[Tuple[int, str]] = None
        for (olo, ohi), orid in spans:
            if orid == rid or not (olo < lo and hi <= ohi):
                continue
            width = ohi - olo
            if best is None or width < best[0]:
                best = (width, orid)
        if best is None:
            continue
        outer = ir.resources[best[1]]
        if outer.wraps is None:
            outer.wraps = rid
        ir.resources[rid].wrapped_by = best[1]


def _branch_path(node) -> Tuple[Tuple[int, int], ...]:
    """Conditional arms enclosing a node, outermost first."""
    path: List[Tuple[int, int]] = []
    cur = node
    while cur is not None and cur.parent is not None:
        parent = cur.parent
        if parent.type in ("if_statement", "switch_block_statement_group", "switch_block"):
            arm = next(
                (i for i, c in enumerate(parent.children)
                 if c.start_byte == cur.start_byte and c.end_byte == cur.end_byte),
                -1,
            )
            path.append((parent.start_byte, arm))
        cur = parent
    return tuple(reversed(path))


def _branch_exclusive(a: Tuple[Tuple[int, int], ...], b: Tuple[Tuple[int, int], ...]) -> bool:
    """Whether two branch paths select different arms of the same branch."""
    for (na, ia), (nb, ib) in zip(a, b):
        if na != nb:
            return False
        if ia != ib:
            return True
    return False


def _owned_by_enclosing_wrapper(expr) -> bool:
    """Whether a nested allocation is adopted by a wrapper built around it.

    In ``new BufferedReader(new FileReader(f))`` the reader carries no separate
    obligation: closing the wrapper closes it.
    """
    node = expr.parent
    while node is not None and node.type not in ("block", "method_declaration"):
        if node.type == "object_creation_expression":
            outer = _ctor_type(node).split(".")[-1].split("<")[0]
            if outer in PROPAGATING_WRAPPERS:
                return True
        node = node.parent
    return False


def _bound_value_nodes(body) -> List:
    """Allocation expressions that a declarator, resource, or assignment binds."""
    out = []
    for holder in _walk(body, {"variable_declarator", "resource", "assignment_expression"}):
        val = holder.child_by_field_name("value") or holder.child_by_field_name("right")
        while val is not None and val.type == "parenthesized_expression":
            val = val.named_child(0) if val.named_child_count else None
        if val is not None:
            out.append(val)
            # The outer call of a chain such as new A(new B(x)) binds B too.
            if val.type == "method_invocation":
                obj = val.child_by_field_name("object")
                while obj is not None and obj.type == "method_invocation":
                    out.append(obj)
                    obj = obj.child_by_field_name("object")
    return out


def _parameter_types(body) -> Dict[str, str]:
    method = _enclosing(body, {"method_declaration", "constructor_declaration"})
    out: Dict[str, str] = {}
    if method is None:
        return out
    params = method.child_by_field_name("parameters")
    if params is None:
        return out
    for p in params.children:
        if p.type in ("formal_parameter", "spread_parameter"):
            nmn = p.child_by_field_name("name")
            tn = p.child_by_field_name("type")
            if nmn is not None:
                out[_txt(nmn)] = _txt(tn)
    return out


def _resource_from_expr(
    ir: MethodIR, expr, decl_type: str, sym: str, known: Set[str],
    counter: int, line: int,
) -> Tuple[Optional[str], int]:
    """Create an abstract resource for an allocating expression."""
    if expr.type == "object_creation_expression":
        rtype = _ctor_type(expr) or decl_type
        rid = f"a{counter}:{rtype.split('.')[-1]}"
        counter += 1
        res = AbstractResource(
            rid=rid, rtype=rtype, alloc_line=line, symbols={sym},
            from_call=f"new {rtype}", branch_path=_branch_path(expr),
            alloc_span=(expr.start_byte, expr.end_byte),
        )
        delegate_sym = _first_resource_arg(expr, known)
        if delegate_sym:
            for drid in ir.resources_of_symbol(delegate_sym, line):
                res.wraps = res.wraps or drid
                ir.resources[drid].wrapped_by = rid
        ir.resources[rid] = res
        return rid, counter

    if expr.type == "method_invocation":
        name = _txt(expr.child_by_field_name("name"))
        rtype = decl_type or ACQUIRING_CALLS.get(name, "")
        rid = f"c{counter}:{(rtype or name).split('.')[-1]}"
        counter += 1
        ir.resources[rid] = AbstractResource(
            rid=rid, rtype=rtype, alloc_line=line, symbols={sym},
            from_call=name, ownership=Ownership.UNKNOWN,
            branch_path=_branch_path(expr),
            alloc_span=(expr.start_byte, expr.end_byte),
        )
        return rid, counter

    if expr.type == "identifier":
        # Aliasing: the symbol now denotes an existing resource.
        src = ir.resource_of_symbol(_txt(expr), line)
        if src and src in ir.resources:
            ir.resources[src].symbols.add(sym)
            return src, counter
    return None, counter


def _node_at_line(ir: MethodIR, line: int) -> int:
    best = 0
    for n in ir.nodes:
        if n.line <= line:
            best = n.index
    return best


def _guard_of(ir: MethodIR, line: int, sym: str) -> str:
    """Strongest guard protecting a release at ``line``.

    A release only discharges an obligation on every path when it sits in a
    finally block or a try-with-resources, so this distinction decides between
    SATISFIED_SAFE and SATISFIED_WEAK.
    """
    guard = "none"
    for n in ir.nodes:
        if not (n.line <= line <= n.end_line):
            continue
        if sym in n.twr_resources:
            return "try-with-resources"
        if n.in_finally:
            guard = "finally"
    return guard


def _static_release_events(ir: MethodIR, body) -> None:
    """Releases performed by a helper that takes the resource as an argument."""
    for call in _walk(body, {"method_invocation"}):
        mname = _txt(call.child_by_field_name("name"))
        if mname not in STATIC_RELEASERS:
            continue
        obj = call.child_by_field_name("object")
        holder = _txt(obj).split(".")[-1] if obj is not None else ""
        if holder and holder not in RELEASER_HOLDERS:
            continue
        args = call.child_by_field_name("arguments")
        if args is None:
            continue
        line = _line_of(call)
        if ir.is_dead(line):
            continue
        for arg in args.named_children:
            if arg.type != "identifier":
                continue
            sym = _txt(arg)
            for rid in ir.resources_of_symbol(sym, line):
                ir.events.append(ResourceEvent(
                    EventKind.RELEASE, rid, _node_at_line(ir, line), line,
                    f"{holder or ''}{'.' if holder else ''}{mname}({sym})",
                    guarded=_guard_of(ir, line, sym),
                ))


def _build_events(ir: MethodIR, body) -> None:
    """Record acquire/use/release/escape events against abstract resources."""
    for rid, res in ir.resources.items():
        if not res.is_parameter:
            ir.events.append(ResourceEvent(
                EventKind.ACQUIRE, rid, _node_at_line(ir, res.alloc_line),
                res.alloc_line, res.from_call,
            ))
        if res.wraps:
            ir.events.append(ResourceEvent(
                EventKind.WRAP, rid, _node_at_line(ir, res.alloc_line),
                res.alloc_line, res.from_call, target=res.wraps,
            ))

    _static_release_events(ir, body)

    for call in _walk(body, {"method_invocation"}):
        obj = call.child_by_field_name("object")
        mname = _txt(call.child_by_field_name("name"))
        if obj is None or obj.type != "identifier":
            continue
        sym = _txt(obj)
        line = _line_of(call)
        if mname in RELEASE_METHODS and ir.is_dead(line):
            # A close() the CFG cannot reach discharges nothing; counting it
            # would accept a patch that appends cleanup after a return.
            continue
        kind = EventKind.RELEASE if mname in RELEASE_METHODS else EventKind.USE
        # A release names a symbol, and the symbol may denote any of its
        # branch-exclusive allocations, so it discharges all of them.
        rids = ir.resources_of_symbol(sym, line) if kind == EventKind.RELEASE \
            else [r for r in (ir.resource_of_symbol(sym, line),) if r]
        for rid in rids:
            ir.events.append(ResourceEvent(
                kind, rid, _node_at_line(ir, line), line, f"{sym}.{mname}()",
                guarded=_guard_of(ir, line, sym),
            ))

    # try-with-resources releases its declared resources implicitly, and it does
    # so when the block *exits*. Recording the release at the header line would
    # make every use inside the block look like a use-after-release.
    # Every node inside the block carries the declaration, so the release line
    # is the last of them: the block exit.
    twr_scope: Dict[str, Tuple[int, int]] = {}
    for n in ir.nodes:
        for sym in n.twr_resources:
            lo, hi = twr_scope.get(sym, (n.line, n.end_line))
            twr_scope[sym] = (min(lo, n.line), max(hi, n.end_line))
    for sym, (lo, hi) in twr_scope.items():
        rid = ir.resource_of_symbol(sym, lo)
        if rid is None:
            continue
        ir.events.append(ResourceEvent(
            EventKind.RELEASE, rid, 0, hi,
            f"try-with-resources({sym})", guarded="try-with-resources",
        ))

    for ret in _walk(body, {"return_statement"}):
        # Only the value actually returned escapes. ``return r.next()`` yields a
        # value derived from r; r itself stays owned here, so blaming it as an
        # escape would hide that its obligation was never discharged.
        for name in _returned_values(ret):
            rid = ir.resource_of_symbol(name, _line_of(ret))
            if rid:
                ir.events.append(ResourceEvent(
                    EventKind.ESCAPE, rid, _node_at_line(ir, _line_of(ret)),
                    _line_of(ret), _txt(ret)[:80],
                ))

    for asg in _walk(body, {"assignment_expression"}):
        left = asg.child_by_field_name("left")
        right = asg.child_by_field_name("right")
        if left is None or right is None:
            continue
        if left.type == "field_access" or _txt(left).startswith("this."):
            rid = ir.resource_of_symbol(_txt(right), _line_of(asg))
            if rid:
                ir.events.append(ResourceEvent(
                    EventKind.STORE_FIELD, rid, _node_at_line(ir, _line_of(asg)),
                    _line_of(asg), _txt(asg)[:80],
                ))

    ir.events.sort(key=lambda e: (e.line, e.kind.value))


def _returned_values(ret) -> Set[str]:
    """Identifiers whose value is the one a return statement propagates out."""
    expr = None
    for c in ret.children:
        if c.type not in ("return", ";"):
            expr = c
            break

    def peel(node) -> Set[str]:
        if node is None:
            return set()
        t = node.type
        if t == "identifier":
            return {_txt(node)}
        if t == "parenthesized_expression":
            return peel(node.named_child(0) if node.named_child_count else None)
        if t == "cast_expression":
            return peel(node.child_by_field_name("value"))
        if t == "ternary_expression":
            return (peel(node.child_by_field_name("consequence"))
                    | peel(node.child_by_field_name("alternative")))
        # method_invocation, field_access, object_creation: the result escapes,
        # not the receiver.
        return set()

    return peel(expr)


def _infer_ownership(ir: MethodIR, body) -> None:
    for rid, res in ir.resources.items():
        if res.is_parameter:
            res.ownership = Ownership.CALLER
            continue
        evs = [e for e in ir.events if e.resource == rid]
        if any(e.kind == EventKind.ESCAPE for e in evs):
            res.ownership = Ownership.ESCAPED
        elif any(e.kind == EventKind.STORE_FIELD for e in evs):
            res.ownership = Ownership.FIELD
        elif res.ownership == Ownership.UNKNOWN:
            res.ownership = Ownership.UNKNOWN
        else:
            res.ownership = Ownership.LOCAL


def close_propagates(res: AbstractResource) -> Optional[bool]:
    """Whether closing this resource also closes the resource it wraps."""
    t = res.simple_type
    if t in PROPAGATING_WRAPPERS:
        return True
    if t in NON_PROPAGATING:
        return False
    return None

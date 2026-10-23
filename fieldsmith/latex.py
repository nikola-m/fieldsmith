"""A LaTeX-subset parser for field-theory expressions.

Grammar (informal)::

    expr     := term (('+'|'-') term)*
    term     := factor ( ('*' | '\\cdot' | '\\times' | '/' | <implicit>) factor )*
    factor   := ('-'|'+') factor | postfix
    postfix  := operator* atom ( '^' exponent | '^\\dagger' )*
    operator := \\partial_mu | \\nabla_mu | \\Box | D_mu | \\bar | <registered ops>
    atom     := number | \\frac{..}{..} | \\sqrt{..} | \\sin(..) ... | (expr) | {expr} | symbol[indices]

Derivative operators act on the *next postfix term*: ``\\partial_\\mu \\phi^2`` is
``d_mu (phi^2)``; use parentheses if you mean otherwise. Repeated index labels are summed.
"""
from __future__ import annotations

import re
import sympy as sp

from .tensors import Tensor, IndexError_

TOKEN_RE = re.compile(r"""
   (?P<ws>\s+|\\[,;:!\ ]|\\quad|\\qquad)
  |(?P<cmd>\\[A-Za-z]+)
  |(?P<num>\d+(?:\.\d+)?)
  |(?P<ch>[A-Za-z])
  |(?P<sym>[\^_{}()\[\]+\-*/=,|'.!])
""", re.X)

DROP_CMDS = {"left", "right", "mathrm", "mathbf", "mathit", "text", "operatorname", "displaystyle"}
BINARY_CMDS = {"cdot", "times"}
FUNCS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sinh": sp.sinh, "cosh": sp.cosh,
         "tanh": sp.tanh, "exp": sp.exp, "ln": sp.log, "log": sp.log, "arcsin": sp.asin,
         "arccos": sp.acos, "arctan": sp.atan, "sech": sp.sech, "cot": sp.cot}
DAGGER_TOKENS = {"\\dagger", "*", "\\ast", "\\star"}


class ParseError(Exception):
    pass


def tokenize(s: str):
    out, pos = [], 0
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise ParseError(f"unexpected character {s[pos]!r} at position {pos} in {s!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        val = m.group()
        if kind == "cmd" and val[1:] in DROP_CMDS:
            continue
        out.append((kind, val))
    # accents: \tilde{F} / \tilde F  ->  one symbol named "Ftilde" (likewise hat, bar-less variants)
    merged, i = [], 0
    while i < len(out):
        k, v = out[i]
        if k == "cmd" and v[1:] in ("tilde", "widetilde", "hat", "widehat", "dot", "ddot"):
            acc = {"tilde": "tilde", "widetilde": "tilde", "hat": "hat", "widehat": "hat",
                   "dot": "dot", "ddot": "ddot"}[v[1:]]
            if i + 3 < len(out) and out[i + 1][1] == "{" and out[i + 3][1] == "}":
                merged.append(("cmd", "\\" + out[i + 2][1].lstrip("\\") + acc))
                i += 4
                continue
            if i + 1 < len(out) and out[i + 1][0] in ("ch", "cmd"):
                merged.append(("cmd", "\\" + out[i + 1][1].lstrip("\\") + acc))
                i += 2
                continue
        merged.append((k, v))
        i += 1
    return merged


class OpExpr:
    """Operator-valued expression such as ``(i\\gamma^\\mu D_\\mu - m)`` awaiting an operand."""

    def __init__(self, parser, terms):
        self.parser, self.terms = parser, terms     # terms: [(left Tensor|None, [op, ...])]

    def apply(self, operand):
        out = None
        for left, ops in self.terms:
            r = operand
            for op in reversed(ops):
                r = self.parser.apply_op(op, r)
            term = r if left is None else left * r
            out = term if out is None else out + term
        return out


class Parser:
    def __init__(self, ctx, text: str):
        self.ctx, self.text = ctx, text
        self.toks = tokenize(text)
        self.pos = 0

    # ------------------------------------------------------------- token helpers
    def peek(self, k=0):
        i = self.pos + k
        return self.toks[i] if i < len(self.toks) else (None, None)

    def next(self):
        t = self.peek()
        self.pos += 1
        return t

    def at(self, val, k=0):
        return self.peek(k)[1] == val

    def expect(self, val):
        t = self.next()
        if t[1] != val:
            raise ParseError(f"expected {val!r} but found {t[1]!r} in {self.text!r}")

    def error(self, msg):
        raise ParseError(f"{msg} (in {self.text!r}, token {self.pos})")

    # ------------------------------------------------------------- entry
    def parse(self) -> Tensor:
        t = self.expr()
        if self.pos != len(self.toks):
            self.error(f"unexpected token {self.peek()[1]!r}")
        if isinstance(t, OpExpr):
            self.error("operator without an operand")
        return t

    # ------------------------------------------------------------- grammar
    def expr(self) -> Tensor:
        t = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            r = self.term()
            t = self._addx(t, r, 1 if op == "+" else -1)
        return t

    def _as_op(self, a):
        return a if isinstance(a, OpExpr) else OpExpr(self, [(a, [])])

    def _negx(self, a):
        if isinstance(a, OpExpr):
            return OpExpr(self, [((-l if l is not None else Tensor.scalar(self.ctx, -1)), ops)
                                 for l, ops in a.terms])
        return -a

    def _addx(self, a, b, sign):
        if isinstance(a, OpExpr) or isinstance(b, OpExpr):
            a, b = self._as_op(a), self._as_op(b)
            bt = b.terms if sign > 0 else self._negx(b).terms
            return OpExpr(self, a.terms + bt)
        return a + b if sign > 0 else a - b

    def _mulx(self, a, b):
        if isinstance(a, OpExpr):
            if isinstance(b, OpExpr):
                self.error("product of two operator expressions is not supported")
            return a.apply(b)
        if isinstance(b, OpExpr):
            return OpExpr(self, [((a if l is None else a * l), ops) for l, ops in b.terms])
        return a * b

    def _starts_atom(self):
        k, v = self.peek()
        if k in ("num", "ch"):
            return True
        if k == "cmd":
            return v[1:] not in BINARY_CMDS
        return v in ("(", "{")

    def term(self) -> Tensor:
        t = self.factor()
        while True:
            k, v = self.peek()
            if v in ("*",) or (k == "cmd" and v[1:] in BINARY_CMDS):
                self.next()
                t = self._mulx(t, self.factor())
            elif v == "/":
                self.next()
                d = self.factor()
                t = t * self._inverse(d)
            elif self._starts_atom():
                t = self._mulx(t, self.factor())
            else:
                return t

    def factor(self) -> Tensor:
        if self.peek()[1] == "-":
            self.next()
            return self._negx(self.factor())
        if self.peek()[1] == "+":
            self.next()
            return self.factor()
        return self.postfix()

    def _inverse(self, d: Tensor) -> Tensor:
        if not d.is_scalar:
            self.error("can only divide by a scalar")
        v = d.value()
        if isinstance(v, sp.MatrixBase):
            self.error("cannot divide by a matrix")
        return Tensor.scalar(self.ctx, 1 / v)

    # ------------------------------------------------------------- postfix / operators
    def postfix(self) -> Tensor:
        k, v = self.peek()
        if k == "cmd":
            name = v[1:]
            if name in ("partial", "nabla"):
                return self.derivative(name)
            if name == "Box":
                self.next()
                return self.operator_term(("box", None, 1))
            if name in ("bar", "overline"):
                self.next()
                return self.ctx.bar(self.postfix())
        if k == "ch" and v in self.ctx.operators and self._is_index_start(1):
            self.next()
            return self.operator_term((v, self.operator_indices(), 1))
        t = self.atom()
        return self.postfix_ops(t)

    def _is_index_start(self, k):
        return self.peek(k)[1] in ("_", "^")

    def postfix_ops(self, t: Tensor) -> Tensor:
        while self.peek()[1] in ("^", "!"):
            if self.peek()[1] == "!":
                self.next()
                t = Tensor.scalar(self.ctx, sp.factorial(t.value()))
                continue
            if self._dagger_ahead():
                self._eat_dagger()
                t = t.dagger()
                continue
            self.next()
            e = self.script_argument()
            t = self.power(t, e)
        return t

    def script_argument(self) -> Tensor:
        if self.at("{"):
            self.next()
            e = self.expr()
            self.expect("}")
            return e
        return self.atom()

    def power(self, base: Tensor, e: Tensor) -> Tensor:
        ev = e.value()
        if base.is_scalar:
            return Tensor.scalar(self.ctx, base.value() ** ev, base.rep)
        if ev == 2:
            return base.self_contract()
        self.error("only squares of tensors with free indices are supported")

    def _dagger_ahead(self):
        # ^\dagger  ^{\dagger}  ^*  ^{*}
        if self.peek()[1] != "^":
            return False
        a = self.peek(1)[1]
        if a in DAGGER_TOKENS:
            return True
        return a == "{" and self.peek(2)[1] in DAGGER_TOKENS and self.peek(3)[1] == "}"

    def _eat_dagger(self):
        self.next()
        if self.at("{"):
            self.next(); self.next(); self.next()
        else:
            self.next()

    def derivative(self, name: str):
        self.next()
        idx = self.operator_indices()
        reps = 1
        if self.at("^") and self.peek(1)[0] == "num" and idx and idx[0][3] == "coord":
            reps = int(self.peek(1)[1])
            self.next(); self.next()
        return self.operator_term((name, idx, reps))

    def operator_term(self, op):
        if self._starts_atom():
            return self.apply_op(op, self.postfix())
        return OpExpr(self, [(None, [op])])

    def apply_op(self, op, operand):
        name, idx, reps = op
        if isinstance(operand, OpExpr):
            if any(l is not None for l, _ in operand.terms):
                self.error("cannot compose a derivative with an operator that has coefficients")
            return OpExpr(self, [(None, [op] + ops) for _, ops in operand.terms])
        for _ in range(reps):
            operand = self._apply_one(name, idx, operand)
        return operand

    def _apply_one(self, name, idx, operand):
        if name == "box":
            return self.box(operand)
        if name in ("partial", "nabla"):
            return self.apply_derivative(name, idx, operand)
        tok, var, kind, how = idx[0]
        op = self.ctx.operators[name]
        if how == "number":
            res = op(operand, "__dn", "st", "d").set_var(0, var)
            return res.select(0, int(tok)).contract(self.ctx.strict)
        return op(operand, tok, kind, var).contract(self.ctx.strict)

    def apply_derivative(self, name, idx, operand):
        tok, var, kind, how = idx[0]
        if how == "coord":
            return operand.partial_coord(self.ctx.st.index(tok))
        if how == "number":
            return operand.partial_coord(int(tok))
        fn = operand.nabla if name == "nabla" else operand.partial
        return fn(tok, kind, var).contract(self.ctx.strict)

    def operator_indices(self):
        """Parse ``_x`` / ``^x`` after an operator; returns [(token, var, kind, how)]."""
        out = []
        while self.peek()[1] in ("_", "^"):
            if self.peek()[1] == "^" and not out and self.peek(1)[0] == "num":
                break
            var = "d" if self.next()[1] == "_" else "u"
            toks = self.read_script_tokens()
            for tk in toks:
                out.append(self.classify_index(tk, var))
            break
        if not out:
            self.error("derivative operator needs an index, e.g. \\partial_\\mu")
        return out

    def classify_index(self, tk, var):
        name = tk[1:] if tk.startswith("\\") else tk
        if name.isdigit():
            return (name, var, None, "number")
        if self.ctx.is_coord_name(name):
            return (name, var, None, "coord")
        kind = self.ctx.index_kind(name)
        if kind is None:
            self.error(f"unknown index name {tk!r}")
        return (name, var, kind, "index")

    def read_script_tokens(self):
        """Raw tokens of a sub/superscript argument (single token or braced group)."""
        toks = []
        if self.at("{"):
            self.next()
            while not self.at("}"):
                if self.peek()[0] is None:
                    self.error("unbalanced braces")
                toks.append(self.next()[1])
            self.next()
        else:
            k, v = self.next()
            if k == "num" and len(v) > 1 and "." not in v:
                self.toks.insert(self.pos, ("num", v[1:]))
                v = v[0]
            toks.append(v)
        flat = []
        for t in toks:
            if re.fullmatch(r"\d\d+", t):
                flat.extend(list(t))
            else:
                flat.append(t)
        return flat

    def box(self, operand):
        st = self.ctx.st
        if st.is_flat_cartesian:
            res = None
            for mu in range(st.dim):
                term = operand.partial_coord(mu).partial_coord(mu).scale(st.g[mu, mu])
                res = term if res is None else res + term
            return res
        a = operand.nabla("__b1", "st", "d")
        b = a.nabla("__b2", "st", "d")
        b = b.set_var(b.labels.index("__b2"), "u")
        b = b.relabel([("__bx" if l == "__b1" else l) for l in b.labels])
        b = b.relabel([("__bx" if l == "__b2" else l) for l in b.labels])
        return b.contract()

    # ------------------------------------------------------------- atoms
    def group(self, close):
        e = self.expr()
        self.expect(close)
        return e

    def atom(self) -> Tensor:
        k, v = self.next()
        if k == "num":
            return Tensor.scalar(self.ctx, sp.Rational(v) if "." not in v else sp.Rational(v))
        if v == "(":
            return self.group(")")
        if v == "{":
            return self.group("}")
        if k == "ch":
            return self.symbol(v)
        if k == "cmd":
            name = v[1:]
            if name == "frac":
                return self.frac()
            if name == "sqrt":
                return self.sqrt()
            if name in FUNCS:
                return self.function(name)
            if name == "pi":
                return Tensor.scalar(self.ctx, sp.pi)
            return self.symbol(name)
        self.error(f"unexpected token {v!r}")

    def frac(self):
        def arg():
            if self.at("{"):
                self.next()
                return self.group("}")
            k, v = self.peek()
            if k == "num" and len(v) == 2 and "." not in v:
                self.next()
                self.toks[self.pos:self.pos] = [("num", v[1])]
                return Tensor.scalar(self.ctx, sp.Integer(v[0]))
            return self.atom()
        a = arg()
        b = arg()
        return a * self._inverse(b)

    def sqrt(self):
        n = 2
        if self.at("["):
            self.next()
            n = self.expr().value()
            self.expect("]")
        val = self.script_argument().value()
        return Tensor.scalar(self.ctx, sp.sqrt(val) if n == 2 else sp.root(val, n))

    def function(self, name):
        power = 1
        if self.at("^"):
            self.next()
            power = self.script_argument().value()
        arg = self.postfix()
        if not arg.is_scalar:
            self.error(f"\\{name} needs a scalar argument")
        return Tensor.scalar(self.ctx, FUNCS[name](arg.value()) ** power)

    # ------------------------------------------------------------- symbols and tensors
    def symbol(self, name: str) -> Tensor:
        ctx = self.ctx
        while self.at("'"):
            self.next()
            name += "'"
        tdef = ctx.lookup_tensor(name)
        if tdef is not None and tdef.rank > 0 and self._subscript_symbol_ahead(name):
            tdef = None
        if tdef is not None and tdef.rank > 0:
            if not (name in ctx.symbols and not self._indexish_ahead()):
                return self.tensor_atom(tdef)
            shadowed = True
        else:
            shadowed = False
        # scalar-like: gather subscripts into the name
        full = name
        while self.at("_"):
            self.next()
            toks = self.read_script_tokens()
            full += "_" + "".join(t.lstrip("\\") for t in toks)
        tdef = None if (shadowed and full == name) else ctx.lookup_tensor(full)
        conj = False
        if tdef is not None and tdef.rank == 0:
            while self._dagger_ahead():
                self._eat_dagger()
                conj = not conj
            t = ctx.make_atom(tdef.name, [])
            return t.dagger() if conj else t
        if tdef is not None and tdef.rank > 0:   # e.g. "A_mu" typed with mixed subscripts
            self.error(f"tensor {full!r} needs index arguments")
        if full == "i" and "i" not in ctx.symbols:
            return Tensor.scalar(ctx, sp.I)
        return Tensor.scalar(ctx, ctx.symbol(full))

    def _subscript_symbol_ahead(self, name) -> bool:
        """True if ``name_sub`` (e.g. the coupling ``g_s``) is a declared symbol/tensor, so the
        subscript is part of a name and not an index."""
        if not self.at("_"):
            return False
        if self.at("{", 1):
            j, parts = 2, []
            while self.peek(j)[1] not in ("}", None):
                parts.append(self.peek(j)[1].lstrip("\\"))
                j += 1
            sub = "".join(parts)
        else:
            sub = self.peek(1)[1].lstrip("\\")
        cand = f"{name}_{sub}"
        return cand in self.ctx.symbols or self.ctx.lookup_tensor(cand) is not None

    def _indexish_ahead(self) -> bool:
        """True if the upcoming sub/superscript contains a letter or command (a real index)."""
        if self.peek()[1] not in ("_", "^"):
            return False
        if self.at("{", 1):
            j = 2
            while self.peek(j)[1] not in ("}", None):
                if self.peek(j)[0] in ("ch", "cmd"):
                    return True
                j += 1
            return False
        return self.peek(1)[0] in ("ch", "cmd")

    def parse_lhs(self):
        """Parse ``F^a_{\\mu\\nu}`` -> (name, [(token, var, kind, how), ...])."""
        k, v = self.next()
        name = v.lstrip("\\")
        idx = []
        while self.peek()[1] in ("_", "^") or (self.at("{") and self.at("}", 1)):
            if self.at("{"):
                self.next(); self.next()
                continue
            var = "d" if self.next()[1] == "_" else "u"
            for tk in self.read_script_tokens():
                idx.append(self.classify_index(tk, var))
        return name, idx

    def tensor_atom(self, tdef) -> Tensor:
        parsed = []
        conj = False
        variable = getattr(tdef, "variable_rank", False)
        while self.peek()[1] in ("_", "^") or (self.at("{") and self.at("}", 1)):
            if self.at("{"):
                self.next(); self.next()
                continue
            if self._dagger_ahead():
                self._eat_dagger()
                conj = not conj
                continue
            if len(parsed) >= tdef.rank and not variable:
                break
            var = "d" if self.peek()[1] == "_" else "u"
            self.next()
            for tk in self.read_script_tokens():
                parsed.append(self.classify_index(tk, var))
        if variable and not parsed:
            return Tensor.scalar(self.ctx, self.ctx.symbol(tdef.name))
        if len(parsed) != tdef.rank and not variable:
            self.error(f"tensor {tdef.name!r} has {tdef.rank} indices, got {len(parsed)}")
        t = self.ctx.make_atom(tdef.name, parsed)
        return t.dagger() if conj else t

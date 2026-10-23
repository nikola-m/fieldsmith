"""The :class:`Model` ties together spacetime, fields, parameters and the LaTeX parser."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field
from typing import Optional

import sympy as sp
from sympy.core.function import AppliedUndef

from .spacetime import Spacetime
from .tensors import Tensor, Slot, IndexError_
from .latex import Parser, ParseError

ST_NAMES = "mu nu rho sigma lambda kappa tau alpha beta gamma".split()
SP_NAMES = "i j k l m n".split()
AD_NAMES = "a b c d e f".split()


@dataclass
class TensorDef:
    name: str
    spaces: tuple = ()
    vars: tuple = ()
    data: dict = dc_field(default_factory=dict)
    rep: Optional[str] = None
    kind: str = "field"          # field | derived | const | builtin
    builder: object = None       # callable(parsed) -> Tensor, for dynamic tensors
    dynamical: bool = True
    latex: str = ""
    rngs: tuple = ()
    funcs: dict = dc_field(default_factory=dict)   # component -> applied Function (fields)

    @property
    def rank(self):
        return len(self.spaces)


class Model:
    def __init__(self, spacetime: Spacetime | None = None, strict: bool = False, eps0123: int = 1):
        self.st = spacetime or Spacetime.minkowski()
        self.strict = strict
        self.eps0123 = eps0123
        self.symbols: dict[str, sp.Symbol] = {}
        self.tensors: dict[str, TensorDef] = {}
        self.operators: dict = {}
        self.latex_names: dict[str, str] = {}
        self._conj_subs: dict = {}
        self.space_dims: dict[str, int] = {}
        self.index_kinds: dict[str, str] = {}
        for n in ST_NAMES:
            self.index_kinds[n] = "st"
        for n in SP_NAMES:
            self.index_kinds[n] = "sp"
        self._coord_names = {c.name for c in self.st.coords}
        self.gamma0 = None
        self._install_builtins()

    # ------------------------------------------------------------------ registries
    @property
    def coords(self):
        return self.st.coords

    def is_coord_name(self, name):
        return name in self._coord_names

    def index_kind(self, name):
        return self.index_kinds.get(name)

    def add_index_space(self, kind, names, dim):
        """Register an index family (e.g. gauge-adjoint ``a,b,c`` of dimension 8)."""
        self.space_dims[kind] = dim
        for n in names:
            self.index_kinds[n] = kind

    def _range(self, kind):
        if kind == "st":
            return tuple(range(self.st.dim))
        if kind == "sp":
            return tuple(range(1, self.st.dim))
        if kind not in self.space_dims:
            raise IndexError_(f"index space {kind!r} has no registered dimension")
        return tuple(range(self.space_dims[kind]))

    def lookup_tensor(self, name):
        return self.tensors.get(name)

    def symbol(self, name):
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, real=True)
        return self.symbols[name]

    def parameters(self, names, **assume):
        """Declare real constants: ``m, lam = M.parameters('m lambda', positive=True)``."""
        assume.setdefault("real", True)
        out = []
        for n in names.replace(",", " ").split():
            self.symbols[n] = sp.Symbol(n, **assume)
            out.append(self.symbols[n])
        return out[0] if len(out) == 1 else tuple(out)

    # ------------------------------------------------------------------ fields
    def _new_func(self, fname, tex):
        self.latex_names[fname] = tex
        return sp.Function(fname)(*self.coords)

    def field(self, name, spaces=(), var="", complex=False, symmetry=None,
              dynamical=True, latex=None):
        """Register a (tensor-valued) field and return its :class:`TensorDef`.

        ``spaces``: index families per slot, e.g. ``('st',)`` or ``('ad','st')``.
        ``var``: variance string per slot, ``'d'``, ``'u'`` or ``'-'`` (no variance).
        ``symmetry``: ``'sym'`` / ``'antisym'`` for rank-2 spacetime tensors.
        """
        spaces = tuple(spaces)
        var = tuple((var or "-" * len(spaces)))
        if len(var) != len(spaces):
            raise ValueError("variance string must have one character per slot")
        tex = latex or ("\\" + name if len(name) > 1 and name.isalpha() and name in GREEK else name)
        rngs = tuple(self._range(k) for k in spaces)
        data, funcs = {}, {}
        for key in itertools.product(*rngs):
            if symmetry and len(key) == 2 and key[0] > key[1]:
                continue
            if symmetry == "antisym" and len(key) == 2 and key[0] == key[1]:
                continue
            fname = name if not key else f"{name}[{','.join(map(str, key))}]"
            ctex = tex + self._comp_tex(var, key)
            f = self._new_func(fname, ctex)
            funcs[key] = f
            data[key] = f
            if complex:
                pf = self._new_func(fname + "*", "\\bar{" + tex + "}" + self._comp_tex(var, key))
                self._conj_subs[f] = pf
                self._conj_subs[pf] = f
        if symmetry:
            for key in list(funcs):
                if key[0] != key[1]:
                    data[(key[1], key[0])] = funcs[key] if symmetry == "sym" else -funcs[key]
        td = TensorDef(name, spaces, var, data, None, "field", None, dynamical, tex, rngs, funcs)
        td.complex = complex
        self.tensors[name] = td
        return td

    @staticmethod
    def _comp_tex(var, key):
        up = "".join(str(k) for v, k in zip(var, key) if v == "u")
        dn = "".join(str(k) for v, k in zip(var, key) if v == "d")
        nv = ",".join(str(k + 1) for v, k in zip(var, key) if v == "-")
        s = ""
        if up or nv:
            s += "^{" + nv + up + "}"
        if dn:
            s += "_{" + dn + "}"
        return s

    def scalar(self, name, complex=False, **kw):
        td = self.field(name, **kw, complex=complex)
        f = td.funcs[()]
        return (f, self._conj_subs[f]) if complex else f

    def vector(self, name, var="d", complex=False, **kw):
        return self.field(name, ("st",), var, complex=complex, **kw)

    def tensor(self, name, var="dd", symmetry=None, **kw):
        return self.field(name, ("st",) * len(var), var, symmetry=symmetry, **kw)

    def multiplet(self, name, n, complex=True, rep="fund", dynamical=True, latex=None):
        """A column vector of ``n`` fields in a representation space (doublet, spinor ...)."""
        tex = latex or ("\\" + name if name in GREEK else name)
        fs, ps = [], []
        for k in range(1, n + 1):
            f = self._new_func(f"{name}[{k}]", f"{tex}_{{{k}}}")
            fs.append(f)
            if complex:
                pf = self._new_func(f"{name}[{k}]*", f"{tex}_{{{k}}}^{{*}}")
                self._conj_subs[f], self._conj_subs[pf] = pf, f
                ps.append(pf)
        col = fs[0] if n == 1 else sp.Matrix(fs)
        funcs = {(): fs[0]} if n == 1 else {(i,): f for i, f in enumerate(fs)}
        td = TensorDef(name, (), (), {(): col}, rep, "field", None, dynamical, tex, (), funcs)
        td.complex = complex
        self.tensors[name] = td
        return col

    # ------------------------------------------------------------------ tensor atoms
    def make_atom(self, name, parsed):
        d = self.tensors[name]
        if d.builder is not None:
            return d.builder(parsed)
        if d.rank == 0:
            return Tensor(self, [], dict(d.data), d.rep)
        spec = [(k, "st" if k in ("st", "sp") else k, v if v != "-" else None, r)
                for k, v, r in zip(d.spaces, d.vars, d.rngs)]
        return self._apply_indices(name, spec, d.data, d.rep, parsed)

    @staticmethod
    def _compatible(slot_kind, pi):
        tok, var, kind, how = pi
        if how == "number":
            return True
        if how != "index":
            return False
        return kind == slot_kind or (slot_kind == "st" and kind == "sp")

    def _assign(self, name, kinds, parsed):
        remaining = list(parsed)
        out = []
        for k in kinds:
            for pi in remaining:
                if self._compatible(k, pi):
                    out.append(pi)
                    remaining.remove(pi)
                    break
            else:
                raise IndexError_(f"cannot match indices {[p[0] for p in parsed]} to the slots "
                                  f"{list(kinds)} of tensor '{name}'")
        return out

    def _apply_indices(self, name, spec, data, rep, parsed):
        kinds = [s[0] for s in spec]
        slots = [Slot(f"_{p}", k, ax, v, r) for p, (k, ax, v, r) in enumerate(spec)]
        T = Tensor(self, slots, data, rep)
        assign = self._assign(name, kinds, parsed)
        for p, (tok, var, kind, how) in enumerate(assign):
            if T.slots[p].axis == "st" and T.slots[p].var is not None:
                T = T.set_var(p, var)
        sels = []
        for p, (tok, var, kind, how) in enumerate(assign):
            s = T.slots[p]
            if how == "number":
                val = int(tok) - (1 if s.axis != "st" else 0)
                if val not in s.rng:
                    raise IndexError_(f"component {tok} out of range for slot {s.kind} of '{name}'")
                sels.append((p, val))
            else:
                if kind == "sp" and s.kind == "st":
                    T = T.restrict(p, tuple(range(1, self.st.dim)))
                slots = list(T.slots)
                slots[p] = T.slots[p].replace(label=tok, kind=kind)
                T = Tensor(self, slots, T.data, T.rep)
        for p, val in sorted(sels, reverse=True):
            T = T.select(p, val)
        return T

    # ------------------------------------------------------------------ builtins
    def _install_builtins(self):
        st = self.st
        d = st.dim
        if st.signature == "mostly_plus":
            flat = sp.diag(*([-1] + [1] * (d - 1)))
        elif st.signature == "euclidean":
            flat = sp.eye(d)
        else:
            flat = sp.diag(*([1] + [-1] * (d - 1)))
        rngs = (self._range("st"),) * 2
        for nm, M in (("g", st.g), ("eta", flat)):
            data = {(i, j): M[i, j] for i in range(d) for j in range(d) if M[i, j] != 0}
            self.tensors[nm] = TensorDef(nm, ("st", "st"), ("d", "d"), data, None, "builtin",
                                         None, False, nm, rngs)
        self.tensors["delta"] = TensorDef("delta", ("*", "*"), (), kind="builtin",
                                          builder=self._delta_builder, dynamical=False)
        self.tensors["epsilon"] = TensorDef("epsilon", ("*",), (), kind="builtin",
                                            builder=self._eps_builder, dynamical=False)
        self.tensors["epsilon"].variable_rank = True
        self.tensors["delta"].spaces = ("*", "*")

    def _kind_of(self, parsed):
        kinds = {p[2] for p in parsed if p[3] == "index"}
        if not kinds:
            return "st"
        if len(kinds) == 1:
            return kinds.pop()
        if kinds <= {"st", "sp"}:
            return "st"
        raise IndexError_(f"mixed index families {kinds}")

    def _delta_builder(self, parsed):
        kind = self._kind_of(parsed)
        if kind in ("st", "sp"):
            rng = self._range("st")
            data = {(i, i): sp.S.One for i in rng}
            spec = [("st", "st", "u", rng), ("st", "st", "d", rng)]
            return self._apply_indices("delta", spec, data, None, parsed)
        rng = self._range(kind)
        data = {(i, i): sp.S.One for i in rng}
        spec = [(kind, kind, None, rng)] * 2
        return self._apply_indices("delta", spec, data, None, parsed)

    def _eps_builder(self, parsed):
        r = len(parsed)
        kind = self._kind_of(parsed)
        if all(p[3] == "number" for p in parsed) and r == self.st.dim - 1:
            kind = "sp"                       # e.g. epsilon_{123}: spatial symbol
        if kind in ("st", "sp") and r == self.st.dim:
            rng = self._range("st")
            fac = sp.Integer(self.eps0123) / self.st.sqrt_g
            data = {p: fac * sp.LeviCivita(*p) for p in itertools.permutations(rng)}
            spec = [("st", "st", "u", rng)] * r
            return self._apply_indices("epsilon", spec, data, None, parsed)
        if kind == "sp" and r == self.st.dim - 1:
            rng = self._range("sp")
            data = {p: sp.LeviCivita(*p) for p in itertools.permutations(rng)}
            spec = [("sp", "st", "d", rng)] * r
            return self._apply_indices("epsilon", spec, data, None, parsed)
        if kind not in ("st", "sp") and r == self.space_dims.get(kind):
            rng = self._range(kind)
            data = {p: sp.LeviCivita(*p) for p in itertools.permutations(rng)}
            spec = [(kind, kind, None, rng)] * r
            return self._apply_indices("epsilon", spec, data, None, parsed)
        raise IndexError_(f"epsilon with {r} indices of family {kind!r} is not defined here")

    # ------------------------------------------------------------------ conj / bar
    def conj(self, v):
        if isinstance(v, sp.MatrixBase):
            return v.applyfunc(self.conj)
        v = sp.sympify(v)
        return v.xreplace(self._conj_subs).xreplace({sp.I: -sp.I})

    def bar(self, t):
        d = t.dagger()
        is_spinor = (isinstance(t.rep, tuple) and t.rep[0] in getattr(self, "spinors", ())) \
            or t.rep == "spinor"
        if is_spinor and self.gamma0 is not None:
            return d.map(lambda row: row * self.gamma0)
        return d

    # ------------------------------------------------------------------ parsing
    def parse(self, text, as_tensor=False):
        """Parse a LaTeX expression. Scalars come back as SymPy expressions, otherwise Tensors."""
        t = Parser(self, text).parse()
        if t.is_scalar and not as_tensor:
            return t.value()
        return t

    def lagrangian(self, text):
        v = self.parse(text)
        if isinstance(v, Tensor):
            raise IndexError_("Lagrangian must be a scalar; free indices remain: " + str(v.labels))
        return sp.expand(v) if not isinstance(v, sp.MatrixBase) else v

    def define(self, lhs, rhs):
        """Define a derived tensor, e.g. ``M.define(r'F_{\\mu\\nu}', r'\\partial_\\mu A_\\nu - ...')``."""
        name, idx = Parser(self, lhs).parse_lhs()
        T = self.parse(rhs, as_tensor=True)
        if not isinstance(T, Tensor):
            T = Tensor.scalar(self, T)
        labels = [i[0] for i in idx]
        if len(set(labels)) != len(labels):
            raise IndexError_("defining indices must be distinct")
        T = T.permute(labels)
        slots_spec, new = [], T
        for p, (tok, var, kind, how) in enumerate(idx):
            if T.slots[p].axis == "st" and T.slots[p].var is not None:
                new = new.set_var(p, var)
        spaces = tuple(i[2] for i in idx)
        vars_ = tuple(i[1] if new.slots[p].var is not None else "-" for p, i in enumerate(idx))
        rngs = tuple(s.rng for s in new.slots)
        td = TensorDef(name, spaces, vars_, dict(new.data), new.rep, "derived", None, False,
                       "\\" + name if name in GREEK else name, rngs)
        self.tensors[name] = td
        return td

    def evaluate(self, text):
        return self.parse(text)

    # ------------------------------------------------------------------ equations
    def functions_in(self, expr, dynamical_only=True):
        names = {n: d for n, d in self.tensors.items() if d.kind == "field"}
        funcs = set()
        for a in sp.sympify(expr).atoms(AppliedUndef):
            base = a.func.__name__.rstrip("*").split("[")[0]
            d = names.get(base)
            if d is not None and (d.dynamical or not dynamical_only):
                funcs.add(a)
        return sorted(funcs, key=lambda f: f.func.__name__)

    def euler_lagrange(self, L, fields=None):
        from .variational import euler_lagrange
        return euler_lagrange(self, L, fields)

    def check(self, text, simplify=sp.simplify):
        """Verify a LaTeX identity ``lhs = rhs`` (identically, as functions)."""
        if "=" not in text:
            raise ParseError("check() expects an equation 'lhs = rhs'")
        lhs, rhs = text.split("=", 1)
        a, b = self.parse(lhs, as_tensor=True), self.parse(rhs, as_tensor=True)
        if not isinstance(a, Tensor):
            a = Tensor.scalar(self, a)
        return a.equals(b, simplify)

    def latex(self, obj):
        from .printing import latex
        return latex(self, obj)


GREEK = set("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma "
            "tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega "
            "varphi vartheta varepsilon".split())

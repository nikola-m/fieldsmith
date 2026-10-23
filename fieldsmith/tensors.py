"""Index-aware tensors with explicit components.

A :class:`Tensor` carries labelled slots (``mu``, ``a``, ``i`` ...). Each slot has

* ``kind``  - the *index family* of the label (``st`` spacetime, ``sp`` spatial, ``ad`` adjoint ...)
* ``axis``  - the underlying vector space (``sp`` labels live on the ``st`` axis, restricted to 1..n-1)
* ``var``   - ``'u'`` (upper), ``'d'`` (lower) or ``None`` (no variance, e.g. adjoint indices)
* ``rng``   - the component values the slot ranges over

Component values are SymPy expressions *or* SymPy matrices (for representation-space
vectors such as Higgs doublets or Dirac spinors). Matrices multiply in written order and
1x1 matrices collapse to scalars, so ``phi^dagger T^a phi`` just works.
Repeated labels are contracted when tensors are multiplied (Einstein convention).
"""
from __future__ import annotations

import itertools
import sympy as sp


class IndexError_(Exception):
    pass


class Slot:
    __slots__ = ("label", "kind", "axis", "var", "rng")

    def __init__(self, label, kind, axis, var, rng):
        self.label, self.kind, self.axis, self.var, self.rng = label, kind, axis, var, tuple(rng)

    def replace(self, **kw):
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kw)
        return Slot(**d)

    def __repr__(self):
        return f"{self.label}:{self.kind}{self.var or ''}"


def _is_zero(v):
    if isinstance(v, sp.MatrixBase):
        return bool(v.is_zero_matrix)
    return v == 0


def _norm(v):
    if isinstance(v, sp.MatrixBase) and v.shape == (1, 1):
        return v[0, 0]
    return v


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, sp.MatrixBase) != isinstance(b, sp.MatrixBase):
        raise IndexError_("cannot add a scalar to a matrix-valued quantity")
    return _norm(a + b)


def _mul(a, b):
    return _norm(a * b)


class Tensor:
    def __init__(self, ctx, slots, data, rep=None):
        self.ctx = ctx
        self.slots = list(slots)
        self.data = {k: _norm(v) for k, v in data.items() if not _is_zero(v)}
        self.rep = rep

    # ----------------------------------------------------------------- basics
    @classmethod
    def scalar(cls, ctx, value, rep=None):
        return cls(ctx, [], {(): value}, rep)

    @property
    def labels(self):
        return [s.label for s in self.slots]

    @property
    def rank(self):
        return len(self.slots)

    @property
    def is_scalar(self):
        return not self.slots

    def value(self):
        if self.slots:
            raise IndexError_(f"tensor still has free indices {self.labels}")
        return self.data.get((), sp.S.Zero)

    def copy(self, **kw):
        t = Tensor(self.ctx, self.slots, dict(self.data), self.rep)
        for k, v in kw.items():
            setattr(t, k, v)
        return t

    def map(self, fn):
        return Tensor(self.ctx, self.slots, {k: fn(v) for k, v in self.data.items()}, self.rep)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self.data.get(key, sp.S.Zero)

    def components(self):
        return dict(self.data)

    def __repr__(self):
        head = f"Tensor[{', '.join(map(repr, self.slots))}]"
        if len(self.data) <= 12:
            body = ", ".join(f"{k}: {v}" for k, v in sorted(self.data.items()))
        else:
            body = f"{len(self.data)} nonzero components"
        return f"{head}{{{body}}}"

    # ----------------------------------------------------------------- metric ops
    def set_var(self, p, var):
        s = self.slots[p]
        if s.var == var or s.var is None:
            return self
        if s.axis != "st":
            slots = list(self.slots)
            slots[p] = s.replace(var=var)
            return Tensor(self.ctx, slots, self.data, self.rep)
        st = self.ctx.st
        M = st.ginv if var == "u" else st.g
        full = set(range(st.dim))
        if set(s.rng) != full:
            for mu in s.rng:
                if any(M[mu, nu] != 0 for nu in full - set(s.rng)):
                    raise IndexError_("cannot raise/lower a spatial-restricted index "
                                      "when the metric mixes time and space; use a spacetime index")
        new = {}
        for key, v in self.data.items():
            for mu in s.rng:
                c = M[mu, key[p]]
                if c != 0:
                    nk = key[:p] + (mu,) + key[p + 1:]
                    new[nk] = _add(new.get(nk), c * v)
        slots = list(self.slots)
        slots[p] = s.replace(var=var)
        return Tensor(self.ctx, slots, new, self.rep)

    def raise_(self, label):
        return self.set_var(self.labels.index(label), "u")

    def lower(self, label):
        return self.set_var(self.labels.index(label), "d")

    # ----------------------------------------------------------------- selection
    def select(self, p, value):
        """Fix slot ``p`` to the component ``value`` (removes the slot)."""
        slots = self.slots[:p] + self.slots[p + 1:]
        new = {k[:p] + k[p + 1:]: v for k, v in self.data.items() if k[p] == value}
        return Tensor(self.ctx, slots, new, self.rep)

    def restrict(self, p, rng):
        s = self.slots[p]
        slots = list(self.slots)
        slots[p] = s.replace(rng=rng)
        return Tensor(self.ctx, slots, {k: v for k, v in self.data.items() if k[p] in rng}, self.rep)

    def relabel(self, labels):
        slots = [s.replace(label=l) for s, l in zip(self.slots, labels)]
        return Tensor(self.ctx, slots, self.data, self.rep)

    def transpose(self, l1, l2):
        """Swap the roles of two indices: returns T'_{..l1..l2..} = T_{..l2..l1..}."""
        labs = [l2 if l == l1 else l1 if l == l2 else l for l in self.labels]
        return self.relabel(labs)

    def is_symmetric(self, l1, l2, simplify=sp.simplify):
        return self.equals(self.transpose(l1, l2), simplify)

    def is_antisymmetric(self, l1, l2, simplify=sp.simplify):
        return (self + self.transpose(l1, l2)).is_zero(simplify)

    def permute(self, labels):
        """Reorder slots to follow ``labels``."""
        if sorted(labels) != sorted(self.labels):
            raise IndexError_(f"free indices {self.labels} do not match {list(labels)}")
        perm = [self.labels.index(l) for l in labels]
        slots = [self.slots[i] for i in perm]
        return Tensor(self.ctx, slots, {tuple(k[i] for i in perm): v for k, v in self.data.items()},
                      self.rep)

    # ----------------------------------------------------------------- algebra
    def _aligned(self, other):
        if sorted(self.labels) != sorted(other.labels):
            raise IndexError_(f"cannot add tensors with free indices {self.labels} and {other.labels}")
        o = other.permute(self.labels)
        for p, s in enumerate(self.slots):
            if o.slots[p].var != s.var:
                o = o.set_var(p, s.var)
        return o

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.scalar(self.ctx, sp.sympify(other))
        if other.is_scalar and not self.is_scalar and other.data.get((), 0) == 0:
            return self                      # adding a scalar zero (e.g. "... = 0")
        if self.is_scalar and not other.is_scalar and self.data.get((), 0) == 0:
            return other
        o = self._aligned(other)
        new = dict(self.data)
        for k, v in o.data.items():
            new[k] = _add(new.get(k), v)
        return Tensor(self.ctx, self.slots, new, self.rep or other.rep)

    __radd__ = __add__

    def __neg__(self):
        return self.map(lambda v: -v)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.scalar(self.ctx, sp.sympify(other))
        return self + (-other)

    def scale(self, c):
        return self.map(lambda v: c * v)

    def outer(self, other):
        slots = self.slots + other.slots
        new = {}
        for ka, va in self.data.items():
            for kb, vb in other.data.items():
                new[ka + kb] = _mul(va, vb)
        if self.rep and other.rep:
            rep = None
        else:
            rep = self.rep or other.rep
        return Tensor(self.ctx, slots, new, rep)

    def contract(self, strict=False):
        t = self
        while True:
            labs = t.labels
            pair = next(((i, j) for i in range(len(labs)) for j in range(i + 1, len(labs))
                         if labs[i] == labs[j]), None)
            if pair is None:
                return t
            p, q = pair
            sp_, sq = t.slots[p], t.slots[q]
            if sp_.kind == "st" and sp_.var == sq.var and sp_.var is not None:
                if strict:
                    raise IndexError_(
                        f"index '{sp_.label}' appears twice with the same position; "
                        f"raise or lower one copy (or disable strict mode)")
                t = t.set_var(q, "u" if sq.var == "d" else "d")
            new = {}
            for key, v in t.data.items():
                if key[p] == key[q]:
                    nk = tuple(k for i, k in enumerate(key) if i not in (p, q))
                    new[nk] = _add(new.get(nk), v)
            slots = [s for i, s in enumerate(t.slots) if i not in (p, q)]
            t = Tensor(self.ctx, slots, new, t.rep)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = sp.sympify(other)
            return self.map(lambda v: other * v)
        return self.outer(other).contract(strict=getattr(self.ctx, "strict", False))

    def __rmul__(self, other):
        other = sp.sympify(other)
        return self.map(lambda v: other * v)

    def self_contract(self):
        """Full contraction T_{..} T^{..} (used for ``(F_{mn})^2`` shorthand)."""
        t = self
        dup = t.relabel([f"{l}'" for l in t.labels])
        for p, s in enumerate(t.slots):
            if s.var is not None and s.axis == "st" and s.kind == "st":
                dup = dup.set_var(p, "u" if s.var == "d" else "d")
        both = t.outer(dup)
        n = t.rank
        # contract slot p with slot n+p
        res = both
        for p in range(n):
            lab = f"__c{p}"
            slots = list(res.slots)
            slots[p] = slots[p].replace(label=lab)
            slots[n + p] = slots[n + p].replace(label=lab)
            res = Tensor(self.ctx, slots, res.data, res.rep)
        return res.contract()

    # ----------------------------------------------------------------- calculus
    def partial(self, label, kind="st", var="d"):
        """d_label T : new leading slot."""
        st = self.ctx.st
        rng = range(1, st.dim) if kind == "sp" else range(st.dim)
        new = {}
        for key, v in self.data.items():
            for mu in rng:
                dv = v.diff(st.coords[mu])
                if not _is_zero(dv):
                    new[(mu,) + key] = dv
        slot = Slot(label, kind, "st", "d", rng)
        t = Tensor(self.ctx, [slot] + self.slots, new, self.rep)
        return t.set_var(0, var) if var == "u" else t

    def partial_coord(self, i):
        c = self.ctx.st.coords[i]
        return self.map(lambda v: v.diff(c))

    def nabla(self, label, kind="st", var="d"):
        """Levi-Civita covariant derivative on spacetime slots."""
        st = self.ctx.st
        if st.is_flat_cartesian:
            return self.partial(label, kind, var)
        G = st.christoffel()
        rng = range(1, st.dim) if kind == "sp" else range(st.dim)
        new = {}
        full = [range(st.dim) if s.axis == "st" else s.rng for s in self.slots]
        for key in itertools.product(*full):
            for mu in rng:
                val = self.data.get(key, sp.S.Zero)
                tot = val.diff(st.coords[mu]) if not _is_zero(val) else sp.S.Zero
                for p, s in enumerate(self.slots):
                    if s.axis != "st" or s.kind != "st":
                        continue
                    for lam in range(st.dim):
                        k2 = key[:p] + (lam,) + key[p + 1:]
                        v2 = self.data.get(k2)
                        if v2 is None:
                            continue
                        c = G[key[p]][mu][lam] if s.var == "u" else -G[lam][mu][key[p]]
                        if c != 0:
                            tot = tot + c * v2
                if not _is_zero(tot):
                    new[(mu,) + key] = tot
        slot = Slot(label, kind, "st", "d", rng)
        t = Tensor(self.ctx, [slot] + self.slots, new, self.rep)
        return t.set_var(0, var) if var == "u" else t

    def dagger(self):
        def f(v):
            c = self.ctx.conj(v)
            return c.T if isinstance(c, sp.MatrixBase) else c
        if isinstance(self.rep, tuple):
            rep = (self.rep[0], not self.rep[1])
        else:
            rep = {"fund": "conj", "conj": "fund"}.get(self.rep, self.rep)
        return Tensor(self.ctx, self.slots, {k: f(v) for k, v in self.data.items()}, rep)

    # ----------------------------------------------------------------- comparison
    def is_zero(self, simplify=sp.simplify):
        for v in self.data.values():
            vv = v if not isinstance(v, sp.MatrixBase) else v.applyfunc(simplify)
            if isinstance(vv, sp.MatrixBase):
                if not vv.is_zero_matrix:
                    return False
            elif simplify(sp.expand(vv)) != 0:
                return False
        return True

    def equals(self, other, simplify=sp.simplify):
        if not isinstance(other, Tensor):
            other = Tensor.scalar(self.ctx, sp.sympify(other))
        if other.is_scalar and other.value() == 0:
            return self.is_zero(simplify)
        if self.is_scalar and self.value() == 0:
            return other.is_zero(simplify)
        return (self - other).is_zero(simplify)

    def simplified(self, simplify=sp.simplify):
        return self.map(lambda v: v.applyfunc(simplify) if isinstance(v, sp.MatrixBase) else simplify(v))

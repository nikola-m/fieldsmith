"""Variational calculus: Euler-Lagrange equations (any derivative order) and helpers."""
from __future__ import annotations

import sympy as sp
from sympy.calculus.euler import euler_equations
from sympy.core.function import AppliedUndef

from .tensors import Tensor, Slot


class EquationSet:
    """Field equations ``E_i = delta S / delta phi_i`` (each should vanish on shell)."""

    def __init__(self, model, funcs, exprs):
        self.model, self.funcs, self.exprs = model, list(funcs), list(exprs)

    def __iter__(self):
        return iter(self.exprs)

    def __len__(self):
        return len(self.exprs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.exprs[key]
        if isinstance(key, str):
            for f, e in zip(self.funcs, self.exprs):
                if f.func.__name__ == key:
                    return e
            raise KeyError(key)
        return self.exprs[self.funcs.index(key)]

    def as_dict(self):
        return dict(zip(self.funcs, self.exprs))

    def simplified(self, fn=sp.simplify):
        return EquationSet(self.model, self.funcs, [fn(e) for e in self.exprs])

    def tensor(self, name, slots=None):
        """Assemble the equations of field ``name`` into a tensor ``E^{..}`` (opposite variance)."""
        td = self.model.tensors[name]
        by_func = self.as_dict()
        data = {}
        for key, v in td.data.items():
            f = v if not (v.is_Mul or v.is_Add) else None
            e = by_func.get(f)
            if e is not None:
                data[key] = e
        if td.rank == 0:
            return Tensor(self.model, [], data)
        st_slots = [Slot(f"_{p}", k, "st" if k in ("st", "sp") else k,
                         {"u": "d", "d": "u"}.get(v), r)
                    for p, (k, v, r) in enumerate(zip(td.spaces, td.vars, td.rngs))]
        return Tensor(self.model, st_slots, data)

    def solve(self, *unknowns, **kw):
        return sp.solve(self.exprs, list(unknowns), dict=True, **kw)

    def subs(self, mapping):
        return EquationSet(self.model, self.funcs, [e.subs(mapping).doit() for e in self.exprs])

    def check_solution(self, mapping, simplify=sp.simplify, numeric=False, tol=1e-9, **kw):
        """True if substituting ``mapping`` (field -> expression) solves every equation.

        With ``numeric=True`` the residual is evaluated at random points instead of simplified
        symbolically (useful for transcendental solutions such as breathers or monopoles).
        """
        if numeric:
            from .analysis import numeric_residual
            return numeric_residual(self.model, self.exprs, mapping, **kw) < tol
        for e in self.exprs:
            r = simplify(e.subs(mapping).doit())
            if r != 0:
                return False
        return True

    def latex(self):
        lines = []
        for f, e in zip(self.funcs, self.exprs):
            tex = self.model.latex_names.get(f.func.__name__, f.func.__name__)
            lines.append(r"\frac{\delta S}{\delta %s} = %s = 0" % (tex, self.model.latex(e)))
        return "\n".join(lines)

    def __repr__(self):
        return "EquationSet(\n  " + "\n  ".join(
            f"{f.func.__name__}: {e} = 0" for f, e in zip(self.funcs, self.exprs)) + "\n)"


def euler_lagrange(model, L, fields=None):
    """Euler-Lagrange equations of the action density ``sqrt(|g|) L`` w.r.t. ``fields``.

    ``L`` may be a LaTeX string, a SymPy expression, or a scalar Tensor. Fields default to every
    dynamical field function appearing in ``L``. Complex fields: ``phi`` and ``phi*`` are varied
    independently, so ``delta/delta phi*`` yields the equation of motion of ``phi``.
    Higher-derivative Lagrangians are handled (Ostrogradsky-style EL operator).
    """
    if isinstance(L, str):
        L = model.lagrangian(L)
    elif isinstance(L, Tensor):
        L = L.value()
    L = sp.sympify(L)
    funcs = list(fields) if fields is not None else model.functions_in(L)
    dens = L if model.st.sqrt_g == 1 else model.st.sqrt_g * L
    exprs = []
    for f in funcs:
        eq = euler_equations(dens, [f], model.coords)[0]
        e = eq.lhs - eq.rhs
        exprs.append(e if model.st.sqrt_g == 1 else sp.simplify(e / model.st.sqrt_g))
    return EquationSet(model, funcs, exprs)

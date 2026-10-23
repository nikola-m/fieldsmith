"""Analysis helpers: fitting covariant forms, checking explicit solutions, energies."""
from __future__ import annotations

import random

import sympy as sp
from sympy.core.function import AppliedUndef

from .tensors import Tensor


def fit(model, target, basis, labels=None):
    """Express ``target`` as a linear combination of covariant building blocks.

    ``basis`` maps names to LaTeX strings (or Tensors) carrying the same free indices as
    ``target`` (in any order/variance). Returns ``{name: coefficient}`` or ``None`` if the target is
    not in the span. This turns component-level output of ``euler_lagrange`` back into covariant
    index notation, e.g. ``E^nu = -(d_mu F^{mu nu}) - m^2 A^nu``.
    """
    names = list(basis)
    bt = [model.parse(b, as_tensor=True) if isinstance(b, str) else b for b in basis.values()]
    if labels is None:
        labels = bt[0].labels
    if target.rank != len(labels):
        raise ValueError("target and basis must have the same number of free indices")
    target = target.relabel(labels)
    cs = sp.symbols(f"c0:{len(names)}")
    res = target
    for c, b in zip(cs, bt):
        res = res - b.scale(c)
    atoms = {}
    eqs = []
    for comp in res.data.values():
        expr = sp.expand(comp)
        rep = {}
        for a in sorted(expr.atoms(sp.Derivative) | expr.atoms(AppliedUndef), key=str):
            if isinstance(a, AppliedUndef) and any(a in d.atoms() for d in expr.atoms(sp.Derivative)):
                pass
            rep[a] = atoms.setdefault(a, sp.Dummy("u"))
        # replace derivatives first (they contain functions)
        ders = {a: v for a, v in rep.items() if isinstance(a, sp.Derivative)}
        fns = {a: v for a, v in rep.items() if not isinstance(a, sp.Derivative)}
        e2 = expr.xreplace(ders).xreplace(fns)
        gens = [v for v in list(ders.values()) + list(fns.values()) if e2.has(v)]
        if gens:
            eqs += sp.Poly(e2, *gens).coeffs()
        elif e2 != 0:
            eqs.append(e2)
    sol = sp.solve(eqs, cs, dict=True)
    if not sol:
        return None
    s = sol[0]
    # sympy.solve silently ignores equations that do not involve the unknowns: re-check them all
    if any(sp.simplify(e.subs(s)) != 0 for e in eqs):
        return None
    out = {n: sp.simplify(s.get(c, c)) for n, c in zip(names, cs)}
    return out if all(not v.has(*cs) for v in out.values()) else None


def numeric_residual(model, exprs, mapping, n_points=3, seed=1, param_values=None):
    """Largest |residual| of ``exprs`` after substituting ``mapping`` at random spacetime points.

    Free parameters are given random rational values (override with ``param_values``).
    """
    rng = random.Random(seed)
    worst = 0.0
    pv = dict(param_values or {})
    for e in exprs:
        r = sp.sympify(e).subs(mapping).doit()
        syms = [s for s in r.free_symbols if s not in model.coords]
        for s in syms:
            pv.setdefault(s, sp.Rational(rng.randint(5, 20), 10))
        for _ in range(n_points):
            pt = {c: sp.Rational(rng.randint(-15, 15), 10) or sp.Rational(1, 7) for c in model.coords}
            pt.update(pv)
            worst = max(worst, abs(complex(r.subs(pt).evalf(30))))
    return worst


def energy_density(model, L):
    """Canonical energy density ``T^{00}`` (equals the Hamiltonian density on shell)."""
    from .noether import canonical_stress_tensor
    T = canonical_stress_tensor(model, L)
    return sp.expand(T[0, 0])


def scalar_mass_matrix(model, V, fields, vacuum):
    """Hessian of the potential ``V`` w.r.t. ``fields`` at ``vacuum`` (dict field -> value).

    Fields not listed in ``vacuum`` sit at zero. Returns ``(M2, labels)`` where
    ``M2_ij = d^2 V / d phi_i d phi_j``; its eigenvalues are the tree-level masses squared
    (zero eigenvalues are Goldstone bosons when a continuous symmetry is broken).
    """
    V = sp.sympify(V)
    dummies = [sp.Dummy(f.func.__name__) for f in fields]
    sub = dict(zip(fields, dummies))
    Vd = V.subs(sub).doit()
    vac = {d: vacuum.get(f, 0) for f, d in zip(fields, dummies)}
    H = sp.Matrix(len(fields), len(fields), lambda i, j: sp.diff(Vd, dummies[i], dummies[j]).subs(vac))
    return H.applyfunc(sp.simplify), [f.func.__name__ for f in fields]

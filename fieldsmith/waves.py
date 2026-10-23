"""Linearisation about a background and plane-wave dispersion relations."""
from __future__ import annotations

import itertools

import sympy as sp

from .variational import EquationSet


def linearize(model, eoms, background=None, perturbation_prefix="d_"):
    """Expand field equations to first order about ``background`` (``{field: expression}``).

    Fields not in ``background`` expand about zero. Returns ``(linear_exprs, perturbations)`` where
    ``perturbations`` maps each original field function to its perturbation function.
    """
    background = {(model.tensors[k].funcs[()] if isinstance(k, str) else k): sp.sympify(v)
                  for k, v in (background or {}).items()}
    exprs = list(eoms.exprs if isinstance(eoms, EquationSet) else eoms)
    funcs = set()
    for e in exprs:
        funcs |= set(model.functions_in(e, dynamical_only=False))
    if isinstance(eoms, EquationSet):
        funcs |= set(eoms.funcs)
    eps = sp.Dummy("eps")
    pert, sub = {}, {}
    for f in sorted(funcs, key=lambda f: f.func.__name__):
        pf = sp.Function(perturbation_prefix + f.func.__name__)(*model.coords)
        model.latex_names[pf.func.__name__] = "\\delta " + model.latex_names.get(
            f.func.__name__, f.func.__name__)
        pert[f] = pf
        sub[f] = background.get(f, 0) + eps * pf
    out = []
    for e in exprs:
        ee = e.subs(sub).doit()
        out.append(sp.expand(sp.diff(ee, eps).subs(eps, 0)))
    return out, pert


class PlaneWave:
    """Substitution ``f -> a_f exp(-i(omega t - k.x))`` and the resulting algebraic matrix."""

    def __init__(self, model):
        st = model.st
        self.model = model
        if st.signature == "euclidean":
            self.omega = None
            self.k = sp.symbols(" ".join(f"k{i}" for i in range(1, st.dim + 1)), real=True)
            self.phase = sp.I * sum(k * x for k, x in zip(self.k, st.coords))
        else:
            self.omega = sp.Symbol("omega", real=True)
            ks = sp.symbols(" ".join(f"k_{c.name}" for c in st.coords[1:]), real=True)
            self.k = ks if isinstance(ks, tuple) else (ks,)
            self.phase = -sp.I * (self.omega * st.coords[0]
                                  - sum(k * x for k, x in zip(self.k, st.coords[1:])))

    def matrix(self, exprs, funcs):
        """Return (K, amplitudes) with ``K a = 0`` the plane-wave equations."""
        amps = [sp.Symbol(f"a_{f.func.__name__}", complex=True) for f in funcs]
        ph = sp.exp(self.phase)
        sub = {f: a * ph for f, a in zip(funcs, amps)}
        rows = []
        for e in exprs:
            ee = sp.expand(sp.simplify(e.subs(sub).doit() / ph))
            rows.append([sp.expand(ee.coeff(a)) for a in amps])
        return sp.Matrix(rows), amps


def dispersion(model, exprs, funcs=None, extra_symbols=()):
    """Plane-wave dispersion analysis of *linear* equations.

    Returns a dict with the matrix ``K``, its determinant, the factored determinant, and
    solutions for ``omega`` (when the determinant is not identically zero).
    """
    exprs = list(exprs)
    if funcs is None:
        funcs = []
        for e in exprs:
            for f in model.functions_in(e, dynamical_only=False):
                if f not in funcs:
                    funcs.append(f)
        funcs = sorted(funcs, key=lambda f: f.func.__name__)
    pw = PlaneWave(model)
    K, amps = pw.matrix(exprs, funcs)
    out = {"K": K, "amplitudes": amps, "wave": pw}
    if K.shape[0] == K.shape[1]:
        det = sp.factor(sp.simplify(K.det()))
        out["det"] = det
        if det != 0 and pw.omega is not None:
            out["omega"] = sp.solve(det, pw.omega)
    out["rank"] = K.rank(simplify=True)
    n = K.shape[0]
    if K.shape == (n, n) and out.get("det", 1) == 0 and out["rank"] > 0:
        # Gauge null directions. Physical modes are where the rank drops below its generic value,
        # i.e. the common zeros of all r x r minors (r = generic rank): take their gcd.
        r = out["rank"]
        minors = []
        for rows in itertools.combinations(range(n), r):
            for cols in itertools.combinations(range(n), r):
                mnr = sp.factor(K.extract(list(rows), list(cols)).det())
                if mnr != 0:
                    minors.append(mnr)
        pdet = sp.factor(sp.gcd_list(minors)) if minors else sp.S.Zero
        out["pdet"] = pdet
        if pdet != 0 and pw.omega is not None:
            out["omega"] = sp.solve(pdet, pw.omega)
    return out

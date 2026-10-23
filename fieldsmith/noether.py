"""Noether currents, canonical and Hilbert stress tensors, charges."""
from __future__ import annotations

import copy
import itertools
from math import factorial
from collections import Counter

import sympy as sp
from sympy.core.function import AppliedUndef

from .tensors import Tensor, Slot
from .spacetime import Spacetime


# --------------------------------------------------------------------------- helpers
def _field_funcs(model, L):
    return model.functions_in(L)


def _momenta(model, L, f):
    """P[(a1,..,ak)] = share of dL/d(d_{a1..ak} f) assigned to each ordering (symmetrised)."""
    X = model.coords
    derivs = [d for d in L.atoms(sp.Derivative) if d.expr == f]
    P = {}
    if not derivs:
        return P
    dummies = {d: sp.Dummy("D") for d in derivs}
    Ls = L.xreplace(dummies)
    for d, dm in dummies.items():
        c = sp.diff(Ls, dm).xreplace({v: k for k, v in dummies.items()})
        if c == 0:
            continue
        multi = []
        for v, n in d.variable_count:
            multi += [X.index(v)] * int(n)
        orderings = set(itertools.permutations(multi))
        share = c / len(orderings)
        for o in orderings:
            P[o] = P.get(o, 0) + share
    return P


def delta_L(model, L, variations):
    """First-order change of L under f -> f + eps * variations[f]."""
    eps = sp.Dummy("eps")
    sub = {f: f + eps * v for f, v in variations.items()}
    Le = L.subs(sub).doit()
    return sp.diff(Le, eps).subs(eps, 0)


class NoetherCurrent:
    """Conserved current ``J^mu`` of a continuous symmetry (components, upper index)."""

    def __init__(self, model, L, variations, J, K, E):
        self.model, self.L, self.variations, self.J, self.K = model, L, variations, J, K
        self.eoms = E

    def divergence(self):
        X = self.model.coords
        sg = self.model.st.sqrt_g
        return sum(sp.diff(sg * self.J[m], X[m]) for m in range(len(X))) / sg

    def tensor(self):
        rng = tuple(range(self.model.st.dim))
        return Tensor(self.model, [Slot("mu", "st", "st", "u", rng)],
                      {(m,): self.J[m] for m in rng})

    def identity_holds(self, simplify=sp.simplify):
        """Noether's identity  div J + sum_f E_f delta f == 0  (an off-shell identity)."""
        tot = self.divergence() + sum(self.eoms.as_dict()[f] * v for f, v in self.variations.items())
        return simplify(sp.expand(tot)) == 0

    def conserved_on_shell(self, simplify=sp.simplify):
        return self.identity_holds(simplify)

    def charge_density(self):
        return self.J[0]

    def __repr__(self):
        return "NoetherCurrent(" + ", ".join(f"J^{m}={self.J[m]}" for m in range(len(self.J))) + ")"


def symmetry_defect(model, L, variations, K=None):
    """``delta L - d_mu K^mu``; zero iff the variation is a symmetry (with boundary term K)."""
    L = sp.sympify(L)
    dL = delta_L(model, L, variations)
    div = 0
    if K is not None:
        div = sum(sp.diff(K[m], model.coords[m]) for m in range(len(K)))
    return sp.simplify(sp.expand(dL - div))


def noether_current(model, L, variations, K=None, check=True):
    """Noether current for ``delta f = variations[f]`` (any derivative order).

    ``variations``: ``{field_function_or_name: expression}``. For complex fields supply both
    ``phi`` and ``phi*`` variations (use :func:`complex_variation`).  ``K`` is the optional boundary
    term with ``delta L = d_mu K^mu``. Returns J = J_0 - K.
    """
    if isinstance(L, str):
        L = model.lagrangian(L)
    L = sp.expand(sp.sympify(L))
    var = {}
    for f, v in variations.items():
        if isinstance(f, str):
            f = model.tensors[f].funcs[()]
        var[f] = model.parse(v) if isinstance(v, str) else sp.sympify(v)
    X, d = model.coords, model.st.dim
    if check:
        defect = symmetry_defect(model, L, var, K)
        if defect != 0:
            raise ValueError(f"not a symmetry: delta L - div K = {defect}")
    J0 = [sp.S.Zero] * d
    for f, df in var.items():
        P = _momenta(model, L, f)
        for mu in range(d):
            for alpha, p in P.items():
                if alpha[0] != mu:
                    continue
                k = len(alpha)
                for j in range(k):
                    nu = alpha[1:1 + j]
                    rest = alpha[1 + j:]
                    term = p
                    for n in nu:
                        term = sp.diff(term, X[n])
                    dd = df
                    for n in rest:
                        dd = sp.diff(dd, X[n])
                    J0[mu] += (-1) ** j * term * dd
    J = list(J0)
    if K is not None:
        J = [J[m] - K[m] for m in range(d)]
    J = [sp.expand(j) for j in J]
    E = model.euler_lagrange(L, fields=list(var.keys()))
    nc = NoetherCurrent(model, L, var, J, K, E)
    return nc


def complex_variation(model, name, expr):
    """Variation of a complex scalar / multiplet ``name`` and of its conjugate.

    ``expr`` (LaTeX or SymPy; a column vector for multiplets) is ``delta Phi``; the conjugate field
    varies by the complex conjugate. Returns ``{Phi_k: delta Phi_k, Phi_k*: conj(delta Phi_k)}``.
    """
    td = model.tensors[name]
    e = model.parse(expr) if isinstance(expr, str) else expr
    if () in td.data:                       # multiplet (or one-component multiplet / scalar)
        col = td.data[()]
        comps = list(col) if isinstance(col, sp.MatrixBase) else [col]
        vals = list(e) if isinstance(e, sp.MatrixBase) else [sp.sympify(e)]
    else:
        raise ValueError(f"{name!r} is not a scalar or multiplet field")
    out = {}
    for f, v in zip(comps, vals):
        out[f] = v
        out[model._conj_subs[f]] = model.conj(v)
    return out


# --------------------------------------------------------------------------- stress tensors
def canonical_stress_tensor(model, L):
    """Canonical (Noether) tensor ``T^mu{}_nu = sum P^mu d_nu f - delta^mu_nu L`` (all orders)."""
    if isinstance(L, str):
        L = model.lagrangian(L)
    L = sp.expand(sp.sympify(L))
    d, X = model.st.dim, model.coords
    data = {}
    for nu in range(d):
        var = {f: sp.diff(f, X[nu]) for f in _field_funcs(model, L)}
        K = [L if m == nu else 0 for m in range(d)]
        nc = noether_current(model, L, var, K, check=False)
        for mu in range(d):
            if nc.J[mu] != 0:
                data[(mu, nu)] = nc.J[mu]
    rng = tuple(range(d))
    T = Tensor(model, [Slot("mu", "st", "st", "u", rng), Slot("nu", "st", "st", "d", rng)], data)
    return T


def hilbert_stress_tensor(model, L_latex):
    """Symmetric Hilbert tensor ``T_{mu nu} = (2 s/sqrt|g|) d(sqrt|g| L)/d g^{mu nu}``.

    ``L_latex`` must be a LaTeX string so it can be re-parsed on a perturbed metric
    ``g^{mu nu} = eta^{mu nu} + h^{mu nu}``. Derived tensors used inside should be defined
    with lower-index (metric-independent) building blocks, e.g. ``F_{mu nu}``.
    Returns the tensor with both indices lowered.
    """
    st = model.st
    if not st.is_flat_cartesian:
        raise NotImplementedError("Hilbert tensor is evaluated about flat space")
    d = st.dim
    eta = st.g
    hs = {}
    H = sp.zeros(d, d)
    for i in range(d):
        for j in range(i, d):
            hs[(i, j)] = sp.Symbol(f"h_{i}{j}")
            H[i, j] = H[j, i] = hs[(i, j)]
    g_up = st.ginv + H
    g_dn = eta - eta * H * eta
    sqrt_g = 1 - (H * eta).trace() / 2
    pst = Spacetime(st.coords, g_dn, ginv=g_up, sqrt_g=sqrt_g, signature=st.signature, name="perturbed")
    m2 = copy.copy(model)
    m2.st = pst
    m2.tensors = dict(model.tensors)
    m2._install_builtins()
    # restore the metric tensor values used for index gymnastics but keep eta flat
    Lp = m2.parse(L_latex)
    F = sp.expand(sqrt_g * Lp)
    zero = {h: 0 for h in hs.values()}
    data = {}
    for (i, j), h in hs.items():
        c = sp.diff(F, h).subs(zero)
        if i != j:
            c = c / 2
        val = st.sign * 2 * c
        if val != 0:
            data[(i, j)] = sp.expand(val)
            if i != j:
                data[(j, i)] = sp.expand(val)
    rng = tuple(range(d))
    return Tensor(model, [Slot("mu", "st", "st", "d", rng), Slot("nu", "st", "st", "d", rng)], data)

"""Gauge theory toolkit: Lie algebras, Yang-Mills fields, matter multiplets, covariant derivatives.

Conventions (Peskin & Schroeder)
--------------------------------
    D_mu      = d_mu - i g A^a_mu T^a                      (on a multiplet)
    D_mu phi^a = d_mu phi^a + g f^{abc} A^b_mu phi^c        (adjoint index)
    F^a_{mn}  = d_m A^a_n - d_n A^a_m + g f^{abc} A^b_m A^c_n
    delta phi = i alpha^a T^a phi
    delta A^a_mu = (1/g) d_mu alpha^a + f^{abc} A^b_mu alpha^c
Adjoint (colour) indices are written ``a,b,c,d,e,f`` for the first group factor and numbered from 1.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field

import sympy as sp

from .model import TensorDef
from .tensors import Tensor, Slot, IndexError_


# --------------------------------------------------------------------------- Lie algebras
class LieAlgebra:
    def __init__(self, name, generators, f=None):
        self.name = name
        self.generators = [sp.Matrix(g) for g in generators]
        self.dim = len(self.generators)
        self.N = self.generators[0].shape[0]
        self.f = f if f is not None else self._structure_constants()

    def _structure_constants(self):
        f = {}
        T = self.generators
        for a, b, c in itertools.product(range(self.dim), repeat=3):
            v = sp.nsimplify(sp.simplify(-2 * sp.I * ((T[a] * T[b] - T[b] * T[a]) * T[c]).trace()))
            if v != 0:
                f[(a, b, c)] = sp.simplify(v)
        return f

    @classmethod
    def su(cls, N):
        gens = []
        for k in range(2, N + 1):
            for j in range(1, k):
                S = sp.zeros(N, N)
                S[j - 1, k - 1] = S[k - 1, j - 1] = 1
                gens.append(S / 2)
                A = sp.zeros(N, N)
                A[j - 1, k - 1], A[k - 1, j - 1] = -sp.I, sp.I
                gens.append(A / 2)
            l = k - 1
            D = sp.zeros(N, N)
            for i in range(l):
                D[i, i] = 1
            D[l, l] = -l
            gens.append(D * sp.sqrt(sp.Rational(2, l * (l + 1))) / 2)
        return cls(f"su({N})", gens)

    @classmethod
    def u1(cls):
        return cls("u(1)", [sp.Matrix([[1]])], f={})

    def adjoint_matrices(self):
        n = self.dim
        return [sp.Matrix(n, n, lambda b, c, a=a: -sp.I * self.f.get((a, b, c), 0)) for a in range(n)]

    def jacobi_holds(self):
        T = self.generators
        n = self.dim
        for a, b, c in itertools.product(range(n), repeat=3):
            lhs = sum(self.f.get((a, b, e), 0) * self.f.get((e, c, d), 0)
                      + self.f.get((b, c, e), 0) * self.f.get((e, a, d), 0)
                      + self.f.get((c, a, e), 0) * self.f.get((e, b, d), 0)
                      for e in range(n) for d in [0])
            if sp.simplify(lhs) != 0:
                return False
        return True

    def casimir_fundamental(self):
        return sp.simplify(sum((g * g for g in self.generators), sp.zeros(self.N, self.N)))

    def __repr__(self):
        return f"LieAlgebra({self.name}, dim={self.dim})"


# --------------------------------------------------------------------------- gauge theory
DEFAULT_STRENGTH = {"A": "F", "B": "H", "W": "V", "G": "S", "Z": "Y"}
DEFAULT_INDEX_NAMES = ["abcdef", "pqrs", "uvwx"]


@dataclass
class Factor:
    name: str
    algebra: LieAlgebra
    coupling: sp.Symbol
    kind: str
    strength: str
    gen_name: str
    f_name: str

    @property
    def abelian(self):
        return self.algebra.dim == 1


@dataclass
class Matter:
    name: str
    n: int
    reps: dict
    spinor: bool = False


def _act(R, v):
    """Matrix ``R`` acting on a multiplet value (scalar for one-component multiplets)."""
    if isinstance(v, sp.MatrixBase):
        return R * v
    return (R * sp.Matrix([[v]]))[0, 0]


class GaugeTheory:
    """Attach gauge groups and matter multiplets to a :class:`Model`."""

    def __init__(self, model):
        self.model = model
        self.factors: list[Factor] = []
        self.matter: dict[str, Matter] = {}
        model.operators["D"] = self.D
        model.spinors = set()
        model.gauge = self

    # ------------------------------------------------------------------ groups
    def add_group(self, name, algebra, coupling="g", strength=None, index_names=None,
                  gen_name=None, f_name=None):
        """Register gauge field ``name``, its strength tensor and the structure constants."""
        M = self.model
        gsym = M.symbols.get(coupling) if isinstance(coupling, str) else coupling
        if gsym is None:
            gsym = M.parameters(coupling, positive=True)
        strength = strength or DEFAULT_STRENGTH.get(name)
        if strength is None:
            raise ValueError("give an explicit single-letter `strength` name for this gauge field")
        n_prev = len(self.factors)
        kind = f"ad_{name}"
        gen_name = gen_name or ("T" if n_prev == 0 else f"S")
        f_name = f_name or ("f" if n_prev == 0 else "h")
        fac = Factor(name, algebra, gsym, kind, strength, gen_name, f_name)
        if fac.abelian:
            M.vector(name)
            M.define(rf"{strength}_{{\mu\nu}}", rf"\partial_\mu {name}_\nu - \partial_\nu {name}_\mu")
        else:
            names = index_names or DEFAULT_INDEX_NAMES[min(n_prev, 2)]
            M.add_index_space(kind, list(names), algebra.dim)
            rng = tuple(range(algebra.dim))
            M.field(name, (kind, "st"), "-d")
            M.tensors[f_name] = TensorDef(
                f_name, (kind,) * 3, ("-",) * 3, dict(algebra.f), None, "const", None, False,
                f_name, (rng,) * 3)
            M.tensors[gen_name] = TensorDef(
                gen_name, (kind,), ("-",), {(a,): algebra.generators[a] for a in rng},
                None, "const", None, False, gen_name, (rng,))
            a, b, c = names[:3]
            gname = gsym.name
            M.define(rf"{strength}^{a}_{{\mu\nu}}",
                     rf"\partial_\mu {name}^{a}_\nu - \partial_\nu {name}^{a}_\mu"
                     rf" + {gname} {f_name}^{{{a}{b}{c}}} {name}^{b}_\mu {name}^{c}_\nu")
        self.factors.append(fac)
        return fac

    def factor(self, ref):
        for fc in self.factors:
            if fc.name == ref or fc is ref:
                return fc
        raise KeyError(ref)

    def A(self, fac, a, mu):
        """Component function ``A^a_mu`` (``a`` is 0-based; ignored for abelian factors)."""
        td = self.model.tensors[fac.name]
        return td.funcs[(mu,)] if fac.abelian else td.funcs[(a, mu)]

    # ------------------------------------------------------------------ matter
    def add_matter(self, name, n, reps, complex=True, spinor=False, latex=None):
        """Multiplet of ``n`` components. ``reps``: ``{group: 'fund' | 'adj' | charge | [matrices]}``."""
        M = self.model
        col = M.multiplet(name, n, complex=complex, rep=(name, False), latex=latex)
        M.tensors[name].rep = (name, False)
        k = n
        if spinor:
            if n % 4:
                raise ValueError("spinor multiplets need 4k components")
            k = n // 4
            self._install_gamma(k)
            M.spinors.add(name)
        mats = {}
        for fname, r in reps.items():
            fac = self.factor(fname)
            if isinstance(r, str):
                if r == "fund":
                    ms = fac.algebra.generators
                elif r == "adj":
                    ms = fac.algebra.adjoint_matrices()
                elif r == "trivial":
                    ms = [sp.zeros(k, k)] * fac.algebra.dim
                else:
                    raise ValueError(f"unknown representation {r!r}")
            elif isinstance(r, (list, tuple)):
                ms = [sp.Matrix(m) for m in r]
            else:   # abelian charge
                ms = [sp.sympify(r) * sp.eye(k)]
            ms = [m if m.shape[0] == n else sp.kronecker_product(sp.eye(4), m) for m in ms]
            if ms[0].shape != (n, n):
                raise ValueError(f"representation matrices of {fname} do not fit {n} components")
            mats[fac.name] = ms
        self.matter[name] = Matter(name, n, mats, spinor)
        return col

    def _install_gamma(self, k):
        """Dirac matrices (Dirac representation), tensored with 1_k; sets gamma0 on the model."""
        M = self.model
        I2, Z2 = sp.eye(2), sp.zeros(2)
        sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]),
               sp.Matrix([[1, 0], [0, -1]])]

        def blk(a, b, c, d):
            return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))
        g = [blk(I2, Z2, Z2, -I2)] + [blk(Z2, s, -s, Z2) for s in sig]
        d = M.st.dim
        if d != 4:
            raise NotImplementedError("Dirac matrices are provided for 4D spacetime only")
        ik = sp.eye(k)
        gm = [sp.kronecker_product(m, ik) for m in g]
        if M.st.signature == "mostly_plus":
            gm = [sp.I * m for m in gm]   # {gamma_m, gamma_n} = -2 eta_mn convention shift
        rng = tuple(range(d))
        M.tensors["gamma"] = TensorDef("gamma", ("st",), ("u",), {(m,): gm[m] for m in rng},
                                       None, "const", None, False, r"\gamma", (rng,))
        M.gamma0 = gm[0]

    # ------------------------------------------------------------------ covariant derivative
    def D(self, operand, label, kind, var):
        """Gauge-covariant derivative ``D_label`` of a tensor/multiplet (operator ``D_\\mu``)."""
        M = operand.ctx                      # may be a perturbed-metric copy of the model
        if isinstance(operand.rep, tuple) and operand.rep[1]:
            return self.D(operand.dagger(), label, kind, var).dagger()
        st = M.st
        rng = range(1, st.dim) if kind == "sp" else range(st.dim)
        base = operand.partial(label, kind, "d")
        new = dict(base.data)
        for fac in self.factors:
            g = fac.coupling
            # --- multiplet part
            if isinstance(operand.rep, tuple) and operand.rep[0] in self.matter:
                mats = self.matter[operand.rep[0]].reps.get(fac.name)
                if mats is not None:
                    for key, v in operand.data.items():
                        for mu in rng:
                            acc = None
                            for a, R in enumerate(mats):
                                term = -sp.I * g * self.A(fac, a, mu) * _act(R, v)
                                acc = term if acc is None else acc + term
                            if acc is not None:
                                nk = (mu,) + key
                                new[nk] = new[nk] + acc if nk in new else acc
            # --- adjoint-index part
            if not fac.abelian:
                for p, s in enumerate(operand.slots):
                    if s.kind != fac.kind:
                        continue
                    for key, v in operand.data.items():
                        c = key[p]
                        for (a, b, cc), fv in fac.algebra.f.items():
                            if cc != c:
                                continue
                            for mu in rng:
                                nk = (mu,) + key[:p] + (a,) + key[p + 1:]
                                term = g * fv * self.A(fac, b, mu) * v
                                new[nk] = new[nk] + term if nk in new else term
        slot = Slot(label, kind, "st", "d", tuple(rng))
        t = Tensor(M, [slot] + operand.slots, new, operand.rep)
        return t.set_var(0, "u") if var == "u" else t

    # ------------------------------------------------------------------ gauge transformations
    def gauge_parameters(self, fac, name=None):
        """Register arbitrary gauge-parameter functions alpha^a(x) and return them as a list."""
        fac = self.factor(fac)
        base = name or f"alpha_{fac.name}"
        return [self.model._new_func(f"{base}[{a + 1}]", f"\\alpha^{{{a + 1}}}")
                for a in range(fac.algebra.dim)]

    def gauge_variation(self, params):
        """Infinitesimal gauge transformation as ``{function: variation}``.

        ``params``: ``{factor_name: [alpha^1(x), ...]}`` (functions or expressions).
        """
        M = self.model
        var = {}

        def add(f, v):
            var[f] = var.get(f, 0) + v
        for fname, alphas in params.items():
            fac = self.factor(fname)
            g, X = fac.coupling, M.coords
            al = list(alphas)
            # gauge fields of this factor and any adjoint-index field
            for tname, td in M.tensors.items():
                if td.kind != "field":
                    continue
                if tname == fac.name:
                    for key, f in td.funcs.items():
                        if fac.abelian:
                            add(f, sp.diff(al[0], X[key[0]]) / g)
                        else:
                            a, mu = key
                            add(f, sp.diff(al[a], X[mu]) / g)
                for p, k in enumerate(td.spaces):
                    if fac.abelian or k != fac.kind:
                        continue
                    for key, f in td.funcs.items():
                        a = key[p]
                        for (aa, b, c), fv in fac.algebra.f.items():
                            if aa == a:
                                kb = key[:p] + (b,) + key[p + 1:]
                                f_b = td.funcs.get(kb)
                                if f_b is not None:
                                    add(f, fv * f_b * al[c])
        # matter multiplets: delta Phi = i alpha^a R^a Phi
        for mname, mt in self.matter.items():
            col = M.tensors[mname].data[()]
            comps = list(col) if isinstance(col, sp.MatrixBase) else [col]
            col = sp.Matrix(comps)
            dv = sp.zeros(mt.n, 1)
            for fname, alphas in params.items():
                fac = self.factor(fname)
                mats = mt.reps.get(fac.name)
                if mats is None:
                    continue
                for a, R in enumerate(mats):
                    dv += sp.I * alphas[a] * (R * col)
            for f, d in zip(comps, dv):
                if d != 0:
                    add(f, d)
                    pf = M._conj_subs.get(f)
                    if pf is not None:
                        add(pf, M.conj(d))
        return var

    # ------------------------------------------------------------------ spectra
    def gauge_mass_matrix(self, L, vev, factor=None):
        """Mass-squared matrix of gauge bosons in a constant background.

        ``vev`` maps field functions (or multiplet components) to constant values. The matrix is
        ``M^2_{ij} = d^2 L / d A_i^0 d A_j^0`` over all gauge-field components (all factors).
        Returns ``(M2, labels)``.
        """
        M = self.model
        syms, labels, sub0 = [], [], {}
        for fac in self.factors:
            for a in range(fac.algebra.dim):
                f0 = self.A(fac, a, 0)
                s = sp.Dummy(f"A{fac.name}{a + 1}")
                sub0[f0] = s
                syms.append(s)
                labels.append(fac.name if fac.abelian else f"{fac.name}^{a + 1}")
                for mu in range(1, M.st.dim):
                    sub0[self.A(fac, a, mu)] = 0
        Lv = sp.sympify(L).subs({k: v for k, v in vev.items()})
        Lv = Lv.subs(sub0).doit()
        zero = {s: 0 for s in syms}
        # remaining field functions (fluctuations) set to zero
        for f in M.functions_in(Lv, dynamical_only=False):
            Lv = Lv.subs(f, 0)
        H = sp.Matrix(len(syms), len(syms), lambda i, j: sp.diff(Lv, syms[i], syms[j]).subs(zero))
        return H.applyfunc(sp.simplify), labels

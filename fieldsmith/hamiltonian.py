"""Canonical formalism: momenta, Hamiltonian density, Poisson brackets and Dirac constraint analysis.

Brackets are *local functional* brackets
``{F, G} = int sum_f ( dF/df dG/dpi_f - dF/dpi_f dG/df )`` between densities; functional derivatives
are taken with respect to spatial derivatives only. Smeared constraints (density times a test
function) are used to detect constraints that fail to commute.
"""
from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef


class Hamiltonian:
    def __init__(self, model, L):
        self.model = model
        if isinstance(L, str):
            L = model.lagrangian(L)
        self.L = sp.expand(sp.sympify(L))
        X = model.coords
        self.t, self.space = X[0], X[1:]
        self.fields = model.functions_in(self.L)
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self):
        L, t, model = self.L, self.t, self.model
        for d in L.atoms(sp.Derivative):
            if any(v == t for v, _ in d.variable_count) and sum(c for _, c in d.variable_count) > 1:
                raise NotImplementedError(
                    "higher time derivatives (Ostrogradsky) are not supported by the canonical module")
        self.V = {f: sp.Dummy(f"v_{f.func.__name__}") for f in self.fields}
        vel = {sp.Derivative(f, t): self.V[f] for f in self.fields}
        Ls = L.xreplace(vel)
        self.Ls = Ls
        p_expr = {f: sp.diff(Ls, self.V[f]) for f in self.fields}
        self.p_of = p_expr
        Vs = [self.V[f] for f in self.fields]
        W = sp.Matrix([[sp.diff(p_expr[f], v) for v in Vs] for f in self.fields])
        self.singular = [f for i, f in enumerate(self.fields) if all(W[i, j] == 0 for j in range(len(Vs)))]
        self.regular = [f for f in self.fields if f not in self.singular]
        self.pi = {}
        for f in self.fields:
            name = f"p[{f.func.__name__}]"
            self.pi[f] = sp.Function(name)(*model.coords)
            tex = model.latex_names.get(f.func.__name__, f.func.__name__)
            model.latex_names[name] = r"\pi_{%s}" % tex
        reg_idx = [self.fields.index(f) for f in self.regular]
        Wr = W.extract(reg_idx, reg_idx) if reg_idx else sp.zeros(0, 0)
        if reg_idx and sp.simplify(Wr.det()) == 0:
            raise NotImplementedError("Hessian is singular in a non-diagonal way")
        if any(W[i, j].free_symbols & set(Vs) for i in range(len(Vs)) for j in range(len(Vs))):
            raise NotImplementedError("Lagrangian is not quadratic in velocities (e.g. Born-Infeld)")
        sol = {}
        if reg_idx:
            eqs = [self.pi[f] - p_expr[f] for f in self.regular]
            res = sp.solve(eqs, [self.V[f] for f in self.regular], dict=True)
            sol = res[0]
        self.velocities = {f: sp.simplify(sol[self.V[f]]) for f in self.regular}
        zero_s = {self.V[f]: 0 for f in self.singular}
        Hc = sum(self.pi[f] * sol[self.V[f]] for f in self.regular) - Ls
        Hc = Hc.subs(zero_s).subs(sol)
        self.H = sp.expand(sp.simplify(Hc))
        self.primary = []
        for f in self.singular:
            c = sp.expand(self.pi[f] - p_expr[f].subs(zero_s))
            self.primary.append(c)

    # ------------------------------------------------------------------ calculus
    def fd(self, density, f):
        """Functional derivative of ``int density`` w.r.t. ``f`` (spatial integration by parts)."""
        density = sp.sympify(density)
        if density == 0 or f not in density.atoms(AppliedUndef):
            return sp.S.Zero
        ders = [d for d in density.atoms(sp.Derivative) if d.expr == f]
        dm = {d: sp.Dummy("D") for d in ders}
        rho = density.xreplace(dm)
        total = sp.diff(rho, f)
        for d, sym in dm.items():
            if any(v == self.t for v, _ in d.variable_count):
                continue
            c = sp.diff(rho, sym).xreplace({v: k for k, v in dm.items()})
            k = int(sum(n for _, n in d.variable_count))
            term = c
            for v, n in d.variable_count:
                for _ in range(int(n)):
                    term = sp.diff(term, v)
            total += (-1) ** k * term
        return sp.expand(total)

    def bracket(self, F, G):
        """Integrand of the Poisson bracket of two densities (modulo total spatial derivatives)."""
        tot = 0
        for f in self.fields:
            p = self.pi[f]
            tot += self.fd(F, f) * self.fd(G, p) - self.fd(F, p) * self.fd(G, f)
        return sp.expand(tot)

    def is_divergence(self, expr):
        """True if ``expr`` is a total spatial derivative (so its integral vanishes)."""
        expr = sp.expand(expr)
        if expr == 0:
            return True
        for f in expr.atoms(AppliedUndef):
            if sp.simplify(self.fd(expr, f)) != 0:
                return False
        return True

    def hamilton_equations(self):
        """``{f: dot f, pi_f: dot pi_f}`` from the canonical Hamiltonian (singular fields omitted)."""
        out = {}
        for f in self.regular:
            out[sp.Derivative(f, self.t)] = sp.simplify(self.fd(self.H, self.pi[f]))
            out[sp.Derivative(self.pi[f], self.t)] = sp.simplify(-self.fd(self.H, f))
        return out

    # ------------------------------------------------------------------ weak equality
    def weak(self, expr):
        """Impose the primary constraints (``pi_f -> p_f(q)`` for singular fields)."""
        sub = {}
        for f, c in zip(self.singular, self.primary):
            sub[self.pi[f]] = sp.solve(c, self.pi[f])[0]
        out = sp.sympify(expr).subs(sub).doit()
        return sp.expand(out)

    def _weak_pivot(self, expr, constraints):
        """Further reduce ``expr`` by solving each constraint for a derivative of a momentum."""
        for c in constraints:
            c = self.weak(c)
            cands = [a for a in c.atoms(sp.Derivative) if a.expr in self.pi.values()]
            if not cands:
                continue
            pivot = sorted(cands, key=lambda a: str(a))[0]
            sol = sp.solve(c, pivot)
            if sol:
                expr = sp.expand(expr.subs(pivot, sol[0]).doit())
        return expr

    # ------------------------------------------------------------------ Dirac analysis
    def dirac_analysis(self, max_rounds=3):
        """Dirac-Bergmann algorithm for the constraints generated by ``L``.

        Returns a :class:`DiracResult` with primary/secondary constraints, first/second-class
        labels and the physical degrees of freedom per spatial point.
        """
        xi = sp.Function("xi")(*self.model.coords)
        eta = sp.Function("eta")(*self.model.coords)
        constraints = list(self.primary)
        labels = ["primary"] * len(constraints)
        frontier = list(range(len(constraints)))
        for _ in range(max_rounds):
            newfront = []
            for a in frontier:
                ca = constraints[a]
                G = sp.simplify(self.weak(self.fd(self.bracket(xi * ca, self.H), xi)))
                # does the new condition fix a multiplier (non-commuting with a primary)?
                fixes = False
                for b, cb in enumerate(constraints):
                    if labels[b] != "primary":
                        continue
                    br = self.weak(self.bracket(xi * ca, eta * cb))
                    if not self.is_divergence(br):
                        fixes = True
                        break
                if fixes or G == 0:
                    continue
                G = self._weak_pivot(G, [c for c in constraints if c is not ca])
                if G == 0 or any(sp.simplify(G - c) == 0 or sp.simplify(G + c) == 0 for c in constraints):
                    continue
                constraints.append(sp.expand(G))
                labels.append("secondary" if labels[a] == "primary" else "tertiary")
                newfront.append(len(constraints) - 1)
            if not newfront:
                break
            frontier = newfront
        n = len(constraints)
        nonzero = [[False] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                br = self.weak(self.bracket(xi * constraints[a], eta * constraints[b]))
                if not self.is_divergence(br):
                    br = self._weak_pivot(br, constraints)
                    nonzero[a][b] = not self.is_divergence(br)
        second = [any(nonzero[a][b] for b in range(n)) for a in range(n)]
        return DiracResult(self, constraints, labels, second, nonzero)


class DiracResult:
    def __init__(self, ham, constraints, labels, second_class, nonzero):
        self.ham, self.constraints, self.labels = ham, constraints, labels
        self.second_class, self.bracket_nonzero = second_class, nonzero
        self.n_first = sum(1 for s in second_class if not s)
        self.n_second = sum(1 for s in second_class if s)
        n_fields = len(ham.fields)
        self.dof = sp.Rational(2 * n_fields - 2 * self.n_first - self.n_second, 2)

    def summary(self):
        lines = [f"canonical pairs: {len(self.ham.fields)}",
                 f"first-class constraints: {self.n_first}, second-class: {self.n_second}",
                 f"physical degrees of freedom per point: {self.dof}"]
        for c, lab, s in zip(self.constraints, self.labels, self.second_class):
            lines.append(f"  [{lab}, {'second' if s else 'first'}-class]  {c} ~ 0")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

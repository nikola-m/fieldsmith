"""16 - The BPST instanton of Euclidean SU(2) Yang-Mills               (level: research-flavoured)

Topics: Euclidean spacetime, 't Hooft symbols, verifying an exact non-perturbative solution:
        (anti-)self-duality, the Yang-Mills equations, Bogomolny bound S = 8 pi^2/g^2, topological
        charge Q = 1.
"""
from _util import *
import sympy as sp
from fieldsmith import Model, Spacetime
from fieldsmith.gauge import GaugeTheory, LieAlgebra

M = Model(Spacetime.euclidean(4))
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
g = M.symbols["g"]
rho = M.parameters("rho", positive=True)
X = M.coords
r2 = sum(c**2 for c in X)
Af = M.tensors["A"].funcs

def eta(a, mu, nu):                  # 't Hooft symbol eta^a_{mu nu}  (a = 0,1,2; mu,nu = 0..3)
    if mu < 3 and nu < 3:
        return sp.LeviCivita(a, mu, nu)
    if mu < 3 and nu == 3:
        return 1 if a == mu else 0
    if mu == 3 and nu < 3:
        return -1 if a == nu else 0
    return 0

ansatz = {Af[(a, mu)]: 2 * sum(eta(a, mu, nu) * X[nu] for nu in range(4)) / (g * (r2 + rho**2))
          for a in range(3) for mu in range(4)}

section("Field strength of the ansatz and self-duality")
M.define(r"\tilde{F}^a_{\mu\nu}", r"\frac12\epsilon_{\mu\nu\rho\sigma}F^a_{\rho\sigma}")
Fa = M.tensors["F"].data
Fdual = M.tensors["Ftilde"].data
diff_plus = {k: sp.simplify((Fa[k] - Fdual[k]).subs(ansatz).doit()) for k in Fa}
diff_minus = {k: sp.simplify((Fa[k] + Fdual[k]).subs(ansatz).doit()) for k in Fa}
selfdual = all(v == 0 for v in diff_plus.values())
antiselfdual = all(v == 0 for v in diff_minus.values())
ok("F = +Ftilde (self-dual)  or  F = -Ftilde (anti-self-dual)", selfdual or antiselfdual)
show("duality", "self-dual" if selfdual else "anti-self-dual" if antiselfdual else "neither")
F312 = sp.factor(sp.simplify(Fa[(2, 0, 1)].subs(ansatz).doit()))
show("F^3_{12}  (expect -4 rho^2 / (g (x^2+rho^2)^2))", F312)

section("The Yang-Mills equations are satisfied")
L = M.lagrangian(r"\frac14 F^a_{\mu\nu}F^a_{\mu\nu}")           # Euclidean Lagrangian density
E = M.euler_lagrange(L)
ok("delta S/delta A = 0 for the BPST ansatz (exact, symbolic)", E.check_solution(ansatz))

section("Action and topological charge")
dens = sp.factor(sp.together(sp.simplify(L.subs(ansatz).doit())))
show("action density  F^2/4  (expect 48 rho^4 / (g^2 (x^2+rho^2)^4))", dens)
s = sp.Symbol("s", positive=True)
radial = sp.simplify(dens.subs({X[0]: s, X[1]: 0, X[2]: 0, X[3]: 0}))
S = sp.integrate(2 * sp.pi**2 * s**3 * radial, (s, 0, sp.oo))
show("S = 2 pi^2 int s^3 dens ds", sp.simplify(S))
ok("S = 8 pi^2 / g^2  (Bogomolny bound saturated)", sp.simplify(S - 8 * sp.pi**2 / g**2) == 0)
q_dens = sp.simplify((g**2 / (32 * sp.pi**2) * M.parse(r"F^a_{\mu\nu}\tilde F^a_{\mu\nu}")).subs(ansatz).doit())
Q = sp.integrate(2 * sp.pi**2 * s**3 * sp.simplify(q_dens.subs({X[0]: s, X[1]: 0, X[2]: 0, X[3]: 0})), (s, 0, sp.oo))
show("Q = g^2/(32 pi^2) int F Ftilde", sp.simplify(Q))
ok("|Q| = 1", abs(sp.simplify(Q)) == 1)

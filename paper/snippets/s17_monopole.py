from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
M.parameters("v", positive=True)
M.field("phi", ("ad_A",), "-")                       # adjoint Higgs triplet phi^a
L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu} + \frac12 D_\mu\phi^a D^\mu\phi^a")
E = M.euler_lagrange(L)
M.define(r"B^a_i", r"-\frac12\epsilon_{ijk}F^a_{jk}")
import sympy as sp
from fieldsmith.analysis import numeric_residual
g = M.symbols["g"]; v = M.symbols["v"]
t, x, y, z = M.coords; X = [x, y, z]
r = sp.sqrt(x**2 + y**2 + z**2); u = g*v*r
K = u/sp.sinh(u); H = u/sp.tanh(u) - 1                       # Prasad-Sommerfield profiles
Af, pf = M.tensors["A"].funcs, M.tensors["phi"].funcs
sol = {}
for a in range(3):
    sol[pf[(a,)]] = X[a]*H/(g*r**2)
    sol[Af[(a, 0)]] = 0
    for i in range(3):
        sol[Af[(a, i+1)]] = sum(sp.LeviCivita(a, i, j)*X[j] for j in range(3))*(1 - K)/(g*r**2)
print(numeric_residual(M, E.exprs, sol, n_points=2, param_values={g: sp.Rational(7, 10), v: sp.Rational(13, 10)}))

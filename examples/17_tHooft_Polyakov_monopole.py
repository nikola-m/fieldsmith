"""17 - The 't Hooft-Polyakov magnetic monopole (BPS limit)             (level: research-flavoured)

Topics: SU(2) gauge theory with an adjoint Higgs field, hedgehog ansatz, verification of the full
        field equations and of the Bogomolny equations B^a_i = +/- D_i phi^a for the
        Prasad-Sommerfield solution, and the monopole mass M = 4 pi v / g.
"""
from _util import *
import itertools
import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import numeric_residual

M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
g = M.symbols["g"]
v = M.parameters("v", positive=True)
M.field("phi", ("ad_A",), "-")                # adjoint Higgs triplet phi^a
L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu} + \frac12 D_\mu\phi^a D^\mu\phi^a")   # BPS limit: lambda -> 0
E = M.euler_lagrange(L)

t, x, y, z = M.coords
X = [x, y, z]
r = sp.sqrt(x**2 + y**2 + z**2)
u = g * v * r
K = u / sp.sinh(u)                           # profile functions of the Prasad-Sommerfield solution
H = u / sp.tanh(u) - 1
Af, pf = M.tensors["A"].funcs, M.tensors["phi"].funcs

def config(s):
    m = {}
    for a in range(3):
        m[pf[(a,)]] = X[a] * H / (g * r**2)
        m[Af[(a, 0)]] = 0
        for i in range(3):
            m[Af[(a, i + 1)]] = s * sum(sp.LeviCivita(a, i, j) * X[j] for j in range(3)) * (1 - K) / (g * r**2)
    return m

section("Which sign of the hedgehog gauge field solves the equations?")
pars = {g: sp.Rational(7, 10), v: sp.Rational(13, 10)}
found = None
for s in (+1, -1):
    res = numeric_residual(M, E.exprs, config(s), n_points=2, param_values=pars)
    show(f"max residual of the field equations, s = {s:+d}", f"{res:.2e}")
    if res < 1e-9:
        found = s
ok("an exact monopole solution of the full second-order equations exists", found is not None)
mapping = config(found)

section("Bogomolny equations")
M.define(r"B^a_i", r"-\frac12\epsilon_{ijk}F^a_{jk}")
res = {}
for s2 in (+1, -1):
    d = M.parse(r"B^a_i - %d D_i\phi^a" % s2, as_tensor=True)
    worst = max(numeric_residual(M, [c], mapping, n_points=2, param_values=pars) for c in d.data.values())
    res[s2] = worst
    show(f"max |B^a_i - ({s2:+d}) D_i phi^a|", f"{worst:.2e}")
ok("first-order BPS equations hold (B = +/- D phi)", min(res.values()) < 1e-9)

section("Monopole mass")
eps = M.parse(r"\frac14 F^a_{ij}F^a_{ij} + \frac12 D_i\phi^a D_i\phi^a")
rr = sp.Symbol("rr", positive=True)
rho_axis = eps.subs(mapping).doit().subs({y: 0, z: 0}).subs(x, rr)       # energy density on the x-axis (spherical symmetry)
import mpmath as mp
f = sp.lambdify((rr,), (4 * sp.pi * rr**2 * rho_axis).subs({g: 1, v: 1}), "mpmath")
mass = mp.quad(f, [0, 1, 5, 20, 60, mp.inf])
show("M (numerical, g = v = 1)", mp.nstr(mass, 10))
show("4 pi v / g", mp.nstr(4 * mp.pi, 10))
ok("M = 4 pi v / g", abs(mass - 4 * mp.pi) < 1e-6)

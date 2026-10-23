"""14 - Higher-derivative electrodynamics: Podolsky's theory           (level: research-flavoured)

Topics: Euler-Lagrange equations with second derivatives, covariant form (1 + a^2 Box) d_mu F^{mu nu} = 0,
        spectrum (massless photon + massive Podolsky photon), finite self-energy Coulomb potential,
        gauge invariance.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import symmetry_defect
from fieldsmith.waves import linearize, dispersion

M = Model()
a = M.parameters("a", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"j_\nu", r"\partial^\mu F_{\mu\nu}")          # j_nu = d^mu F_{mu nu}  (second derivatives of A)
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{a^2}{2} j_\nu j^\nu")

section("Gauge invariance")
alpha = sp.Function("alpha")(*M.coords)
var = {M.tensors["A"].funcs[(i,)]: sp.diff(alpha, M.coords[i]) for i in range(4)}
show("delta L", symmetry_defect(M, L, var))

section("Field equations (fourth order in derivatives)")
E = M.euler_lagrange(L)
cov = fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "box": r"\Box\partial_\mu F^{\mu\nu}"})
show("dS/dA_nu = c1 d_mu F^{mu nu} + c2 Box d_mu F^{mu nu}", cov)
print("   i.e.  (1 + a^2 Box) d_mu F^{mu nu} = 0   (Podolsky)")

section("Spectrum")
lin, pert = linearize(M, E)
res = dispersion(M, lin, list(pert.values()))
show("physical dispersion factor", res["pdet"])
show("omega(k)", res["omega"])
print("   two massless modes + three modes of mass 1/a  => 5 degrees of freedom")

section("Static point charge: finite self-energy")
q = sp.Symbol("q", positive=True)
x, y, z = M.coords[1:]
r = sp.sqrt(x**2 + y**2 + z**2)
A0 = q * (1 - sp.exp(-r / a)) / (4 * sp.pi * r)
A = M.tensors["A"].funcs
ok("A_0 = q (1 - e^{-r/a})/(4 pi r) solves the field equations for r > 0  (Coulomb for r >> a)",
   E.check_solution({A[(0,)]: A0, A[(1,)]: 0, A[(2,)]: 0, A[(3,)]: 0}, numeric=True))
rr = sp.Symbol("r", positive=True)
show("limit r -> 0 (finite, unlike 1/r)", sp.limit(q * (1 - sp.exp(-rr / a)) / (4 * sp.pi * rr), rr, 0))

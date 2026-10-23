"""04 - Massive vector field: Proca and Stueckelberg                (level: intermediate)

Topics: Lorenz constraint from the field equations, three polarisations, canonical constraint
        structure (second class), restoring gauge invariance with a Stueckelberg field.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import symmetry_defect
from fieldsmith.waves import linearize, dispersion
from fieldsmith.hamiltonian import Hamiltonian

M = Model()
m = M.parameters("m", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu")

section("Field equation  d_mu F^{mu nu} + m^2 A^nu = 0")
E = M.euler_lagrange(L)
show("EL = c1 d_mu F^{mu nu} + c2 m^2 A^nu", fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "mA": r"m^2 A^\nu"}))
Et = E.tensor("A")
div = sum(sp.diff(Et[(n,)], M.coords[n]) for n in range(4))
ok("d_nu E^nu  ~  m^2 d_nu A^nu : the Lorenz condition is implied", sp.simplify(
    div - M.parse(r"m^2\partial_\nu A^\nu")) == 0)

section("Spectrum")
lin, pert = linearize(M, E)
res = dispersion(M, lin, list(pert.values()))
show("det K", res["det"])
show("omega", res["omega"])
print("(the cubic power of the mass-shell factor = three polarisations)")

section("Canonical structure")
print(Hamiltonian(M, L).dirac_analysis())

section("Stueckelberg trick: gauge invariance with an extra scalar theta")
M2 = Model()
m2 = M2.parameters("m", positive=True)
M2.vector("A")
M2.scalar("theta")
M2.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
Ls = M2.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 (m A_\mu - \partial_\mu\theta)(m A^\mu - \partial^\mu\theta)")
alpha = sp.Function("alpha")(*M2.coords)
var = {M2.tensors["A"].funcs[(i,)]: sp.diff(alpha, M2.coords[i]) for i in range(4)}
var[M2.tensors["theta"].funcs[()]] = m2 * alpha
show("delta L under (delta A = d alpha, delta theta = m alpha)", symmetry_defect(M2, Ls, var))
print(Hamiltonian(M2, Ls).dirac_analysis())

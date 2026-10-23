"""15 - Topologically massive electrodynamics in 2+1 dimensions        (level: research-flavoured)

Topics: a three-dimensional spacetime, the Chern-Simons term (gauge invariant only up to a
        boundary term -> symmetry_defect with K^mu), parity-odd mass for a single polarisation,
        canonical analysis (one degree of freedom).
"""
from _util import *
import sympy as sp
from fieldsmith import Model, Spacetime
from fieldsmith.analysis import fit
from fieldsmith.noether import symmetry_defect
from fieldsmith.waves import linearize, dispersion
from fieldsmith.hamiltonian import Hamiltonian

M = Model(Spacetime.minkowski(3))
mu = M.parameters("mu", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
LCS = M.lagrangian(r"\frac{\mu}{4}\epsilon^{\mu\nu\rho}A_\mu F_{\nu\rho}")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{\mu}{4}\epsilon^{\mu\nu\rho}A_\mu F_{\nu\rho}")

section("Gauge variation of the Chern-Simons term is a total derivative")
alpha = sp.Function("alpha")(*M.coords)
A = M.tensors["A"].funcs
var = {A[(i,)]: sp.diff(alpha, M.coords[i]) for i in range(3)}
show("delta L_CS (not zero ...)", symmetry_defect(M, LCS, var))
# delta L_CS = (mu/2) eps^{mu nu rho} d_mu alpha F_{nu rho}/2 = d_mu K^mu with K^mu = (mu/4) eps^{mu nu rho} alpha F_{nu rho}
Kt = M.parse(r"\frac{\mu}{4}\epsilon^{\mu\nu\rho}F_{\nu\rho}", as_tensor=True)
K = [alpha * Kt[(i,)] for i in range(3)]
show("delta L_CS - d_mu K^mu", symmetry_defect(M, LCS, var, K))
print("   => gauge invariant up to a boundary term (large gauge transformations quantise mu)")

section("Equations of motion")
E = M.euler_lagrange(L)
show("dS/dA_nu = c1 d_mu F^{mu nu} + c2 eps^{nu rho sigma} F_{rho sigma}", fit(
    M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "cs": r"\epsilon^{\nu\rho\sigma}F_{\rho\sigma}"}))

section("Spectrum: one massive mode, mass |mu|")
lin, pert = linearize(M, E)
res = dispersion(M, lin, list(pert.values()))
show("physical dispersion factor", res["pdet"])
show("omega(k)", res["omega"])

section("Canonical analysis")
print(Hamiltonian(M, L).dirac_analysis())

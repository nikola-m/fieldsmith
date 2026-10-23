"""03 - Maxwell electrodynamics from a LaTeX Lagrangian               (level: intermediate)

Topics: covariant field equations, Bianchi identity, gauge invariance and Noether's second
        theorem, Hilbert vs canonical stress tensor, E^2+B^2 energy density, photon polarisations,
        Dirac constraint analysis (2 physical degrees of freedom).
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import hilbert_stress_tensor, canonical_stress_tensor, symmetry_defect
from fieldsmith.waves import linearize, dispersion
from fieldsmith.hamiltonian import Hamiltonian

M = Model()
M.vector("A")                       # A_mu, lower index
M.vector("J", var="u", dynamical=False)   # external current J^mu (not varied)
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
Ls = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} - J^\mu A_\mu")      # with a source

section("Field equations in covariant form")
E = M.euler_lagrange(L)
cov = fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}"})
show("delta S/delta A_nu = c * d_mu F^{mu nu},  c", cov)
src = fit(M, M.euler_lagrange(Ls).tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "J": r"J^\nu"})
show("with a source: delta S/delta A_nu = c1 d_mu F^{mu nu} + c2 J^nu", src)

section("Identities written in LaTeX are verified symbolically")
ok("Bianchi identity", M.check(r"\partial_\mu F_{\nu\rho} + \partial_\nu F_{\rho\mu} + \partial_\rho F_{\mu\nu} = 0"))
ok("dual form  d_mu tilde F^{mu nu} = 0",
   M.check(r"\partial_\mu (\frac12 \epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}) = 0"))
ok("d_nu d_mu F^{mu nu} = 0 (current conservation)", M.check(r"\partial_\nu\partial_\mu F^{\mu\nu} = 0"))

section("Gauge invariance (local U(1): delta A_mu = d_mu alpha)")
alpha = sp.Function("alpha")(*M.coords)
var = {M.tensors["A"].funcs[(m_,)]: sp.diff(alpha, M.coords[m_]) for m_ in range(4)}
show("delta L", symmetry_defect(M, L, var))

section("Stress tensors")
Th = hilbert_stress_tensor(M, r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
Tu = Th.raise_("mu").raise_("nu")
rho = sp.simplify(Tu[0, 0])
Ex, Ey, Ez = [M.parse(r"F_{%d0}" % i) for i in (1, 2, 3)]
Bx, By, Bz = M.parse(r"F_{23}"), M.parse(r"F_{31}"), M.parse(r"F_{12}")
ok("T^{00} = (E^2 + B^2)/2", sp.simplify(rho - (Ex**2 + Ey**2 + Ez**2 + Bx**2 + By**2 + Bz**2) / 2) == 0)
trace = sum(M.st.ginv[i, i] * Th[i, i] for i in range(4))
ok("traceless", sp.simplify(trace) == 0)
ok("symmetric", Th.is_symmetric("mu", "nu"))
Tc = canonical_stress_tensor(M, L)
ok("canonical T^0_0 differs from the gauge-invariant Hilbert one",
   sp.simplify(Tc[0, 0] - Th[0, 0]) != 0)

section("Linearised waves and the gauge direction")
lin, pert = linearize(M, E)
res = dispersion(M, lin, list(pert.values()))
show("rank of the 4x4 wave matrix K (3 => one null direction a ~ k: pure gauge)", res["rank"])

section("Canonical analysis")
print(Hamiltonian(M, L).dirac_analysis())

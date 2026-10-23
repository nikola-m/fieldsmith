"""09 - Quantum electrodynamics, classical Dirac field                 (level: graduate)

Topics: Dirac spinors with gamma matrices from LaTeX, Dirac equation from the action, Maxwell
        equations with the Dirac current, Noether current, squaring the Dirac operator,
        canonical structure (8 second-class constraints => 4 degrees of freedom).
(The spinor is treated as a commuting classical field; Grassmann statistics is a quantum concern.)
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import fit
from fieldsmith.noether import noether_current, complex_variation, symmetry_defect
from fieldsmith.hamiltonian import Hamiltonian

M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.u1(), "e")
m = M.parameters("m", positive=True)
G.add_matter("psi", 4, {"A": 1}, complex=True, spinor=True, latex=r"\psi")
L = M.lagrangian(r"\bar\psi (i\gamma^\mu D_\mu - m)\psi - \frac14 F_{\mu\nu}F^{\mu\nu}")

section("Gauge invariance of QED  (psi -> e^{i alpha} psi, A -> A + d alpha / e)")
(alpha,) = G.gauge_parameters("A")
show("delta L", symmetry_defect(M, L, G.gauge_variation({"A": [alpha]})))

section("Dirac equation from the action")
E = M.euler_lagrange(L)
dirac = M.parse(r"(i\gamma^\mu D_\mu - m)\psi", as_tensor=True).value()
psibar = [M._conj_subs[f] for f in M.tensors["psi"].data[()]]
# delta S / delta psibar_k  with psibar = psi^dagger gamma^0 :  use psi^dagger and multiply by gamma^0 (inverse)
g0 = M.gamma0
resid = sp.Matrix([E[f] for f in psibar])
show("delta S/delta psi^dagger  =  gamma^0 (i gamma^mu D_mu - m) psi ?",
     sp.simplify(resid - g0 * dirac) == sp.zeros(4, 1))

section("Maxwell's equations sourced by the Dirac current")
Et = E.tensor("A")
show("delta S/delta A_nu = c1 d_mu F^{mu nu} + c2 psibar gamma^nu psi", fit(
    M, Et, {"dF": r"\partial_\mu F^{\mu\nu}", "j": r"e \bar\psi\gamma^\nu\psi"}))

section("Noether current of the global phase rotation")
nc = noether_current(M, L, complex_variation(M, "psi", r"i\psi"))
jmu = M.parse(r"\bar\psi\gamma^\mu\psi", as_tensor=True)
ok("J^mu = -psibar gamma^mu psi", all(sp.simplify(nc.J[k] + jmu[(k,)]) == 0 for k in range(4)))
ok("identically conserved on shell (Noether identity)", nc.identity_holds())

section("Squaring the free Dirac operator gives Klein-Gordon")
M0 = Model()
G0 = GaugeTheory(M0)
G0.add_group("A", LieAlgebra.u1(), "e")
M0.parameters("m", positive=True)
G0.add_matter("psi", 4, {"A": 0}, complex=True, spinor=True)
sq = M0.parse(r"(i\gamma^\nu\partial_\nu + m)((i\gamma^\mu\partial_\mu - m)\psi)", as_tensor=True).value()
kg = M0.parse(r"-(\Box + m^2)\psi", as_tensor=True).value()
ok("(i gamma.d + m)(i gamma.d - m) psi = -(Box + m^2) psi", sp.simplify(sq - kg) == sp.zeros(4, 1))

section("Canonical structure of the free Dirac field")
Lfree = M0.lagrangian(r"\bar\psi (i\gamma^\mu\partial_\mu - m)\psi")
res = Hamiltonian(M0, Lfree).dirac_analysis()
show("second-class constraints", res.n_second)
show("physical degrees of freedom per point (2 spins x particle/antiparticle)", res.dof)

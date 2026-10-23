"""05 - Scalar QED and the abelian Higgs mechanism                  (level: intermediate)

Topics: minimal coupling, Noether current = gauge current, spontaneous symmetry breaking,
        unitary gauge, mass spectrum, counting degrees of freedom before and after.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import fit
from fieldsmith.noether import noether_current, complex_variation, symmetry_defect
from fieldsmith.waves import linearize, dispersion
from fieldsmith.hamiltonian import Hamiltonian

M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.u1(), "e")
mu, lam = M.parameters("mu lambda", positive=True)
G.add_matter("Phi", 1, {"A": 1}, complex=True, latex=r"\Phi")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + (D_\mu\Phi)^\dagger (D^\mu\Phi) + \mu^2\Phi^\dagger\Phi - \lambda(\Phi^\dagger\Phi)^2")

section("Minimal coupling: gauge invariance of the whole Lagrangian")
(alpha,) = G.gauge_parameters("A")
show("delta L under Phi -> e^{i alpha} Phi, A -> A + d alpha / e", symmetry_defect(M, L, G.gauge_variation({"A": [alpha]})))

section("The Noether current of the *global* U(1) is the electromagnetic current")
nc = noether_current(M, L, complex_variation(M, "Phi", r"i\Phi"))
show("J^0 (charge density)", sp.simplify(nc.J[0]))
ok("global current conservation identity", nc.identity_holds())

section("Maxwell's equations with the scalar current")
E = M.euler_lagrange(L)
Et = E.tensor("A")
show("EL for A_nu = d_mu F^{mu nu} + e j^nu ?", fit(M, Et, {"dF": r"\partial_\mu F^{\mu\nu}",
     "j": r"i e (\Phi^\dagger D^\nu\Phi - (D^\nu\Phi)^\dagger\Phi)"}))

section("Spontaneous symmetry breaking  (mu^2 > 0 convention: V = -mu^2|Phi|^2 + lam|Phi|^4)")
Phi = M.tensors["Phi"].funcs[()]; Phib = M._conj_subs[Phi]
V = M.parse(r"-\mu^2\Phi^\dagger\Phi + \lambda(\Phi^\dagger\Phi)^2")
v = sp.symbols("v", positive=True)
vac = sp.solve(sp.diff(V.subs({Phi: v / sp.sqrt(2), Phib: v / sp.sqrt(2)}), v), v)
show("extrema of V along the real direction: v =", vac)

# shift Phi = (v + h + i G)/sqrt2  with v = mu/sqrt(lambda)  (here V = -mu^2|Phi|^2 + lam|Phi|^4)
v0 = mu / sp.sqrt(lam)
h = M.scalar("h")
Gold = M.scalar("G")
sub = {Phi: (v0 + h + sp.I * Gold) / sp.sqrt(2), Phib: (v0 + h - sp.I * Gold) / sp.sqrt(2)}
Lsh = sp.expand(L.subs(sub).doit())
show("coefficient of A_x d_x G in L (Goldstone-gauge mixing, expect e (v + h)):",
     sp.simplify(Lsh.coeff(M.tensors["A"].funcs[(1,)]).coeff(sp.Derivative(Gold, M.coords[1]))))

section("Unitary gauge (G = 0): massive vector + massive Higgs")
Lu = sp.expand(Lsh.subs(Gold, 0).doit())
Eu = M.euler_lagrange(Lu)
lin, pert = linearize(M, Eu)
res = dispersion(M, lin, list(pert.values()))
show("det K", res["det"])
masses = sorted(set(sp.simplify(w**2 - sum(k**2 for k in res["wave"].k)) for w in res["omega"]), key=str)
show("(omega^2 - k^2) values -> masses squared", masses)
print("expected: M_A = e v = e mu/sqrt(lam),  M_h^2 = 2 mu^2")

section("Degrees of freedom: 2 (photon) + 2 (complex scalar) -> 3 (massive vector) + 1 (Higgs)")
print("before:", Hamiltonian(M, L).dirac_analysis().dof)
print("after (unitary gauge):", Hamiltonian(M, Lu).dirac_analysis().dof)

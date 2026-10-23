"""08 - The electroweak Higgs mechanism, SU(2)_L x U(1)_Y             (level: graduate)

Topics: product gauge group, doublet with hypercharge 1/2, the full Higgs-sector Lagrangian is
        gauge invariant, gauge-boson mass matrix, W and Z masses, massless photon, Weinberg angle.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.noether import symmetry_defect

M = Model()
G = GaugeTheory(M)
G.add_group("W", LieAlgebra.su(2), "g", strength="V")      # W^a_mu, field strength V^a_{mu nu}
G.add_group("B", LieAlgebra.u1(), "g'", strength="H")      # B_mu,   field strength H_{mu nu}
v, lam, mu = M.parameters("v lambda mu", positive=True)
G.add_matter("Phi", 2, {"W": "fund", "B": sp.Rational(1, 2)}, complex=True, latex=r"\Phi")

L = M.lagrangian(r"-\frac14 V^a_{\mu\nu}V^{a\mu\nu} - \frac14 H_{\mu\nu}H^{\mu\nu}"
                 r" + (D_\mu\Phi)^\dagger (D^\mu\Phi) + \mu^2\Phi^\dagger\Phi - \lambda(\Phi^\dagger\Phi)^2")

section("Gauge invariance of the complete Higgs-sector Lagrangian")
ws = G.gauge_parameters("W")
(b,) = G.gauge_parameters("B")
show("delta L under SU(2) x U(1) (4 independent local parameters)",
     symmetry_defect(M, L, G.gauge_variation({"W": ws, "B": [b]})))

section("Vacuum  <Phi> = (0, v/sqrt2)")
Phi = M.tensors["Phi"].data[()]
vev = {}
for f, val in zip(list(Phi), (0, v / sp.sqrt(2))):
    vev[f] = val
    vev[M._conj_subs[f]] = val
Lk = M.lagrangian(r"(D_\mu\Phi)^\dagger (D^\mu\Phi)")
M2, labels = G.gauge_mass_matrix(Lk, vev)
show("basis", labels)
sp.pprint(M2)
ev = M2.eigenvals()
show("eigenvalues (mass^2 : multiplicity)", ev)
g, gp = M.symbols["g"], M.symbols["g'"]
mW2, mZ2 = g**2 * v**2 / 4, (g**2 + gp**2) * v**2 / 4
ok("two W bosons with M_W^2 = g^2 v^2 / 4", ev.get(sp.simplify(mW2)) == 2)
ok("one Z boson with  M_Z^2 = (g^2 + g'^2) v^2 / 4", any(sp.simplify(k - mZ2) == 0 for k in ev))
ok("one massless photon", ev.get(0) == 1)

section("Weinberg angle")
thW = sp.Symbol("theta_W", positive=True)
show("cos(theta_W) = M_W / M_Z =", sp.simplify(sp.sqrt(mW2 / mZ2)))
show("photon direction (null vector of the mass matrix):", M2.nullspace()[0].T)
print("  i.e. the photon is  A ~ g' W^3 + g B  =  sin(theta_W) W^3 + cos(theta_W) B  (up to normalisation)")
rho = sp.simplify(mW2 / (mZ2 * sp.cos(thW) ** 2)).subs(sp.cos(thW), g / sp.sqrt(g**2 + gp**2))
show("rho = M_W^2 / (M_Z^2 cos^2 theta_W)", sp.simplify(rho))

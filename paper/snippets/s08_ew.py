import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra

M = Model()
G = GaugeTheory(M)
G.add_group("W", LieAlgebra.su(2), "g", strength="V")
G.add_group("B", LieAlgebra.u1(), "g'", strength="H")
G.add_matter("Phi", 2, {"W": "fund", "B": sp.Rational(1, 2)}, complex=True, latex=r"\Phi")
L = M.lagrangian(r"-\frac14 V^a_{\mu\nu}V^{a\mu\nu} - \frac14 H_{\mu\nu}H^{\mu\nu}"
                 r" + (D_\mu\Phi)^\dagger (D^\mu\Phi) + \mu^2\Phi^\dagger\Phi - \lambda(\Phi^\dagger\Phi)^2")
v = M.parameters("v", positive=True)
Phi = M.tensors["Phi"].data[()]
vev = {}
for f, val in zip(list(Phi), (0, v / sp.sqrt(2))):          # <Phi> = (0, v/sqrt2)
    vev[f] = val
    vev[M._conj_subs[f]] = val
M2, labels = G.gauge_mass_matrix(M.lagrangian(r"(D_\mu\Phi)^\dagger (D^\mu\Phi)"), vev)
print(labels, M2.eigenvals())        # {g^2 v^2/4: 2, (g^2+g'^2) v^2/4: 1, 0: 1}

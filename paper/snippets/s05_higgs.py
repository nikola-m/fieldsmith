from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.noether import symmetry_defect

M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.u1(), "e")                        # photon A_mu, coupling e
G.add_matter("Phi", 1, {"A": 1}, complex=True, latex=r"\Phi") # charge-1 complex scalar
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + (D_\mu\Phi)^\dagger (D^\mu\Phi)"
                 r" + \mu^2\Phi^\dagger\Phi - \lambda(\Phi^\dagger\Phi)^2")
(alpha,) = G.gauge_parameters("A")
print(symmetry_defect(M, L, G.gauge_variation({"A": [alpha]})))   # 0: local U(1) invariance

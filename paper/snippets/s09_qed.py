from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.u1(), "e")
G.add_matter("psi", 4, {"A": 1}, complex=True, spinor=True, latex=r"\psi")
M.parameters("m", positive=True)
L = M.lagrangian(r"\bar\psi (i\gamma^\mu D_\mu - m)\psi - \frac14 F_{\mu\nu}F^{\mu\nu}")
E = M.euler_lagrange(L)
from fieldsmith.analysis import fit
from fieldsmith.noether import noether_current, complex_variation
print(fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "j": r"e \bar\psi\gamma^\nu\psi"}))
nc = noether_current(M, L, complex_variation(M, "psi", r"i\psi"))   # global phase rotation
print(nc.identity_holds())

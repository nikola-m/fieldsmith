from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra

su3 = LieAlgebra.su(3)
print(su3.dim, su3.jacobi_holds(), su3.casimir_fundamental()[0, 0])   # 8 True 4/3
M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")
print(M.check(r"D_\mu(\frac12\epsilon^{\mu\nu\rho\sigma}F^a_{\rho\sigma}) = 0"))   # Bianchi, non-abelian

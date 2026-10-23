from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.hamiltonian import Hamiltonian

M = Model()
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
E = M.euler_lagrange(L)
print(fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}"}))                  # {'dF': 1}
print(M.check(r"\partial_\nu\partial_\mu F^{\mu\nu} = 0"))                         # True
print(M.check(r"\partial_\mu (\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}) = 0"))  # Bianchi
print(Hamiltonian(M, L).dirac_analysis().summary())

from fieldsmith import Model, Spacetime
from fieldsmith.hamiltonian import Hamiltonian
M = Model(Spacetime.minkowski(3))
M.parameters("mu", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{\mu}{4}\epsilon^{\mu\nu\rho}A_\mu F_{\nu\rho}")
print(Hamiltonian(M, L).dirac_analysis().dof)             # 1

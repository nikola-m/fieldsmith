from fieldsmith import Model
from fieldsmith.hamiltonian import Hamiltonian
M = Model()
M.parameters("m", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu")
print(Hamiltonian(M, L).dirac_analysis().dof)         # 3

S = Model()                                           # Stueckelberg form
S.parameters("m", positive=True)
S.vector("A"); S.scalar("theta")
S.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
Ls = S.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 (m A_\mu - \partial_\mu\theta)(m A^\mu - \partial^\mu\theta)")
print(Hamiltonian(S, Ls).dirac_analysis().summary())  # 2 first-class constraints, 3 dof

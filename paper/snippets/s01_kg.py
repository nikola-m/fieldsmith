from fieldsmith import Model
from fieldsmith.waves import linearize, dispersion
from fieldsmith.noether import canonical_stress_tensor
from fieldsmith.hamiltonian import Hamiltonian

M = Model()                                   # 3+1 Minkowski, signature (+,-,-,-)
M.parameters("m", positive=True)
M.scalar("phi")
L = M.lagrangian(r"\frac12 \partial_\mu\phi\,\partial^\mu\phi - \frac12 m^2\phi^2")
E = M.euler_lagrange(L)                       # delta S / delta phi
print(E[0])                                   # -(Box phi + m^2 phi), componentwise
lin, pert = linearize(M, E)
print(dispersion(M, lin, list(pert.values()))["omega"])
T = canonical_stress_tensor(M, L)             # T^mu_nu
print(T[0, 0])                                # energy density
H = Hamiltonian(M, L)
print(H.H)

import sympy as sp
from fieldsmith import Model
from fieldsmith.noether import noether_current, symmetry_defect
M = Model()
M.parameters("m lambda", positive=True)
phi = M.scalar("phi")
t, x, y, z = X = M.coords
L = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2 - \frac{\lambda}{4!}\phi^4")
dx = [x, t, 0, 0]                                           # boost generator  delta x^mu
var = {phi: -sum(dx[m]*sp.diff(phi, X[m]) for m in range(4))}
K = [-dx[m]*L for m in range(4)]                            # delta L = d_mu K^mu
J = noether_current(M, L, var, K)                           # checks the symmetry first
print(J.identity_holds())                                   # True

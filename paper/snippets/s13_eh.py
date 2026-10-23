from fieldsmith import Model
M = Model()
M.parameters("c_1 c_2", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"\tilde{F}^{\mu\nu}", r"\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + c_1 (F_{\mu\nu}F^{\mu\nu})^2 + c_2 (F_{\mu\nu}\tilde F^{\mu\nu})^2")
E = M.euler_lagrange(L)
import sympy as sp
from fieldsmith.waves import linearize, dispersion
B0 = M.parameters("B_0", positive=True)
x = M.coords[1]
A = M.tensors["A"].funcs
lin, pert = linearize(M, E, background={A[(2,)]: B0 * x})     # uniform B_0 along z
res = dispersion(M, lin, list(pert.values()))
w, ky = res["wave"].omega, res["wave"].k[1]
K = res["K"].subs({res["wave"].k[0]: 0, res["wave"].k[2]: 0})  # propagate along y
print(sp.factor(K[1, 1]), sp.factor(K[3, 3]))                   # the two linear polarisations

from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.waves import linearize, dispersion
M = Model()
M.parameters("a", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"j_\nu", r"\partial^\mu F_{\mu\nu}")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{a^2}{2} j_\nu j^\nu")
E = M.euler_lagrange(L)                                   # fourth order
print(fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "box": r"\Box\partial_\mu F^{\mu\nu}"}))
lin, pert = linearize(M, E)
print(dispersion(M, lin, list(pert.values()))["pdet"])

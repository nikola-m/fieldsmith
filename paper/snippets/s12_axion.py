from fieldsmith import Model
from fieldsmith.analysis import fit
M = Model()
M.parameters("m_a g", positive=True)
M.vector("A"); M.scalar("a")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"\tilde{F}^{\mu\nu}", r"\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12\partial_\mu a\partial^\mu a"
                 r" - \frac12 m_a^2 a^2 - \frac{g}{4} a F_{\mu\nu}\tilde F^{\mu\nu}")
E = M.euler_lagrange(L)
print(fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "ax": r"\partial_\mu a\,\tilde F^{\mu\nu}"}))
print(M.check(r"F_{\mu\nu}\tilde F^{\mu\nu} = \partial_\mu(2\epsilon^{\mu\nu\rho\sigma}A_\nu\partial_\rho A_\sigma)"))

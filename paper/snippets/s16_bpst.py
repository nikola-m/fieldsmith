import sympy as sp
from fieldsmith import Model, Spacetime
from fieldsmith.gauge import GaugeTheory, LieAlgebra
M = Model(Spacetime.euclidean(4))
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
g = M.symbols["g"]
rho = M.parameters("rho", positive=True)
X = M.coords
r2 = sum(c**2 for c in X)
def eta(a, mu, nu):                                  # 't Hooft symbol
    if mu < 3 and nu < 3: return sp.LeviCivita(a, mu, nu)
    if mu < 3 and nu == 3: return 1 if a == mu else 0
    if mu == 3 and nu < 3: return -1 if a == nu else 0
    return 0
Af = M.tensors["A"].funcs
ansatz = {Af[(a, mu)]: 2*sum(eta(a, mu, nu)*X[nu] for nu in range(4))/(g*(r2 + rho**2))
          for a in range(3) for mu in range(4)}
M.define(r"\tilde{F}^a_{\mu\nu}", r"\frac12\epsilon_{\mu\nu\rho\sigma}F^a_{\rho\sigma}")
Fa, Fd = M.tensors["F"].data, M.tensors["Ftilde"].data
print(all(sp.simplify((Fa[k] - Fd[k]).subs(ansatz).doit()) == 0 for k in Fa))   # True: self-dual

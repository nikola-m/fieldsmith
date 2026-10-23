import sympy as sp
from fieldsmith import Model, Spacetime

M = Model(Spacetime.minkowski(2))             # 1+1 dimensions
m, lam = M.parameters("m lambda", positive=True)
phi = M.scalar("phi")
x = M.coords[1]
V = r"\frac{\lambda}{4}\left(\phi^2 - \frac{m^2}{\lambda}\right)^2"
E = M.euler_lagrange(r"\frac12\partial_\mu\phi\,\partial^\mu\phi - " + V)
kink = (m / sp.sqrt(lam)) * sp.tanh(m * x / sp.sqrt(2))
print(E.check_solution({phi: kink}))          # True: exact kink

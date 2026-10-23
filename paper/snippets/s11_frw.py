import sympy as sp
from fieldsmith import Model, Spacetime
t, x, y, z = sp.symbols("t x y z", real=True)
a = sp.Function("a", positive=True)(t)
frw = Spacetime([t, x, y, z], sp.diag(1, -a**2, -a**2, -a**2))
print(sp.simplify(frw.einstein()[0, 0]))                 # 3 a'^2 / a^2
M = Model(frw)
M.parameters("m", positive=True)
M.scalar("phi")
E = M.euler_lagrange(r"\frac12\partial_\mu\phi\,\partial^\mu\phi - \frac12 m^2\phi^2")
M.define(r"T_{\mu\nu}", r"\partial_\mu\phi\,\partial_\nu\phi - g_{\mu\nu}"
         r"\left(\frac12\partial_\alpha\phi\,\partial^\alpha\phi - \frac12 m^2\phi^2\right)")
print(M.check(r"\nabla_\mu T^{\mu\nu} = (\Box\phi + m^2\phi)\,\partial^\nu\phi"))

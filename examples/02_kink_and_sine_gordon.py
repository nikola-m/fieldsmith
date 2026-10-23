"""02 - Solitons in 1+1 dimensions: phi^4 kink and sine-Gordon     (level: elementary)

Topics: static solutions, Bogomolny trick, kink energy, exact breather solution check.
"""
from _util import *
import sympy as sp
from fieldsmith import Model, Spacetime

# ---- phi^4 kink -----------------------------------------------------------------------------
M = Model(Spacetime.minkowski(2))
m, lam = M.parameters("m lambda", positive=True)
phi = M.scalar("phi")
t, x = M.coords
V = r"\frac{\lambda}{4}\left(\phi^2 - \frac{m^2}{\lambda}\right)^2"
L = M.lagrangian(r"\frac12 \partial_\mu\phi\partial^\mu\phi - " + V)
eom = M.euler_lagrange(L)
section("phi^4 theory: field equation")
show("EL", eom[0])

kink = (m / sp.sqrt(lam)) * sp.tanh(m * x / sp.sqrt(2))
ok("kink (m/sqrt(lam)) tanh(m x/sqrt2) solves the EOM", eom.check_solution({phi: kink}))

section("Bogomolny: energy of the kink")
V_expr = M.parse(V)
v = sp.Symbol("v")
Vv = V_expr.subs(phi, v)
dphi = sp.sqrt(2 * Vv)
E = sp.integrate(sp.simplify(dphi), (v, -m / sp.sqrt(lam), m / sp.sqrt(lam)))
show("E_kink = int sqrt(2V) dphi", sp.simplify(E))
ok("E = 2 sqrt(2) m^3 / (3 lambda)", sp.simplify(E - 2 * sp.sqrt(2) * m**3 / (3 * lam)) == 0)
# energy by direct integration of T^00
rho = sp.simplify((sp.diff(kink, x) ** 2 / 2 + V_expr.subs(phi, kink)))
show("energy density of the kink", sp.simplify(rho))
s_ = sp.Symbol("s", real=True)
E2 = sp.simplify(sp.integrate(sp.simplify(rho.subs(x, s_)), (s_, -sp.oo, sp.oo)))
show("int rho dx", E2)

# ---- sine-Gordon -----------------------------------------------------------------------------
M2 = Model(Spacetime.minkowski(2))
M2.scalar("phi")
f = M2.tensors["phi"].funcs[()]
t, x = M2.coords
eom2 = M2.euler_lagrange(r"\frac12\partial_\mu\phi\partial^\mu\phi + \cos\phi - 1")
section("Sine-Gordon")
show("EL", eom2[0])
v_ = sp.Rational(1, 2)
gam = 1 / sp.sqrt(1 - v_**2)
soliton = 4 * sp.atan(sp.exp(gam * (x - v_ * t)))
ok("moving soliton solves sine-Gordon", eom2.check_solution({f: soliton}, numeric=True))
w = sp.Rational(3, 5)
breather = 4 * sp.atan(sp.sqrt(1 - w**2) / w * sp.sin(w * t) / sp.cosh(sp.sqrt(1 - w**2) * x))
ok("breather solves sine-Gordon", eom2.check_solution({f: breather}, numeric=True))
wrong = 4 * sp.atan(sp.exp(x - t / 3))
ok("(sanity) a wrong velocity is rejected", not eom2.check_solution({f: wrong}, numeric=True))

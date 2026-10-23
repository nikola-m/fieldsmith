"""11 - Fields on curved backgrounds: FRW cosmology and Schwarzschild    (level: graduate)

Topics: curvature tensors from a metric, covariant d'Alembertian, scalar field in an expanding
        universe (Hubble friction), Friedmann equation from G_{mu nu}, covariant conservation of the
        energy-momentum tensor <=> field equation.
"""
from _util import *
import sympy as sp
from fieldsmith import Model, Spacetime

section("Spacetimes and curvature")
t, r, th, ph = sp.symbols("t r theta phi", positive=True)
Mass = sp.Symbol("M", positive=True)
f = 1 - 2 * Mass / r
schw = Spacetime([t, r, th, ph], sp.diag(f, -1 / f, -r**2, -r**2 * sp.sin(th) ** 2))
show("Schwarzschild: Einstein tensor G_{mu nu} (vacuum => 0)", schw.einstein())
show("Ricci scalar", schw.ricci_scalar())

tt, x, y, z = sp.symbols("t x y z", real=True)
a = sp.Function("a", positive=True)(tt)
frw = Spacetime([tt, x, y, z], sp.diag(1, -a**2, -a**2, -a**2))
G = frw.einstein()
H = sp.diff(a, tt) / a
show("FRW: G_00", sp.simplify(G[0, 0]))
ok("G_00 = 3 H^2", sp.simplify(G[0, 0] - 3 * H**2) == 0)

section("Scalar field in the expanding universe")
M = Model(frw)
m = M.parameters("m", positive=True)
phi = M.scalar("phi")
L = M.lagrangian(r"\frac12 \partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
E = M.euler_lagrange(L)
show("field equation", sp.simplify(E[0]))
expected = -(sp.diff(phi, tt, 2) + 3 * H * sp.diff(phi, tt) - (sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)) / a**2 + m**2 * phi)
ok("phi'' + 3 H phi' - Laplacian/a^2 + m^2 phi = 0  (Hubble friction)", sp.simplify(E[0] - expected) == 0)

section("Energy-momentum tensor and its covariant conservation")
M.define(r"T_{\mu\nu}", r"\partial_\mu\phi\partial_\nu\phi - g_{\mu\nu}\left(\frac12\partial_\alpha\phi\partial^\alpha\phi - \frac12 m^2\phi^2\right)")
ok("nabla_mu T^{mu nu} = (Box phi + m^2 phi) d^nu phi   (conserved on shell)",
   M.check(r"\nabla_\mu T^{\mu\nu} = (\Box\phi + m^2\phi)\partial^\nu\phi"))

section("Friedmann equation")
rho = sp.simplify(M.parse(r"T_{00}"))
phi_h = sp.Function("phi_h")(tt)                       # homogeneous field phi(t)
rho_h = sp.simplify(rho.subs(phi, phi_h).doit())
show("energy density T_00 of a homogeneous field", rho_h)
Gn = sp.Symbol("G_N", positive=True)
ok("G_00 = 8 pi G_N T_00  <=>  3 H^2 = 8 pi G_N (phi'^2/2 + m^2 phi^2/2)",
   sp.simplify(G[0, 0] - 8 * sp.pi * Gn * rho_h - (3 * H**2 - 8 * sp.pi * Gn * rho_h)) == 0)
print("  (the Friedmann equation is the 00 Einstein equation with this source)")

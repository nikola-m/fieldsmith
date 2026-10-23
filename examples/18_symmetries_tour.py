"""18 - A tour of spacetime symmetries and their Noether currents      (level: intermediate-advanced)

Topics: translations, Lorentz boosts, rotations, shifts, dilatations (scale invariance and its
        breaking by a mass term), boundary terms K^mu, Noether's identity, conserved charges.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.noether import noether_current, symmetry_defect

M = Model()
m, lam = M.parameters("m lambda", positive=True)
phi = M.scalar("phi")
t, x, y, z = M.coords
X = M.coords
L_free = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac{\lambda}{4!}\phi^4")        # massless phi^4
L_mass = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2 - \frac{\lambda}{4!}\phi^4")

def dx(mu, spec):                      # components of an infinitesimal coordinate change delta x^mu
    return spec[mu]

section("Translations  (delta phi = d_nu phi,  K^mu = delta^mu_nu L)")
for nu in range(4):
    var = {phi: sp.diff(phi, X[nu])}
    K = [L_mass if mu == nu else 0 for mu in range(4)]
    nc = noether_current(M, L_mass, var, K)
    ok(f"nu = {nu}: canonical stress tensor row, conservation identity", nc.identity_holds())

section("Lorentz boost along x   (delta x^mu = (x, t, 0, 0) eps)")
dxs = [x, t, 0, 0]
var = {phi: -sum(dxs[mu] * sp.diff(phi, X[mu]) for mu in range(4))}
K = [-dxs[mu] * L_mass for mu in range(4)]
nc = noether_current(M, L_mass, var, K)
ok("boost is a symmetry; Noether identity holds", nc.identity_holds())
show("charge density J^0 of the boost", sp.simplify(nc.J[0]))
print("   = x T^{00} - t T^{01}: conservation <=> uniform motion of the centre of energy")

section("Rotation about z   (delta x^mu = (0, -y, x, 0) eps)")
dxs = [0, -y, x, 0]
var = {phi: -sum(dxs[mu] * sp.diff(phi, X[mu]) for mu in range(4))}
K = [-dxs[mu] * L_mass for mu in range(4)]
nc = noether_current(M, L_mass, var, K)
show("angular-momentum density J^0", sp.simplify(nc.J[0]))
ok("conserved", nc.identity_holds())

section("Shift symmetry of the free massless scalar  (delta phi = 1)")
L0 = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi")
nc = noether_current(M, L0, {phi: 1})
show("J^mu = d^mu phi:", [sp.simplify(j) for j in nc.J])
try:
    noether_current(M, L_free, {phi: 1})
except ValueError as e:
    print("   with a phi^4 potential the shift is not a symmetry ->", str(e)[:60], "...")

section("Dilatations  (delta x = x, delta phi = -(1 + x.d) phi,  K^mu = -x^mu L)")
dxs = list(X)
var = {phi: -(phi + sum(dxs[mu] * sp.diff(phi, X[mu]) for mu in range(4)))}
K = [-dxs[mu] * L_free for mu in range(4)]
show("massless phi^4:  delta L - d_mu K^mu", symmetry_defect(M, L_free, var, K))
nc = noether_current(M, L_free, var, K)
ok("massless phi^4 is scale invariant: dilatation current conserved", nc.identity_holds())
K2 = [-dxs[mu] * L_mass for mu in range(4)]
d = symmetry_defect(M, L_mass, var, K2)
show("with a mass term:  delta L - d_mu K^mu", d)
ok("scale invariance is broken by exactly the mass term, defect = m^2 phi^2",
   sp.simplify(d - m**2 * phi**2) == 0 or sp.simplify(d + m**2 * phi**2) == 0)

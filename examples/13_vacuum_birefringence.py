"""13 - Nonlinear electrodynamics: vacuum birefringence               (level: research-flavoured)

Topics: Euler-Heisenberg-type Lagrangian L = -F^2/4 + c1 (F^2)^2 + c2 (F Ftilde)^2, nonlinear field
        equations, linearisation about a constant magnetic field, polarisation-dependent refractive
        index, the famous 4:7 ratio of the Euler-Heisenberg effective action.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.waves import linearize, dispersion

M = Model()
c1, c2, B0 = M.parameters("c_1 c_2 B_0", positive=True)
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"\tilde{F}^{\mu\nu}", r"\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + c_1 (F_{\mu\nu}F^{\mu\nu})^2 + c_2 (F_{\mu\nu}\tilde F^{\mu\nu})^2")
E = M.euler_lagrange(L)
section("Field equations (cubic in the field strength)")
show("number of terms in E^0", len(sp.expand(E[0]).args))

section("Linearise about a uniform magnetic field B_0 along z   (A_y = B_0 x)")
x = M.coords[1]
A = M.tensors["A"].funcs
lin, pert = linearize(M, E, background={A[(2,)]: B0 * x})
res = dispersion(M, lin, list(pert.values()))
w = res["wave"].omega
ky = res["wave"].k[1]
K = res["K"].subs({res["wave"].k[0]: 0, res["wave"].k[2]: 0})      # propagate along y, perpendicular to B_0
# transverse polarisations: A_x (E perpendicular to B_0) and A_z (E parallel to B_0)
for name, i in (("E perpendicular B_0 (A_x)", 1), ("E parallel B_0 (A_z)", 3)):
    d = sp.factor(K[i, i])
    sol = [s_ for s_ in sp.solve(d, w) if sp.simplify(s_.subs({ky: 1, B0: 0})) > 0]
    n = sp.series(sp.simplify(ky / sol[0]), B0, 0, 3).removeO()
    show(f"{name}: omega^2 equation", d)
    globals()[f"n_{i}"] = sp.simplify(n)
    show("   refractive index n = k/omega", sp.simplify(n))
dn_perp, dn_par = sp.simplify(n_1 - 1), sp.simplify(n_3 - 1)
show("n_perp - 1", dn_perp)
show("n_par  - 1", dn_par)
print("   birefringence: n_par - n_perp =", sp.simplify(dn_par - dn_perp))

section("Euler-Heisenberg values")
alpha, m_e = sp.symbols("alpha m_e", positive=True)
# L_EH = (2 alpha^2 / 45 m^4) [ (E^2-B^2)^2 + 7 (E.B)^2 ],  F^2 = -2(E^2-B^2),  F Ftilde = -4 E.B
c1_eh = (2 * alpha**2 / (45 * m_e**4)) / 4
c2_eh = (2 * alpha**2 / (45 * m_e**4)) * 7 / 16
sub = {c1: c1_eh, c2: c2_eh}
r_perp, r_par = dn_perp.subs(sub), dn_par.subs(sub)
show("n_perp - 1 (EH)", sp.simplify(r_perp))
show("n_par  - 1 (EH)", sp.simplify(r_par))
ratio = sp.simplify(r_par / r_perp)
show("(n_par - 1)/(n_perp - 1)", ratio)
print("   which of the two polarisations is 'parallel' is a convention; the Euler-Heisenberg ratio is 7/4 (or 4/7).")
ok("ratio is 7/4 or 4/7", ratio in (sp.Rational(7, 4), sp.Rational(4, 7)))

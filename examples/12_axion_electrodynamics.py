"""12 - Axion electrodynamics and topological terms                      (level: research-flavoured)

Topics: the F F-dual coupling, Chern-Simons current (F Ftilde is a total derivative), modified
        Maxwell equations, cosmic birefringence in a rolling axion background.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.waves import dispersion

M = Model()
ma, g = M.parameters("m_a g", positive=True)
M.vector("A")
M.scalar("a")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
M.define(r"\tilde{F}^{\mu\nu}", r"\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}")

section("F Ftilde is a total derivative (Chern-Simons current)")
ok("F_{mu nu} Ftilde^{mu nu} = d_mu (2 eps^{mu nu rho sigma} A_nu d_rho A_sigma)",
   M.check(r"F_{\mu\nu}\tilde F^{\mu\nu} = \partial_\mu(2\epsilon^{\mu\nu\rho\sigma}A_\nu\partial_\rho A_\sigma)"))
Ex, Ey, Ez = M.parse("F_{10}"), M.parse("F_{20}"), M.parse("F_{30}")      # E_i = F_{i0}
Bx, By, Bz = M.parse("F_{23}"), M.parse("F_{31}"), M.parse("F_{12}")      # B_i = (1/2) eps_{ijk} F_{jk}
ok("F_{mu nu} Ftilde^{mu nu} = -4 E.B  (eps^{0123} = +1)",
   sp.simplify(M.parse(r"F_{\mu\nu}\tilde F^{\mu\nu}") + 4 * (Ex * Bx + Ey * By + Ez * Bz)) == 0)

section("Axion-photon Lagrangian")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12\partial_\mu a\partial^\mu a - \frac12 m_a^2 a^2 - \frac{g}{4} a F_{\mu\nu}\tilde F^{\mu\nu}")
E = M.euler_lagrange(L)
show("photon: dS/dA_nu = c1 d_mu F^{mu nu} + c2 (d_mu a) Ftilde^{mu nu}", fit(
    M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "ax": r"\partial_\mu a\,\tilde F^{\mu\nu}"}))
show("axion: dS/da = -(Box a + m_a^2 a) - (g/4) F Ftilde ?", sp.simplify(
    E["a"] + M.parse(r"\Box a + m_a^2 a") + g / 4 * M.parse(r"F_{\mu\nu}\tilde F^{\mu\nu}")) == 0)

section("A constant axion decouples (the term is topological); a rolling axion makes light birefringent")
A, a_f = M.tensors["A"].funcs, M.tensors["a"].funcs[()]
t = M.coords[0]
kappa = sp.Symbol("kappa", positive=True)
EA = [sp.simplify(e.subs(a_f, kappa * t).doit()) for e in (E[A[(n,)]] for n in range(4))]
res = dispersion(M, EA, [A[(n,)] for n in range(4)])
K = res["K"].subs({res["wave"].k[0]: 0, res["wave"].k[1]: 0})    # propagation along z
kz, w = res["wave"].k[2], res["wave"].omega
Kt = K.extract([1, 2], [1, 2])                                  # transverse polarisations x, y
show("2x2 transverse wave matrix", Kt)
det = sp.factor(Kt.det())
show("det", det)
sol = sp.solve(det, w)
show("omega(k) for the two circular polarisations", [sp.simplify(s_) for s_ in sol])
print("=> omega^2 = k^2 +/- g kappa k : the two circular polarisations propagate differently")

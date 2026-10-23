"""07 - Yang-Mills theory: SU(2) and SU(3)                            (level: advanced undergrad / grad)

Topics: Lie-algebra data, non-abelian field strength, gauge invariance of the action, covariant
        field equations D_mu F^{mu nu} = 0, Bianchi identity, covariant current conservation,
        the colour Noether current of the global symmetry.
"""
from _util import *
import time
import sympy as sp
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import fit
from fieldsmith.noether import symmetry_defect, noether_current

section("Lie algebra data")
for N in (2, 3):
    alg = LieAlgebra.su(N)
    show(f"su({N}): dim", alg.dim)
    show("  Jacobi identity", alg.jacobi_holds())
    show("  Casimir in the fundamental  C_F 1 =", alg.casimir_fundamental()[0, 0])
alg3 = LieAlgebra.su(3)
show("su(3) nonzero f^{abc} with a<b<c (1-based)", {(a + 1, b + 1, c + 1): v for (a, b, c), v in alg3.f.items() if a < b < c})

section("SU(2) Yang-Mills")
M = Model()
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")
L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")
g = M.symbols["g"]
show("number of terms in L (kinetic + g cubic + g^2 quartic)", len(L.args))
show("highest power of g in L", sp.degree(L, g))

section("Local gauge invariance:  delta A^a_mu = d_mu alpha^a / g + f^{abc} A^b_mu alpha^c")
alpha = G.gauge_parameters("A")
t0 = time.time()
show("delta L", symmetry_defect(M, L, G.gauge_variation({"A": alpha})))
print(f"  ({time.time() - t0:.1f} s)")

section("Covariant field equations")
E = M.euler_lagrange(L)
show("delta S/delta A^a_nu = c D_mu F^{a mu nu},  c", fit(M, E.tensor("A"), {"DF": r"D_\mu F^{a\mu\nu}"}))
ok("Bianchi identity  D_mu tilde F^{a mu nu} = 0", M.check(r"D_\mu (\frac12\epsilon^{\mu\nu\rho\sigma}F^a_{\rho\sigma}) = 0"))
ok("D_nu D_mu F^{a mu nu} = 0   (identically, off shell)", M.check(r"D_\nu D_\mu F^{a\mu\nu} = 0"))
M.field("chi", ("ad_A",), "-")          # an adjoint-valued test field chi^a
ok("[D_mu, D_nu] chi^a = g f^{abc} F^b_{mu nu} chi^c   (curvature = commutator of covariant derivatives)",
   M.check(r"D_\mu D_\nu \chi^a - D_\nu D_\mu \chi^a = g f^{abc} F^b_{\mu\nu}\chi^c"))

section("Global colour symmetry: Noether current")
a1, a2, a3 = [sp.Symbol(f"a{i}") for i in (1, 2, 3)]
al = [a1, a2, a3]
Af = M.tensors["A"].funcs
var = {Af[(a, mu)]: sum(sp.LeviCivita(a, b, c) * Af[(b, mu)] * al[c] for b in range(3) for c in range(3))
       for a in range(3) for mu in range(4)}
nc = noether_current(M, L, var)
show("color charge density J^0 ... conserved on shell:", nc.identity_holds())

section("SU(3) gluodynamics: build and check")
M3 = Model()
G3 = GaugeTheory(M3)
G3.add_group("G", LieAlgebra.su(3), "g_s", strength="S")
L3 = M3.lagrangian(r"-\frac14 S^a_{\mu\nu}S^{a\mu\nu}")
show("gluon fields", len(M3.tensors["G"].funcs))
show("terms in L", len(L3.args))
ok("Bianchi identity for SU(3)", M3.check(r"D_\mu (\frac12\epsilon^{\mu\nu\rho\sigma}S^a_{\rho\sigma}) = 0"))

"""10 - Energy-momentum tensors: canonical, Belinfante, Hilbert, improvement   (level: graduate)

Topics: Noether's canonical tensor vs the metric (Hilbert) tensor, the Belinfante relation for the
        photon, conformal improvement of the scalar field (trace vanishes on shell), conservation.
"""
from _util import *
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import canonical_stress_tensor, hilbert_stress_tensor

section("Photon: canonical vs Hilbert tensor (Belinfante)")
M = Model()
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
LM = r"-\frac14 F_{\mu\nu}F^{\mu\nu}"
Tc = canonical_stress_tensor(M, LM).raise_("nu")        # T_can^{mu nu}: not symmetric, not gauge invariant
Th = hilbert_stress_tensor(M, LM).raise_("mu").raise_("nu")
ok("T_can is not symmetric", not Tc.is_symmetric("mu", "nu"))
ok("T_Hilbert is symmetric", Th.is_symmetric("mu", "nu"))
D = (Th - Tc.relabel(["mu", "nu"]))
rel = fit(M, D, {"belinfante": r"\partial_\lambda(F^{\mu\lambda}A^\nu)", "eom": r"(\partial_\lambda F^{\mu\lambda})A^\nu"},
          labels=["mu", "nu"])
show("T_Hilb - T_can = c1 d_lam(F^{mu lam} A^nu) + c2 (d_lam F^{mu lam}) A^nu", rel)
print("  => on shell they differ by the total derivative d_lam(F^{mu lam} A^nu): same charges,")
print("     but only T_Hilb is gauge invariant and symmetric (a genuine energy-momentum tensor).")

section("Free massless scalar: conformal improvement")
S = Model()
S.scalar("phi")
Ls = r"\frac12\partial_\mu\phi\partial^\mu\phi"
T = hilbert_stress_tensor(S, Ls).raise_("mu").raise_("nu")
phi = S.tensors["phi"].funcs[()]
box = S.parse(r"\Box\phi")
trace = sum(S.st.g[i, i] * T[(i, i)] for i in range(4))
show("trace of the ordinary tensor", sp.simplify(trace))
# improvement  T -> T - (1/6)(d^mu d^nu - g^{mu nu} Box) phi^2
imp = S.parse(r"\partial^\mu\partial^\nu(\phi^2) - g^{\mu\nu}\Box(\phi^2)", as_tensor=True)
Timp = T - imp.scale(sp.Rational(1, 6))
trace_imp = sum(S.st.g[i, i] * Timp[(i, i)] for i in range(4))
show("trace of the improved tensor", sp.simplify(trace_imp))
ok("improved trace = phi Box phi  (vanishes on shell: classical conformal invariance)",
   sp.simplify(trace_imp - phi * box) == 0)

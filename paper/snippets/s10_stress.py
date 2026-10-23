from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import canonical_stress_tensor, hilbert_stress_tensor
M = Model()
M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
LM = r"-\frac14 F_{\mu\nu}F^{\mu\nu}"
Tc = canonical_stress_tensor(M, LM).raise_("nu")
Th = hilbert_stress_tensor(M, LM).raise_("mu").raise_("nu")
print(Tc.is_symmetric("mu", "nu"), Th.is_symmetric("mu", "nu"))     # False True
print(fit(M, Th - Tc.relabel(["mu", "nu"]),
          {"b": r"\partial_\lambda(F^{\mu\lambda}A^\nu)", "e": r"(\partial_\lambda F^{\mu\lambda})A^\nu"},
          labels=["mu", "nu"]))

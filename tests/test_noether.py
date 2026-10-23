import pytest
import sympy as sp
from fieldsmith import Model
from fieldsmith.analysis import fit
from fieldsmith.noether import (noether_current, complex_variation, symmetry_defect,
                                canonical_stress_tensor, hilbert_stress_tensor)


def test_u1_charge_density_and_identity():
    M = Model()
    M.parameters("m", positive=True)
    f, fs = M.scalar("phi", complex=True)
    nc = noether_current(M, r"\partial_\mu\phi^*\partial^\mu\phi - m^2\phi^*\phi",
                         complex_variation(M, "phi", r"i\phi"))
    t = M.coords[0]
    assert sp.expand(nc.J[0] - sp.I * (f * sp.diff(fs, t) - fs * sp.diff(f, t))) == 0
    assert nc.identity_holds()


def test_non_symmetry_is_rejected():
    M = Model()
    M.parameters("lambda", positive=True)
    phi = M.scalar("phi")
    with pytest.raises(ValueError):
        noether_current(M, r"\frac12\partial_\mu\phi\partial^\mu\phi - \lambda\phi^4", {phi: 1})


def test_boost_current_conserved():
    M = Model()
    M.parameters("m", positive=True)
    phi = M.scalar("phi")
    t, x, y, z = M.coords
    L = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    dxs = [x, t, 0, 0]
    var = {phi: -sum(dxs[i] * sp.diff(phi, M.coords[i]) for i in range(4))}
    K = [-dxs[i] * L for i in range(4)]
    assert noether_current(M, L, var, K).identity_holds()


def test_dilatation_broken_by_mass():
    M = Model()
    m = M.parameters("m", positive=True)
    phi = M.scalar("phi")
    X = M.coords
    var = {phi: -(phi + sum(X[i] * sp.diff(phi, X[i]) for i in range(4)))}
    L0 = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi")
    Lm = M.lagrangian(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    assert symmetry_defect(M, L0, var, [-X[i] * L0 for i in range(4)]) == 0
    assert sp.simplify(symmetry_defect(M, Lm, var, [-X[i] * Lm for i in range(4)]) + m**2 * phi**2) == 0


def test_canonical_tensor_scalar_energy():
    M = Model()
    M.parameters("m", positive=True)
    phi = M.scalar("phi")
    T = canonical_stress_tensor(M, r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    t, x, y, z = M.coords
    rho = sp.diff(phi, t) ** 2 / 2 + sum(sp.diff(phi, c) ** 2 for c in (x, y, z)) / 2 + M.symbols["m"] ** 2 * phi**2 / 2
    assert sp.expand(T[0, 0] - rho) == 0


def test_second_order_noether_current():
    M = Model()
    phi = M.scalar("phi")
    nc = noether_current(M, r"\frac12(\Box\phi)^2", {phi: 1})       # shift symmetry of a higher-derivative theory
    assert nc.identity_holds()


def _maxwell():
    M = Model()
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    return M


def test_hilbert_tensor_maxwell():
    M = _maxwell()
    L = r"-\frac14 F_{\mu\nu}F^{\mu\nu}"
    Th = hilbert_stress_tensor(M, L)
    assert Th.is_symmetric("mu", "nu")
    assert sp.simplify(sum(M.st.ginv[i, i] * Th[i, i] for i in range(4))) == 0
    Tu = Th.raise_("mu").raise_("nu")
    E2 = sum(M.parse(f"F_{{{i}0}}") ** 2 for i in (1, 2, 3))
    B2 = sum(M.parse(s) ** 2 for s in ("F_{23}", "F_{31}", "F_{12}"))
    assert sp.simplify(Tu[0, 0] - (E2 + B2) / 2) == 0


def test_belinfante_relation():
    M = _maxwell()
    L = r"-\frac14 F_{\mu\nu}F^{\mu\nu}"
    Tc = canonical_stress_tensor(M, L).raise_("nu")
    assert not Tc.is_symmetric("mu", "nu")
    Th = hilbert_stress_tensor(M, L).raise_("mu").raise_("nu")
    rel = fit(M, Th - Tc.relabel(["mu", "nu"]),
              {"b": r"\partial_\lambda(F^{\mu\lambda}A^\nu)", "e": r"(\partial_\lambda F^{\mu\lambda})A^\nu"},
              labels=["mu", "nu"])
    assert rel == {"b": 1, "e": -1}


def test_gauge_invariance_of_maxwell():
    M = _maxwell()
    alpha = sp.Function("alpha")(*M.coords)
    var = {M.tensors["A"].funcs[(i,)]: sp.diff(alpha, M.coords[i]) for i in range(4)}
    assert symmetry_defect(M, M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}"), var) == 0

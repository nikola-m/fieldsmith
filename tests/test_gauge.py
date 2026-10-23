import sympy as sp
import pytest
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.noether import symmetry_defect, noether_current, complex_variation


def test_su2_structure_constants_and_jacobi():
    alg = LieAlgebra.su(2)
    assert alg.dim == 3 and alg.jacobi_holds()
    for (a, b, c), v in alg.f.items():
        assert v == sp.LeviCivita(a, b, c)


def test_su3_structure_constants():
    alg = LieAlgebra.su(3)
    assert alg.dim == 8 and alg.jacobi_holds()
    assert alg.f[(0, 1, 2)] == 1
    assert alg.f[(3, 4, 7)] == sp.sqrt(3) / 2
    assert alg.f[(0, 3, 6)] == sp.Rational(1, 2)
    assert alg.casimir_fundamental()[0, 0] == sp.Rational(4, 3)


def test_generators_normalisation():
    for N in (2, 3, 4):
        alg = LieAlgebra.su(N)
        for a, Ta in enumerate(alg.generators):
            for b, Tb in enumerate(alg.generators):
                assert sp.simplify((Ta * Tb).trace() - (sp.Rational(1, 2) if a == b else 0)) == 0


def _ym(group=2):
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.su(group), "g")
    return M, G


def test_yang_mills_gauge_invariance_su2():
    M, G = _ym()
    L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")
    alpha = G.gauge_parameters("A")
    assert symmetry_defect(M, L, G.gauge_variation({"A": alpha})) == 0


def test_wrong_sign_convention_is_detected():
    """Flipping the sign of the structure-constant term in delta A must break invariance."""
    M, G = _ym()
    L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")
    alpha = G.gauge_parameters("A")
    var = G.gauge_variation({"A": alpha})
    Af = M.tensors["A"].funcs
    bad = {}
    for (a, mu), f in Af.items():
        bad[f] = sp.diff(alpha[a], M.coords[mu]) / M.symbols["g"] - (var[f] - sp.diff(alpha[a], M.coords[mu]) / M.symbols["g"])
    assert symmetry_defect(M, L, bad) != 0


def test_bianchi_and_commutator_su2():
    M, G = _ym()
    M.field("chi", ("ad_A",), "-")
    assert M.check(r"D_\mu (\frac12\epsilon^{\mu\nu\rho\sigma}F^a_{\rho\sigma}) = 0")
    assert M.check(r"D_\mu D_\nu\chi^a - D_\nu D_\mu\chi^a = g f^{abc}F^b_{\mu\nu}\chi^c")
    assert not M.check(r"D_\mu D_\nu\chi^a - D_\nu D_\mu\chi^a = -g f^{abc}F^b_{\mu\nu}\chi^c")


def test_yang_mills_equation_covariant_form():
    from fieldsmith.analysis import fit
    M, G = _ym()
    E = M.euler_lagrange(M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}"))
    assert fit(M, E.tensor("A"), {"DF": r"D_\mu F^{a\mu\nu}"}) == {"DF": 1}


def test_electroweak_masses():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("W", LieAlgebra.su(2), "g", strength="V")
    G.add_group("B", LieAlgebra.u1(), "g'", strength="H")
    v = M.parameters("v", positive=True)
    G.add_matter("Phi", 2, {"W": "fund", "B": sp.Rational(1, 2)}, complex=True)
    Lk = M.lagrangian(r"(D_\mu\Phi)^\dagger(D^\mu\Phi)")
    Phi = M.tensors["Phi"].data[()]
    vev = {Phi[1]: v / sp.sqrt(2), M._conj_subs[Phi[1]]: v / sp.sqrt(2), Phi[0]: 0, M._conj_subs[Phi[0]]: 0}
    M2, _ = G.gauge_mass_matrix(Lk, vev)
    g, gp = M.symbols["g"], M.symbols["g'"]
    ev = M2.eigenvals()
    assert ev[sp.simplify(g**2 * v**2 / 4)] == 2
    assert ev[0] == 1
    assert any(sp.simplify(k - (g**2 + gp**2) * v**2 / 4) == 0 for k in ev)


def test_scalar_qed_noether_and_gauge_invariance():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.u1(), "e")
    G.add_matter("Phi", 1, {"A": 1}, complex=True)
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + (D_\mu\Phi)^\dagger(D^\mu\Phi)")
    (alpha,) = G.gauge_parameters("A")
    assert symmetry_defect(M, L, G.gauge_variation({"A": [alpha]})) == 0
    assert noether_current(M, L, complex_variation(M, "Phi", r"i\Phi")).identity_holds()


def test_dirac_equation_from_qed_action():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.u1(), "e")
    M.parameters("m", positive=True)
    G.add_matter("psi", 4, {"A": 1}, complex=True, spinor=True)
    L = M.lagrangian(r"\bar\psi(i\gamma^\mu D_\mu - m)\psi")
    E = M.euler_lagrange(L)
    dirac = M.parse(r"(i\gamma^\mu D_\mu - m)\psi", as_tensor=True).value()
    resid = sp.Matrix([E[M._conj_subs[f]] for f in M.tensors["psi"].data[()]])
    assert sp.simplify(resid - M.gamma0 * dirac) == sp.zeros(4, 1)
    (alpha,) = G.gauge_parameters("A")
    assert symmetry_defect(M, L, G.gauge_variation({"A": [alpha]})) == 0


def test_hilbert_tensor_of_charged_scalar_with_covariant_derivative():
    from fieldsmith.noether import hilbert_stress_tensor
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.u1(), "e")
    M.parameters("m", positive=True)
    G.add_matter("Phi", 1, {"A": 1}, complex=True)
    T = hilbert_stress_tensor(M, r"(D_\mu\Phi)^\dagger(D^\mu\Phi) - m^2\Phi^\dagger\Phi")
    hand = M.parse(r"(D_\mu\Phi)^\dagger D_\nu\Phi + (D_\nu\Phi)^\dagger D_\mu\Phi"
                   r" - g_{\mu\nu}((D_\alpha\Phi)^\dagger D^\alpha\Phi - m^2\Phi^\dagger\Phi)", as_tensor=True)
    assert T.equals(hand.relabel(T.labels))
    assert T.is_symmetric("mu", "nu")

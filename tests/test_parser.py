import pytest
import sympy as sp
from fieldsmith import Model, ParseError
from fieldsmith.tensors import IndexError_


@pytest.fixture
def M():
    m = Model()
    m.parameters("m lambda", positive=True)
    m.scalar("phi")
    m.vector("A")
    m.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    return m


def test_numbers_and_fractions(M):
    assert M.parse(r"\frac12") == sp.Rational(1, 2)
    assert M.parse(r"\frac{3}{4}") == sp.Rational(3, 4)
    assert M.parse(r"\frac14 + \frac{1}{4}") == sp.Rational(1, 2)
    assert M.parse("4!") == 24
    assert M.parse(r"\sqrt{9}") == 3
    assert M.parse("0.25") == sp.Rational(1, 4)


def test_implicit_multiplication_and_powers(M):
    m = M.symbols["m"]
    assert M.parse(r"2 m^2") == 2 * m**2
    assert M.parse(r"(m+1)^2") == (m + 1) ** 2
    assert M.parse(r"m^{-1}") == 1 / m


def test_scalar_power_vs_index(M):
    phi = M.tensors["phi"].funcs[()]
    assert M.parse(r"\phi^2") == phi**2
    assert M.parse(r"\phi^4") == phi**4


def test_derivative_scope(M):
    phi = M.tensors["phi"].funcs[()]
    t = M.coords[0]
    assert M.parse(r"\partial_t \phi^2") == sp.diff(phi**2, t)           # acts on the whole power
    assert M.parse(r"\partial_t^2\phi") == sp.diff(phi, t, 2)
    assert M.parse(r"(\partial_t\phi)^2") == sp.diff(phi, t) ** 2


def test_einstein_summation(M):
    phi = M.tensors["phi"].funcs[()]
    X = M.coords
    expect = sp.diff(phi, X[0]) ** 2 - sum(sp.diff(phi, c) ** 2 for c in X[1:])
    assert sp.expand(M.parse(r"\partial_\mu\phi\,\partial^\mu\phi") - expect) == 0
    assert sp.expand(M.parse(r"(\partial_\mu\phi)^2") - expect) == 0     # square of a vector


def test_box(M):
    phi = M.tensors["phi"].funcs[()]
    X = M.coords
    box = sp.diff(phi, X[0], 2) - sum(sp.diff(phi, c, 2) for c in X[1:])
    assert sp.expand(M.parse(r"\Box\phi") - box) == 0


def test_check_true_and_false(M):
    assert M.check(r"\partial_\mu F^{\mu\nu} = \Box A^\nu - \partial^\nu\partial_\mu A^\mu")
    assert not M.check(r"\partial_\mu F^{\mu\nu} = \Box A^\nu")


def test_bianchi_identity(M):
    assert M.check(r"\partial_\mu F_{\nu\rho} + \partial_\nu F_{\rho\mu} + \partial_\rho F_{\mu\nu} = 0")


def test_epsilon_and_delta(M):
    assert M.check(r"\delta^\mu_\nu A^\nu = A^\mu")
    assert M.check(r"\epsilon^{\mu\nu\rho\sigma}\epsilon_{\mu\nu\rho\sigma} = -24")
    assert M.check(r"g_{\mu\nu}g^{\mu\nu} = 4")


def test_imaginary_unit_and_dagger():
    M = Model()
    M.scalar("psi", complex=True)
    psi, psib = M.scalar("chi", complex=True)
    assert M.parse("i") == sp.I
    f = M.tensors["chi"].funcs[()]
    assert M.parse(r"\chi^*") == psib
    assert M.parse(r"i\chi^\dagger") == sp.I * psib


def test_errors(M):
    with pytest.raises(IndexError_):
        M.parse(r"A_\mu + \phi")                  # free index mismatch
    with pytest.raises(ParseError):
        M.parse(r"\frac{1}{2")
    with pytest.raises(ParseError):
        M.parse(r"\partial_\mu")                  # operator without operand
    with pytest.raises(IndexError_):
        M.lagrangian(r"A_\mu")                    # Lagrangian must be scalar


def test_strict_mode_rejects_repeated_lower_indices():
    M = Model(strict=True)
    M.vector("A")
    with pytest.raises(IndexError_):
        M.parse(r"A_\mu A_\mu")
    assert M.parse(r"A_\mu A^\mu") is not None

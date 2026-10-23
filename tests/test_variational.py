import sympy as sp
from fieldsmith import Model, Spacetime
from fieldsmith.analysis import fit


def test_klein_gordon():
    M = Model()
    m = M.parameters("m", positive=True)
    phi = M.scalar("phi")
    E = M.euler_lagrange(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    box_term = M.parse(r"\Box\phi + m^2\phi")
    assert sp.simplify(E[0] + box_term) == 0


def test_complex_scalar_independent_variation():
    M = Model()
    m = M.parameters("m", positive=True)
    M.scalar("phi", complex=True)
    E = M.euler_lagrange(r"\partial_\mu\phi^*\partial^\mu\phi - m^2\phi^*\phi")
    assert len(E) == 2
    f = M.tensors["phi"].funcs[()]
    fs = M._conj_subs[f]
    assert sp.simplify(E[fs] + M.parse(r"\Box\phi + m^2\phi")) == 0


def test_maxwell_covariant_form():
    M = Model()
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    E = M.euler_lagrange(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
    assert fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}"}) == {"dF": 1}


def test_proca_covariant_form():
    M = Model()
    M.parameters("m", positive=True)
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    E = M.euler_lagrange(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu")
    assert fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "mA": r"m^2A^\nu"}) == {"dF": 1, "mA": 1}


def test_higher_derivative_euler_lagrange():
    M = Model()
    M.scalar("phi")
    E = M.euler_lagrange(r"\frac12 (\Box\phi)^2")        # L = (Box phi)^2 / 2  ->  Box^2 phi = 0
    box2 = M.parse(r"\Box\Box\phi")
    assert sp.simplify(E[0] - box2) == 0


def test_curved_space_scalar_hubble_friction():
    t, x, y, z = sp.symbols("t x y z", real=True)
    a = sp.Function("a", positive=True)(t)
    M = Model(Spacetime([t, x, y, z], sp.diag(1, -a**2, -a**2, -a**2)))
    m = M.parameters("m", positive=True)
    phi = M.scalar("phi")
    E = M.euler_lagrange(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    H = sp.diff(a, t) / a
    lap = sum(sp.diff(phi, c, 2) for c in (x, y, z))
    expected = -(sp.diff(phi, t, 2) + 3 * H * sp.diff(phi, t) - lap / a**2 + m**2 * phi)
    assert sp.simplify(E[0] - expected) == 0


def test_check_solution_symbolic_and_numeric():
    M = Model(Spacetime.minkowski(2))
    m, lam = M.parameters("m lambda", positive=True)
    phi = M.scalar("phi")
    x = M.coords[1]
    E = M.euler_lagrange(r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac{\lambda}{4}(\phi^2 - m^2/\lambda)^2")
    kink = m / sp.sqrt(lam) * sp.tanh(m * x / sp.sqrt(2))
    assert E.check_solution({phi: kink})
    assert E.check_solution({phi: kink}, numeric=True)
    assert not E.check_solution({phi: m / sp.sqrt(lam) * sp.tanh(m * x)}, numeric=True)


def test_fit_rejects_targets_outside_the_span():
    M = Model()
    M.parameters("m", positive=True)
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    E = M.euler_lagrange(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu").tensor("A")
    assert fit(M, E, {"dF": r"\partial_\mu F^{\mu\nu}"}) is None          # mass term missing from the basis
    assert fit(M, E, {"dF": r"\partial_\mu F^{\mu\nu}", "mA": r"m^2A^\nu", "extra": r"\Box A^\nu"})["extra"] == 0

import sympy as sp
from fieldsmith import Model, Spacetime


def test_metric_raise_lower_minkowski():
    M = Model()
    M.vector("A")
    A = M.tensors["A"].funcs
    up = M.parse(r"A^\mu", as_tensor=True)
    assert up[(0,)] == A[(0,)] and up[(1,)] == -A[(1,)]


def test_mostly_plus_signature():
    M = Model(Spacetime.minkowski(4, "mostly_plus"))
    M.scalar("phi")
    phi = M.tensors["phi"].funcs[()]
    t, x = M.coords[:2]
    L = sp.expand(M.parse(r"-\frac12\partial_\mu\phi\partial^\mu\phi"))
    assert L.coeff(sp.diff(phi, t) ** 2) == sp.Rational(1, 2)          # + kinetic energy
    assert L.coeff(sp.diff(phi, x) ** 2) == -sp.Rational(1, 2)         # - gradient energy


def test_transpose_and_symmetry():
    M = Model()
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    F = M.parse(r"F_{\mu\nu}", as_tensor=True)
    assert F.is_antisymmetric("mu", "nu")
    assert not F.is_symmetric("mu", "nu")
    S = M.parse(r"\partial_\mu A_\nu + \partial_\nu A_\mu", as_tensor=True)
    assert S.is_symmetric("mu", "nu")


def test_levi_civita_conventions():
    M = Model()
    e = M.parse(r"\epsilon^{0123}")
    assert e == 1
    assert M.parse(r"\epsilon_{0123}") == -1            # lowering with mostly-minus metric
    assert M.parse(r"\epsilon_{123}") == 1              # spatial symbol


def test_spatial_indices_plain_sum():
    M = Model()
    M.vector("A")
    A = M.tensors["A"].funcs
    assert sp.expand(M.parse(r"A_i A_i") - sum(A[(i,)] ** 2 for i in (1, 2, 3))) == 0
    assert sp.expand(M.parse(r"A_i A^i") + sum(A[(i,)] ** 2 for i in (1, 2, 3))) == 0


def test_covariant_derivative_curved_vector_divergence():
    t, r = sp.symbols("t r", positive=True)
    th, ph = sp.symbols("theta phi", positive=True)
    st = Spacetime([t, r, th, ph], sp.diag(1, -1, -r**2, -r**2 * sp.sin(th) ** 2))   # flat space in spherical coords
    assert all(c == 0 for c in st.einstein())
    M = Model(st)
    M.vector("V", var="u", dynamical=False)
    div = M.parse(r"\nabla_\mu V^\mu")
    V = M.tensors["V"].funcs
    expect = sp.diff(V[(0,)], t) + sp.diff(r**2 * V[(1,)], r) / r**2 + sp.diff(sp.sin(th) * V[(2,)], th) / sp.sin(th) + sp.diff(V[(3,)], ph)
    assert sp.simplify(div - expect) == 0

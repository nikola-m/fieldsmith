import sympy as sp
from fieldsmith import Model, Spacetime
from fieldsmith.hamiltonian import Hamiltonian
from fieldsmith.waves import linearize, dispersion


def _vec(mass=False):
    M = Model()
    M.parameters("m", positive=True)
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    L = r"-\frac14 F_{\mu\nu}F^{\mu\nu}" + (r" + \frac12 m^2 A_\mu A^\mu" if mass else "")
    return M, L


def test_scalar_hamiltonian_and_equations():
    M = Model()
    m = M.parameters("m", positive=True)
    phi = M.scalar("phi")
    H = Hamiltonian(M, r"\frac12\partial_\mu\phi\partial^\mu\phi - \frac12 m^2\phi^2")
    p = H.pi[phi]
    eqs = H.hamilton_equations()
    t = M.coords[0]
    assert eqs[sp.Derivative(phi, t)] == p
    assert sp.simplify(eqs[sp.Derivative(p, t)] - (sum(sp.diff(phi, c, 2) for c in M.coords[1:]) - m**2 * phi)) == 0


def test_maxwell_has_two_dof():
    M, L = _vec()
    res = Hamiltonian(M, L).dirac_analysis()
    assert (res.n_first, res.n_second, res.dof) == (2, 0, 2)


def test_proca_has_three_dof_second_class():
    M, L = _vec(True)
    res = Hamiltonian(M, L).dirac_analysis()
    assert (res.n_first, res.n_second, res.dof) == (0, 2, 3)


def test_proca_dispersion_and_maxwell_physical_modes():
    M, L = _vec(True)
    lin, pert = linearize(M, M.euler_lagrange(L))
    res = dispersion(M, lin, list(pert.values()))
    w, ks = res["wave"].omega, res["wave"].k
    k2 = sum(k**2 for k in ks)
    m = M.symbols["m"]
    assert sp.simplify(res["det"] + m**2 * (k2 + m**2 - w**2) ** 3) == 0
    M0, L0 = _vec()
    lin0, pert0 = linearize(M0, M0.euler_lagrange(L0))
    r0 = dispersion(M0, lin0, list(pert0.values()))
    assert r0["rank"] == 3
    assert sp.simplify(r0["pdet"] - (sum(k**2 for k in r0["wave"].k) - r0["wave"].omega ** 2) ** 2) == 0


def test_linearize_about_background():
    M = Model(Spacetime.minkowski(2))
    m, lam = M.parameters("m lambda", positive=True)
    phi = M.scalar("phi")
    E = M.euler_lagrange(r"\frac12\partial_\mu\phi\partial^\mu\phi + \frac{m^2}{2}\phi^2 - \frac{\lambda}{4}\phi^4")
    v = m / sp.sqrt(lam)
    lin, pert = linearize(M, E, background={phi: v})
    res = dispersion(M, lin, list(pert.values()))
    w, (k,) = res["wave"].omega, res["wave"].k
    assert sp.simplify(res["det"] - (w**2 - k**2 - 2 * m**2)) == 0       # fluctuation mass^2 = V''(v) = 2 m^2

"""01 - Klein-Gordon field                                    (level: elementary)

Topics: LaTeX Lagrangian -> Euler-Lagrange equation, plane waves, Noether stress tensor,
        energy density, Hamiltonian and Hamilton's equations.
"""
from _util import *
from fieldsmith import Model
from fieldsmith.noether import canonical_stress_tensor
from fieldsmith.hamiltonian import Hamiltonian
from fieldsmith.waves import linearize, dispersion

M = Model()
m, lam = M.parameters("m lambda", positive=True)
phi = M.scalar("phi")

section("Lagrangian and field equation")
L = M.lagrangian(r"\frac12 \partial_\mu\phi\,\partial^\mu\phi - \frac12 m^2\phi^2")
show("L", L)
eom = M.euler_lagrange(L)
show("EL equation (LaTeX)", M.latex(eom[0]))
box = M.parse(r"\Box\phi + m^2\phi")
ok("EL  =  -(Box phi + m^2 phi)", sp.simplify(eom[0] + box) == 0)

section("Plane waves: dispersion relation")
lin, pert = linearize(M, eom)
res = dispersion(M, lin, list(pert.values()))
show("det K", res["det"])
show("omega", res["omega"])

section("Noether: energy density and momentum density")
T = canonical_stress_tensor(M, L)           # T^mu_nu
show("T^0_0 (energy density)", sp.simplify(T[0, 0]))
show("T^1_0  (flux of energy / Poynting-like)", sp.simplify(T[1, 0]))

section("Hamiltonian formalism")
H = Hamiltonian(M, L)
show("H", H.H)
for k, v in H.hamilton_equations().items():
    show(f"  {k}", v)

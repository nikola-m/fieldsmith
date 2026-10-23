"""06 - O(N) sigma model: Goldstone's theorem                         (level: intermediate)

Topics: user-defined index families, one Lagrangian for any N, Noether currents of all N(N-1)/2
        rotations, vacuum manifold, mass matrix with N-1 massless Goldstone modes,
        broken-generator counting.
"""
from _util import *
import itertools
import sympy as sp
from fieldsmith import Model
from fieldsmith.noether import noether_current
from fieldsmith.analysis import scalar_mass_matrix

for N in (2, 3, 4):
    section(f"N = {N}")
    M = Model()
    mu, lam = M.parameters("mu lambda", positive=True)
    M.add_index_space("flavor", list("pqrs"), N)          # indices p,q,r,s run over 1..N
    M.field("phi", ("flavor",), "-")                      # phi_p, p = 1..N
    L = M.lagrangian(r"\frac12\partial_\mu\phi_p\partial^\mu\phi_p + \frac{\mu^2}{2}\phi_p\phi_p - \frac{\lambda}{4}(\phi_p\phi_p)^2")
    f = M.tensors["phi"].funcs
    phis = [f[(i,)] for i in range(N)]

    # Noether currents of the N(N-1)/2 rotations  delta phi_a = -phi_b, delta phi_b = +phi_a
    broken = 0
    v = mu / sp.sqrt(lam)
    vac = {phis[-1]: v}
    for a, b in itertools.combinations(range(N), 2):
        nc = noether_current(M, L, {phis[a]: -phis[b], phis[b]: phis[a]})
        assert nc.identity_holds()
        # the generator is broken iff it moves the vacuum  phi = (0,..,0,v)
        moved = {phis[a]: -phis[b], phis[b]: phis[a]}
        if any(sp.sympify(val).subs(vac).subs({p: 0 for p in phis}) != 0 for val in moved.values()):
            broken += 1
    show("conserved currents", N * (N - 1) // 2)
    show("broken generators (those that move the vacuum phi=(0,...,0,v))", broken)

    V = -M.parse(r"\frac{\mu^2}{2}\phi_p\phi_p - \frac{\lambda}{4}(\phi_p\phi_p)^2")   # V = -(L without kinetic term)
    M2, lab = scalar_mass_matrix(M, V, phis, vac)
    show("mass^2 eigenvalues at the vacuum", M2.eigenvals())
    ok(f"{N - 1} massless Goldstone bosons (= dim O(N)/O(N-1))", M2.eigenvals().get(0, 0) == N - 1)

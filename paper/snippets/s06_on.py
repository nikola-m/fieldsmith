from fieldsmith import Model
N = 4
M = Model()
M.parameters("mu lambda", positive=True)
M.add_index_space("flavor", "pqrs", N)                  # the index family  p,q,r,s = 1..N
M.field("phi", ("flavor",), "")
L = M.lagrangian(r"\frac12\partial_\mu\phi_p\partial^\mu\phi_p + \frac{\mu^2}{2}\phi_p\phi_p"
                 r" - \frac{\lambda}{4}(\phi_p\phi_p)^2")
print(len(M.euler_lagrange(L)))                         # N field equations

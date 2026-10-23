# fieldsmith user guide

## 1. The mental model

A `Model` owns a spacetime, a registry of **fields**, **parameters**, **derived tensors** (like `F_{\mu\nu}`)
and operators. You declare these, then hand LaTeX strings to the model. Every string is parsed into an
explicit SymPy expression in the coordinate components (`t, x, y, z`), with repeated indices summed.
Because the result is a plain SymPy expression, you can always drop down to SymPy yourself.

```python
from fieldsmith import Model
M = Model()                           # 3+1 Minkowski, (+,-,-,-); Model(Spacetime.minkowski(3)) gives 2+1, etc.
m, lam = M.parameters("m lambda", positive=True)
phi = M.scalar("phi")                 # real scalar;  M.scalar("psi", complex=True) also defines psi*
A = M.vector("A")                     # A_mu (lower index by default; var="u" for A^mu)
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
```

## 2. Declaring things

| Call | Meaning |
|---|---|
| `M.parameters("m g_s", positive=True)` | constant symbols |
| `M.scalar(name, complex=False)` | scalar field |
| `M.vector(name, var="d")` | vector field with components `A[mu]` |
| `M.tensor(name, var="dd", symmetry=...)` | higher-rank field (`"sym"`, `"antisym"`) |
| `M.multiplet(name, n, rep=...)` | n-component complex multiplet (doublet, spinor, ...) |
| `M.define(lhs, rhs)` | derived tensor, e.g. field strength; usable in later strings |
| `M.add_index_space(kind, names, dim)` | custom index family (flavour, colour, ...) |
| `Model(Spacetime(coords, g))` | custom metrics: see `fieldsmith.spacetime.Spacetime` |

Index kinds are recognised by letter: Greek `\mu \nu \rho \sigma \lambda \kappa ...` are spacetime,
`i j k l m n` spatial, `a b c d e f` adjoint (group) indices.

## 3. The LaTeX subset

* Arithmetic: `+ - *`, juxtaposition, `\cdot`, `\,`, `/`, `^`, `\frac{a}{b}` and `\frac12`, `\sqrt{}`, `n!`,
  `\exp \log \sin \cos \tan \sinh \cosh \tanh`.
* Indices: `A_\mu`, `F^{\mu\nu}`, `T^a_{\mu\nu}`, numeric (`A_0`, `A^1_\mu`). A label repeated up/down
  (or twice on `sp`/`ad` indices) is **summed**.
* Operators act on the *next* postfix term: `\partial_\mu \phi^2` is `\partial_\mu(\phi^2)`;
  `\partial_\mu\phi\,\partial^\mu\phi` is a product. Parenthesise when unsure.
  Supported: `\partial_\mu`, `\nabla_\mu` (covariant, curved), `\Box`, `D_\mu` (gauge covariant), `\bar\psi`.
* Conjugation: `\phi^*`, `\phi^\dagger`, `\bar{\psi}`; accents `\tilde{F}` become new symbols (define them!).
* Built-ins: `g_{\mu\nu}`, `\eta_{\mu\nu}`, `\delta^\mu_\nu`, `\epsilon^{\mu\nu\rho\sigma}` (any rank; `\epsilon_{ijk}` spatial).
* `(T_\mu)^2` is the self-contraction `T_\mu T^\mu`.
* Not supported: `\sum`, `\int` (write indices instead), `\text{}` markup, `\left`/`\right` are skipped.

## 4. Derivations

```python
E = M.euler_lagrange(L)               # EquationSet; E.tensor("A") is E^nu, E[0] a raw expression
E.latex(); E.check_solution({phi: sol}); E.check_solution(..., numeric=True)
M.check(r"\partial_\nu\partial_\mu F^{\mu\nu} = 0")     # identity checker -> bool
M.evaluate(r"F_{01}")                                   # parse to a sympy value
```

**Back to index notation** — the most useful trick. Component output is unreadable; `fit` finds the
combination of covariant building blocks that reproduces it:

```python
from fieldsmith.analysis import fit
fit(M, E.tensor("A"), {"maxwell": r"\partial_\mu F^{\mu\nu}"})     # {'maxwell': -1}   (or None if no fit)
```

`fit` re-verifies the solution on every component, so `None` genuinely means "not of this form".

### Symmetries and conserved currents
```python
from fieldsmith.noether import (noether_current, complex_variation, symmetry_defect,
                                canonical_stress_tensor, hilbert_stress_tensor)
var = complex_variation(M, "Phi", r"i\Phi")        # delta Phi = i alpha Phi, delta Phi* = -i alpha Phi*
J = noether_current(M, L, var)                     # raises if not a symmetry (K defaults to 0)
J.tensor(); J.identity_holds(); J.conserved_on_shell(); J.charge_density()
Tc = canonical_stress_tensor(M, L)                 # T^mu_nu, Noether canonical
Th = hilbert_stress_tensor(M, r"-\frac14 F_{\mu\nu}F^{\mu\nu}")   # symmetric, gauge invariant (string only)
```
`variations` is a dict `{field_function: delta}`; for spacetime symmetries include the transport term
`-\xi^\mu\partial_\mu\phi` and pass a boundary term `K` when `\delta L = \partial_\mu K^\mu`.

### Canonical / constraint analysis
```python
from fieldsmith.hamiltonian import Hamiltonian
H = Hamiltonian(M, L)
H.hamilton_equations(); H.bracket(F, G)
res = H.dirac_analysis(); print(res.summary())     # n_first, n_second, dof
```
Requires at most first time derivatives, Lagrangian quadratic in velocities.

### Waves and spectra
```python
from fieldsmith.waves import linearize, dispersion
lin, pert = linearize(M, E, background={...})
d = dispersion(M, lin, list(pert.values()))        # d["det"], d["omega"], d["pdet"] for gauge theories
from fieldsmith.analysis import scalar_mass_matrix, energy_density
```

## 5. Gauge theory

```python
from fieldsmith.gauge import GaugeTheory, LieAlgebra
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")             # gauge field A^a_mu, coupling g, F^a_{mu nu}
G.add_matter("Phi", 2, {"A": "fund"})                # doublet; reps: "fund", "adj", a U(1) charge, or explicit matrices
M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu} + (D_\mu\Phi)^\dagger D^\mu\Phi - V")
```
Conventions: `D_\mu = \partial_\mu - i g A^a_\mu T^a`, `F^a = \partial A - \partial A + g f^{abc}A^b A^c`,
`\delta\phi = i\alpha T\phi`, `\delta A^a = \partial\alpha^a/g + f^{abc}A^b\alpha^c`.
`G.gauge_variation(...)` / `G.gauge_parameters` support infinitesimal gauge-invariance checks, and
`G.gauge_mass_matrix(L, vev)` gives `M^2_{ij} = \partial^2 L/\partial A_i^0\partial A_j^0` at the vacuum `vev`.
Several groups may be added for product groups such as SU(2)×U(1).

## 6. Curved spacetime

```python
from fieldsmith.spacetime import Spacetime
st = Spacetime(coords, g_matrix)       # st.christoffel(), st.riemann(), st.ricci(), st.einstein()
M = Model(st)                       # EL equations use sqrt|g| automatically; \nabla_\mu available
```

## 7. Pitfalls

* `A_\mu A_\mu` (two lower spacetime indices): by default the second copy is silently raised with the metric,
  which is convenient but hides typos. Use `Model(strict=True)` to make this an error and write `A_\mu A^\mu`.
  (Spatial `i,j,...` and adjoint `a,b,...` indices carry no variance and always sum plainly.)
* Derivative scope (see §3). `\partial_\mu\phi^2 \neq \partial_\mu\phi\,\partial_\mu\phi`.
* Numeric spacetime indices start at 0, adjoint indices at 1.
* A parameter named `g` shadows the metric tensor in expressions without indices; use `g_s`, `g'`, etc.
  (or just `\eta_{\mu\nu}` for the metric).
* SU(3) with matter gets large: prefer SU(2)/U(1) for classroom use, or restrict to what you need.
* For `hilbert_stress_tensor` write the Lagrangian with `F_{\mu\nu}`-type tensors and explicit `g^{\mu\nu}`;
  fields with Lorentz indices must be given with the variance they naturally carry.

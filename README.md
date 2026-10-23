# fieldsmith

**Symbolic classical field theory and gauge theory, driven by LaTeX.**

Write a Lagrangian the way you would on a blackboard, and `fieldsmith` derives, checks and analyses
the physics: field equations, Noether currents, stress tensors, canonical constraints, dispersion
relations, gauge invariance, mass spectra. It is built on SymPy and designed for teaching upper-level
undergraduates and early graduate students, and as a scaffold for research-grade calculations.

```python
from fieldsmith import Model
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import fit

M = Model()                                    # 3+1 Minkowski, signature (+,-,-,-)
G = GaugeTheory(M)
G.add_group("A", LieAlgebra.su(2), "g")        # SU(2) gauge field A^a_mu, strength F^a_{mu nu}

L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")             # parsed from LaTeX, indices summed

E = M.euler_lagrange(L)                                           # delta S / delta A^a_nu
fit(M, E.tensor("A"), {"DF": r"D_\mu F^{a\mu\nu}"})               # -> {'DF': 1}   (covariant Yang-Mills equation)
M.check(r"D_\mu(\frac12\epsilon^{\mu\nu\rho\sigma}F^a_{\rho\sigma}) = 0")   # -> True  (Bianchi identity)
```

Everything is computed on explicit components, so every result is *verified*, not pattern-matched:
index notation is parsed, expanded, and compared symbolically.

## Install

```bash
pip install -e .            # needs Python >= 3.9, sympy, mpmath
pip install -e ".[test]"    # + pytest
python -m pytest -m "not slow"      # ~1 minute: unit tests
python -m pytest                    # + runs every example script end to end
```

## What it can do

| Area | Capabilities |
|---|---|
| **Input** | LaTeX subset: `\partial_\mu`, `\nabla_\mu`, `D_\mu`, `\Box`, `\frac`, `\sqrt`, `\exp`..., Einstein summation, `\epsilon^{\mu\nu\rho\sigma}`, `\delta`, `g_{\mu\nu}`, `^\dagger`, `\bar\psi`, `\gamma^\mu`, `\tilde F`, operator-valued groups like `(i\gamma^\mu D_\mu - m)\psi` |
| **Spacetimes** | Minkowski in any dimension, Euclidean, arbitrary metrics (FRW, Schwarzschild, ...) with Christoffel/Riemann/Ricci/Einstein tensors and covariant derivatives |
| **Fields** | scalars (real/complex), vectors, rank-n tensors (symmetric/antisymmetric), multiplets (doublets, spinors), user-defined index families (colour, flavour) |
| **Variational calculus** | Euler-Lagrange equations to any derivative order, complex fields varied independently, curved backgrounds (`sqrt(-g)`), checking exact solutions symbolically or numerically |
| **Symmetries** | Noether currents of *any* continuous symmetry (internal, spacetime, with boundary terms, higher-derivative), automatic symmetry test, Noether's identity, canonical stress tensor, Hilbert (metric) stress tensor |
| **Gauge theory** | su(N)/u(1) algebras with structure constants, Yang-Mills `F^a_{mu nu}`, covariant derivatives on multiplets *and* adjoint indices, local gauge transformations, product groups, Higgs mass matrices |
| **Canonical analysis** | momenta, Hamiltonian, Hamilton's equations, local Poisson brackets, Dirac-Bergmann constraint algorithm with first/second-class classification and degree-of-freedom counting |
| **Waves** | linearisation about any background, plane-wave matrices, dispersion relations, physical modes in the presence of gauge null directions |
| **Teaching tools** | auto-grading of submitted field equations, identities and Noether currents (`fieldsmith.teaching`) |

## Verified results shipped as examples

Every script in [`examples/`](examples) runs and asserts its physics (see [docs/gallery.md](docs/gallery.md) for the output).

| # | Example | Level | Checks |
|---|---|---|---|
| 01 | Klein-Gordon field | elementary | EL equation, dispersion, energy density, Hamilton's equations |
| 02 | phi^4 kink & sine-Gordon | elementary | kink and breather exact solutions, kink energy `2*sqrt(2) m^3/(3 lambda)` |
| 03 | Maxwell | intermediate | covariant EOM, Bianchi, gauge invariance, `T^00 = (E^2+B^2)/2`, 2 dof |
| 04 | Proca & Stueckelberg | intermediate | 3 polarisations, second-class constraints, gauge invariance restored |
| 05 | Abelian Higgs | intermediate | minimal coupling, masses `M_A = e v`, `M_h^2 = 2 mu^2`, dof 4 -> 3+1 |
| 06 | O(N) Goldstone | intermediate | N-1 massless modes, currents, broken generators |
| 07 | Yang-Mills SU(2), SU(3) | advanced | gauge invariance, `D_mu F^{mu nu}`, Bianchi, `[D,D] = g f F` |
| 08 | Electroweak SU(2)xU(1) | graduate | `M_W`, `M_Z`, massless photon, `rho = 1`, Weinberg angle |
| 09 | QED with a Dirac field | graduate | Dirac equation from the action, current, 4 dof from 8 second-class constraints |
| 10 | Stress tensors | graduate | Belinfante relation `T_Hilb - T_can = d(F A) - EOM`, conformal improvement |
| 11 | Curved backgrounds | graduate | FRW Hubble friction, Friedmann equation, `nabla_mu T^{mu nu}` |
| 12 | Axion electrodynamics | research | `F Ftilde = d K`, cosmic birefringence `omega^2 = k^2 +/- g kappa k` |
| 13 | Vacuum birefringence | research | Euler-Heisenberg refractive indices, the 7:4 ratio, `n-1 = 14, 8 alpha^2 B^2/(45 m^4)` |
| 14 | Podolsky electrodynamics | research | 4th-order EOM, 2 massless + 3 massive modes, finite Coulomb potential |
| 15 | Topologically massive CS (2+1) | research | boundary term, mass `|mu|`, modified Gauss law, 1 dof |
| 16 | BPST instanton | research | exact self-duality and EOM, `S = 8 pi^2/g^2`, `Q = 1` |
| 17 | 't Hooft-Polyakov monopole | research | exact BPS solution, Bogomolny equations, `M = 4 pi v/g` |
| 18 | Symmetry tour | advanced | boosts, rotations, shifts, dilatations and their breaking |

## Conventions (all configurable or documented)

* Signature `(+,-,-,-)` by default; `Spacetime.minkowski(4, "mostly_plus")` flips it.
* `epsilon^{0123} = +1` (`Model(eps0123=-1)` for the other convention).
* Peskin & Schroeder gauge conventions: `D_mu = d_mu - i g A^a T^a`,
  `F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + g f^{abc} A^b_mu A^c_nu`.
* Coordinate components are 0-based (`A_0` is the time component); adjoint/colour numerals are
  1-based (`A^1_\mu`, ..., `A^3_\mu`) to match textbooks.
* `D_\mu` is registered once a `GaugeTheory` exists; derivative operators act on the *next* factor
  (`\partial_\mu\phi^2` means `d(phi^2)`) -- use parentheses to be explicit.

## Honest limitations

* **Component-based.** Fields are explicit functions of the coordinates, so cost grows with
  spacetime dimension and group size (SU(3) Yang-Mills has 1650 terms; SU(N>=4) gets slow).
* **Classical commuting fields only** (a Dirac field is treated as a commuting classical spinor).
* **Canonical module**: Lagrangians quadratic in first time derivatives; no Ostrogradsky
  construction for higher time derivatives (EL and Noether *do* support higher derivatives).
* **Hilbert tensor**: about flat space; use metric-independent building blocks (e.g. `F_{\mu\nu}`),
  bosonic fields only (no vierbein for spinors).
* **Gravity**: curvature tensors and covariant derivatives are provided, but there is no
  Einstein-Hilbert dynamics / metric as a dynamical field yet.
* **Dirac analysis** classifies constraints by their mutual brackets (strong, then weak by solving
  for a pivot); it is aimed at textbook and research-prototype theories, not arbitrary systems.
* No gauge fixing / ghost machinery (add gauge-fixing terms to the Lagrangian by hand).
* Gamma matrices are the 4D Dirac representation.

## Layout

```
fieldsmith/       spacetime, tensors, latex parser, model, variational, noether,
                  hamiltonian, waves, gauge, analysis, teaching, printing
examples/         18 runnable, self-checking scripts (elementary -> research)
docs/             guide.md  (API + LaTeX reference), teaching.md (course + exercises), gallery.md
paper/            arXiv-ready LaTeX companion paper (theory background + all 18 examples), main.pdf
tests/            unit tests + example runner
```

See [docs/guide.md](docs/guide.md) to get started and [docs/teaching.md](docs/teaching.md) for a
course outline with exercises and an auto-grading workflow.

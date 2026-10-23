# Teaching with fieldsmith

A suggested one-semester path. Each step lists an example script that already derives the result, and
an exercise students do themselves with the library as a calculator and checker.

| Week | Topic | Example | Exercise |
|---|---|---|---|
| 1 | Lagrangians, EL equations | 01 | Derive the EOM for `\frac12(\partial\phi)^2 - \frac{\lambda}{4!}\phi^4`; verify a given solution |
| 2 | Solitons | 02 | Show the sine-Gordon breather solves the EOM; compute its energy |
| 3 | Noether, stress tensors | 01, 18, 10 | Find the current for U(1) and for a shift symmetry; show dilatation is broken by a mass |
| 4 | Maxwell | 03 | Verify gauge invariance; show 2 dof by Dirac analysis |
| 5 | Massive vector, Stueckelberg | 04 | Count dof of Proca (3); restore gauge invariance |
| 6 | Spontaneous breaking | 05, 06 | Goldstone counting for O(N) -> O(N-1) |
| 7 | Yang-Mills | 07 | Show `[D_\mu,D_\nu] = g f F`; Bianchi identity for SU(2), SU(3) |
| 8 | Higgs mechanism | 05, 08 | Compute W/Z masses and the Weinberg angle; `\rho = 1` |
| 9 | Fermions | 09 | Derive Dirac eq. from the action; count dof |
| 10 | Energy-momentum, gravity coupling | 10, 11 | Belinfante relation; scalar in FRW |
| 11 | Topology | 15, 16, 17 | Verify the instanton action `8\pi^2/g^2`; Bogomolny equations |
| 12 | Research frontier | 12, 13, 14 | Birefringence; Podolsky modes; propose and analyse your own modification |

## Auto-grading homework

```python
from fieldsmith import Model
from fieldsmith.teaching import grade_field_equation, grade_identity, grade_conserved_current

M = Model(); M.parameters("m", positive=True); M.vector("A")
M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
L = r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu"

grade_field_equation(M, L, student_answer, "A")      # truthy / falsy Grade with a message
grade_identity(M, r"\partial_\nu\partial_\mu F^{\mu\nu} = 0")
grade_conserved_current(M, L_complex, variation, r"i(\phi\,\partial^\mu\phi^* - \phi^*\partial^\mu\phi)")
```
* A `Grade` is truthy when correct, and `.message` says why not (wrong sign of mass term, forgotten term,
  unparsable input).
* Overall factors are accepted by default (equations of motion are defined up to normalisation); pass
  `allow_overall_factor=False` to require an exact match.
* Students' LaTeX is parsed, never executed as Python; parse errors are reported as incorrect, not raised.

## Designing your own problems

1. State a Lagrangian in LaTeX; the key answer is `M.euler_lagrange(L)`.
2. Pose a "find the symmetry" problem: give variations, check with `symmetry_defect`.
3. Pose "count the degrees of freedom": `Hamiltonian(M, L).dirac_analysis().summary()`.
4. Pose "find the spectrum": `linearize` + `dispersion`, or `gauge_mass_matrix`.
5. Research-style project: modify a shipped example (add a term, change dimension or gauge group) and
   ask how each quantity changes. Examples 12-14 are written to be edited this way.

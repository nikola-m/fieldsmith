# Alakazam vs fieldsmith: step-level benchmark

Alakazam (https://arxiv.org/abs/2608.20452) is an open-source abstract-index tensor package written in Julia. It tends to be magnitudes faster than xAct and Cadabra in benchmarks. It also has built in support for LaTeX input and output (which should work for all types of script characters) as well as unicode, variational calculus and Euler-Lagrange equations, dedicated functionality for Grassmann numbers, and some basic differential forms (in index form). It can also be used to compute some basic amplitudes.

## What is compared
  18 tasks taken from fieldsmith's own example scripts: the index-level identities and covariant fits
  (field equations checked against their covariant form, Bianchi-type identities, gauge invariance,
  stress-tensor relations). Each task starts from the same Lagrangian on both sides and ends in the same
  final result (fit coefficients, True, or delta L = 0); compare.py checks that they agree.

  The tasks are the parts of fieldsmith's examples that Alakazam can also do. The rest of the examples
  (explicit solutions, energies, Hamiltonian and constraint analysis, dispersion relations, explicit
  metrics) is component-level work that Alakazam, an abstract-index package, does not do, so it is not
  part of this comparison. The claim this benchmark supports is about index-level identities and
  covariant fits only.

## Files
  bench_fieldsmith.py   fieldsmith side (Python)
  bench_alakazam.jl     Alakazam side (Julia)
  Project.toml          Julia environment for bench_alakazam.jl (Alakazam 1.0.7, Symbolics 7.49.0)
  compare.py            runs both, checks the results agree, prints the comparison

## Setup (once)
  pip install sympy mpmath "git+https://github.com/nikola-m/fieldsmith@82a7ca973b568b8d8e0c22fb7e8f31a9ac48cde9"
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  (The first instantiate downloads and precompiles Alakazam and Symbolics, which can take several
  minutes; later runs reuse the compiled packages.)
  (To benchmark another fieldsmith commit, change the hash in the pip line.)

## Run (from this folder)
  python compare.py 5                      # all scenarios, ~10 minutes
  python bench_fieldsmith.py 5             # one side only; --only=K runs task K alone
  julia --project=. bench_alakazam.jl 5

## Timing scenarios (compare.py)
  warm      best of runs 2..5 of each task, all tasks in one process per tool
  first     run 1 of each task in that process; for Julia it includes JIT compilation of whatever the
            task is first to use, so it depends on task order
  one-off   each task alone in a fresh process, from start to exit, including package load and
            compilation: the cost of running a single example script
  suite     all 18 tasks once in one fresh process, from start to exit
  load      a fresh process that only loads the packages
  Summaries are totals, plus the median and range of the per-task warm ratios.

## What runs on each side
  fieldsmith  its own example code. Exception: the two stress-tensor tasks (Belinfante relation,
              improved trace) assemble the canonical tensor dL/d(d_mu f) d_nu f - delta L and the
              Hilbert tensor 2 dL/dg^{mu nu} - g_{mu nu} L by hand from derivatives fieldsmith
              computes, because that is all the Alakazam side can do (Alakazam has no sqrt(-g)
              object). fieldsmith's canonical_stress_tensor and hilbert_stress_tensor would do more
              work than the Julia side, so they are not used.
  Alakazam    Alakazam plus about 100 lines of benchmark helper code in bench_alakazam.jl: zero_abs
              (tries Alakazam's simplify, hard_simplify and canonicalise), avary (first-order variation
              from Alakazam's VariationalDerivative), afit with cnum (covariant fit: matches the
              coefficients of Alakazam's canonical terms and solves the small linear system), fresh
              (unique dummy indices). The helpers are timed; they are not Alakazam features.

  On the Alakazam side, `false` from an identity or `nothing` from a fit means Alakazam did not reduce
  the expression, not that it proved it nonzero. All 18 tasks here are expected to succeed.

Differences between the two packages, by design
  - Alakazam works with abstract indices; fieldsmith expands everything into explicit components. For
    index-level identities that alone is worth orders of magnitude, whatever the language.
  - SU(2) Yang-Mills fit: a generic totally antisymmetric f^{abc} in Alakazam, the explicit su(2)
    structure constants in fieldsmith.
  - QED tasks: an abstract gamma^mu_{ij} symbol in Alakazam (no Clifford algebra is needed for these
    tasks), explicit 4x4 Dirac matrices in fieldsmith.
  - fieldsmith's times include parsing its LaTeX input; Alakazam's include building the expression.
  
  
Author of the benchmark: Jesse Woods, University of Bern (jesse.woods@unibe.ch)

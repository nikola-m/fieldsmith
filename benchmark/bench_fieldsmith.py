"""fieldsmith side of the Alakazam vs fieldsmith step benchmark (self-contained).

    pip install sympy mpmath "git+https://github.com/nikola-m/fieldsmith@82a7ca973b568b8d8e0c22fb7e8f31a9ac48cde9"
    python bench_fieldsmith.py 5          # 5 runs per task
    options: --json (machine-readable), --only=K (task K only, 1-based)

(Or point FIELDSMITH_PATH at a checkout of the fieldsmith repository instead of installing it.)

18 tasks taken from fieldsmith's example scripts (the index-level identities and covariant fits that
Alakazam can also do). Each starts from the same Lagrangian as its counterpart in bench_alakazam.jl and
returns the same final result (fit coefficients, True, or delta L). The code is fieldsmith's own example
code, except the two stress-tensor tasks, which assemble the same formulas by hand as the Julia side
(see canonical_by_hand and hilbert_by_hand). SymPy's cache is cleared before every run; "first" is run 1,
"warm" the best of runs 2..N.
"""
import copy, json, os, sys, time

if os.environ.get("FIELDSMITH_PATH"):
    sys.path.insert(0, os.environ["FIELDSMITH_PATH"])
import sympy as sp
from sympy.core.cache import clear_cache
from fieldsmith import Model, Spacetime
from fieldsmith.gauge import GaugeTheory, LieAlgebra
from fieldsmith.analysis import fit
from fieldsmith.noether import symmetry_defect, _momenta
from fieldsmith.tensors import Tensor, Slot

F_DEF = (r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")


def maxwell():
    M = Model()
    M.vector("A")
    M.define(*F_DEF)
    return M


def t01_eom():
    M = Model()
    M.parameters("m lambda", positive=True)
    M.scalar("phi")
    L = M.lagrangian(r"\frac12 \partial_\mu\phi\,\partial^\mu\phi - \frac12 m^2\phi^2")
    eom = M.euler_lagrange(L)
    box = M.parse(r"\Box\phi + m^2\phi")
    return sp.simplify(eom[0] + box) == 0


def t03_fit():
    M = maxwell()
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
    return fit(M, M.euler_lagrange(L).tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}"})


def t03_fit_source():
    M = maxwell()
    M.vector("J", var="u", dynamical=False)
    Ls = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} - J^\mu A_\mu")
    return fit(M, M.euler_lagrange(Ls).tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "J": r"J^\nu"})


def t03_bianchi():
    M = maxwell()
    return M.check(r"\partial_\mu F_{\nu\rho} + \partial_\nu F_{\rho\mu} + \partial_\rho F_{\mu\nu} = 0")


def t03_ddF():
    M = maxwell()
    return M.check(r"\partial_\nu\partial_\mu F^{\mu\nu} = 0")


def t03_gauge():
    M = maxwell()
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu}")
    alpha = sp.Function("alpha")(*M.coords)
    var = {M.tensors["A"].funcs[(m_,)]: sp.diff(alpha, M.coords[m_]) for m_ in range(4)}
    return symmetry_defect(M, L, var)


def t04_fit():
    M = maxwell()
    M.parameters("m", positive=True)
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu")
    return fit(M, M.euler_lagrange(L).tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "mA": r"m^2 A^\nu"})


def t04_stueckelberg():
    M2 = Model()
    m2 = M2.parameters("m", positive=True)
    M2.vector("A")
    M2.scalar("theta")
    M2.define(*F_DEF)
    Ls = M2.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 (m A_\mu - \partial_\mu\theta)(m A^\mu - \partial^\mu\theta)")
    alpha = sp.Function("alpha")(*M2.coords)
    var = {M2.tensors["A"].funcs[(i,)]: sp.diff(alpha, M2.coords[i]) for i in range(4)}
    var[M2.tensors["theta"].funcs[()]] = m2 * alpha
    return symmetry_defect(M2, Ls, var)


def t05_fit():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.u1(), "e")
    M.parameters("mu lambda", positive=True)
    G.add_matter("Phi", 1, {"A": 1}, complex=True, latex=r"\Phi")
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + (D_\mu\Phi)^\dagger (D^\mu\Phi) + \mu^2\Phi^\dagger\Phi - \lambda(\Phi^\dagger\Phi)^2")
    Et = M.euler_lagrange(L).tensor("A")
    return fit(M, Et, {"dF": r"\partial_\mu F^{\mu\nu}",
                       "j": r"i e (\Phi^\dagger D^\nu\Phi - (D^\nu\Phi)^\dagger\Phi)"})


def t07_fit():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.su(2), "g")
    L = M.lagrangian(r"-\frac14 F^a_{\mu\nu}F^{a\mu\nu}")
    E = M.euler_lagrange(L)
    return fit(M, E.tensor("A"), {"DF": r"D_\mu F^{a\mu\nu}"})


def qed():
    M = Model()
    G = GaugeTheory(M)
    G.add_group("A", LieAlgebra.u1(), "e")
    M.parameters("m", positive=True)
    G.add_matter("psi", 4, {"A": 1}, complex=True, spinor=True, latex=r"\psi")
    L = M.lagrangian(r"\bar\psi (i\gamma^\mu D_\mu - m)\psi - \frac14 F_{\mu\nu}F^{\mu\nu}")
    return M, G, L


def t09_gauge():
    M, G, L = qed()
    (alpha,) = G.gauge_parameters("A")
    return symmetry_defect(M, L, G.gauge_variation({"A": [alpha]}))


def t09_fit():
    M, G, L = qed()
    Et = M.euler_lagrange(L).tensor("A")
    return fit(M, Et, {"dF": r"\partial_\mu F^{\mu\nu}", "j": r"e \bar\psi\gamma^\nu\psi"})


# The two stress-tensor tasks assemble the same formulas by hand that the Julia side does (Alakazam has
# no sqrt(-g) object, so there the canonical tensor and the sqrt(-g) term of the Hilbert tensor are
# written out). Each library derives the pieces it can: dL/d(d_mu f) and dL/dg^{mu nu} from L.

def canonical_by_hand(M, L_latex):
    """T^mu_nu = dL/d(d_mu f) d_nu f - delta^mu_nu L, with dL/d(d_mu f) from fieldsmith (first-order L)."""
    L = sp.expand(M.lagrangian(L_latex))
    d, X = M.st.dim, M.coords
    fs = M.functions_in(L)
    P = {f: _momenta(M, L, f) for f in fs}
    data = {}
    for mu in range(d):
        for nu in range(d):
            v = sp.expand(sum(P[f].get((mu,), 0) * sp.diff(f, X[nu]) for f in fs) - (L if mu == nu else 0))
            if v != 0:
                data[(mu, nu)] = v
    rng = tuple(range(d))
    return Tensor(M, [Slot("mu", "st", "st", "u", rng), Slot("nu", "st", "st", "d", rng)], data)


def hilbert_by_hand(model, L_latex):
    """T_{mu nu} = 2 dL/dg^{mu nu} - g_{mu nu} L: L varied with respect to the metric by fieldsmith (as in
    its hilbert_stress_tensor), the sqrt(-g) term written out by hand."""
    st = model.st
    d = st.dim
    eta = st.g
    hs = {}
    H = sp.zeros(d, d)
    for i in range(d):
        for j in range(i, d):
            hs[(i, j)] = sp.Symbol(f"h_{i}{j}")
            H[i, j] = H[j, i] = hs[(i, j)]
    pst = Spacetime(st.coords, eta - eta * H * eta, ginv=st.ginv + H, sqrt_g=1, signature=st.signature, name="perturbed")
    m2 = copy.copy(model)
    m2.st = pst
    m2.tensors = dict(model.tensors)
    m2._install_builtins()
    Lp = sp.expand(m2.parse(L_latex))
    zero = {h: 0 for h in hs.values()}
    L0 = Lp.subs(zero)
    data = {}
    for (i, j), h in hs.items():
        c = sp.diff(Lp, h).subs(zero)
        if i != j:
            c = c / 2
        val = sp.expand(st.sign * (2 * c - eta[i, j] * L0))
        if val != 0:
            data[(i, j)] = val
            if i != j:
                data[(j, i)] = val
    rng = tuple(range(d))
    return Tensor(model, [Slot("mu", "st", "st", "d", rng), Slot("nu", "st", "st", "d", rng)], data)


def t10_belinfante():
    M = maxwell()
    LM = r"-\frac14 F_{\mu\nu}F^{\mu\nu}"
    Tc = canonical_by_hand(M, LM).raise_("nu")
    Th = hilbert_by_hand(M, LM).raise_("mu").raise_("nu")
    D = Th - Tc.relabel(["mu", "nu"])
    return fit(M, D, {"belinfante": r"\partial_\lambda(F^{\mu\lambda}A^\nu)", "eom": r"(\partial_\lambda F^{\mu\lambda})A^\nu"},
               labels=["mu", "nu"])


def t10_improved_trace():
    S = Model()
    S.scalar("phi")
    Ls = r"\frac12\partial_\mu\phi\partial^\mu\phi"
    T = hilbert_by_hand(S, Ls).raise_("mu").raise_("nu")
    phi = S.tensors["phi"].funcs[()]
    box = S.parse(r"\Box\phi")
    imp = S.parse(r"\partial^\mu\partial^\nu(\phi^2) - g^{\mu\nu}\Box(\phi^2)", as_tensor=True)
    Timp = T - imp.scale(sp.Rational(1, 6))
    trace_imp = sum(S.st.g[i, i] * Timp[(i, i)] for i in range(4))
    return sp.simplify(trace_imp - phi * box) == 0


def t12_axion_eq():
    M = Model()
    _, g = M.parameters("m_a g", positive=True)
    M.vector("A")
    M.scalar("a")
    M.define(*F_DEF)
    M.define(r"\tilde{F}^{\mu\nu}", r"\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}")
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12\partial_\mu a\partial^\mu a - \frac12 m_a^2 a^2 - \frac{g}{4} a F_{\mu\nu}\tilde F^{\mu\nu}")
    E = M.euler_lagrange(L)
    return sp.simplify(E["a"] + M.parse(r"\Box a + m_a^2 a") + g / 4 * M.parse(r"F_{\mu\nu}\tilde F^{\mu\nu}")) == 0


def podolsky():
    M = maxwell()
    M.parameters("a", positive=True)
    M.define(r"j_\nu", r"\partial^\mu F_{\mu\nu}")
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{a^2}{2} j_\nu j^\nu")
    return M, L


def t14_gauge():
    M, L = podolsky()
    alpha = sp.Function("alpha")(*M.coords)
    var = {M.tensors["A"].funcs[(i,)]: sp.diff(alpha, M.coords[i]) for i in range(4)}
    return symmetry_defect(M, L, var)


def t14_fit():
    M, L = podolsky()
    E = M.euler_lagrange(L)
    return fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "box": r"\Box\partial_\mu F^{\mu\nu}"})


def t15_fit():
    M = Model(Spacetime.minkowski(3))
    M.parameters("mu", positive=True)
    M.vector("A")
    M.define(*F_DEF)
    L = M.lagrangian(r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac{\mu}{4}\epsilon^{\mu\nu\rho}A_\mu F_{\nu\rho}")
    E = M.euler_lagrange(L)
    return fit(M, E.tensor("A"), {"dF": r"\partial_\mu F^{\mu\nu}", "cs": r"\epsilon^{\nu\rho\sigma}F_{\rho\sigma}"})


TASKS = [("01 Klein-Gordon field equation check", t01_eom),
         ("03 Maxwell covariant fit", t03_fit),
         ("03 Maxwell fit with source", t03_fit_source),
         ("03 Bianchi identity", t03_bianchi),
         ("03 d_nu d_mu F^{mu nu} = 0", t03_ddF),
         ("03 Maxwell gauge invariance", t03_gauge),
         ("04 Proca fit", t04_fit),
         ("04 Stueckelberg gauge invariance", t04_stueckelberg),
         ("05 Maxwell + scalar current fit", t05_fit),
         ("07 SU(2) Yang-Mills fit", t07_fit),
         ("09 QED gauge invariance", t09_gauge),
         ("09 Maxwell + Dirac current fit", t09_fit),
         ("10 Belinfante relation", t10_belinfante),
         ("10 Improved trace = phi Box phi", t10_improved_trace),
         ("12 Axion field equation", t12_axion_eq),
         ("14 Podolsky gauge invariance", t14_gauge),
         ("14 Podolsky fit", t14_fit),
         ("15 Chern-Simons fit", t15_fit)]

if __name__ == "__main__":
    as_json = "--json" in sys.argv
    only = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--only=")]
    nums = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = int(nums[0]) if nums else 5
    tasks = [TASKS[only[0] - 1]] if only else TASKS
    if not as_json:
        print(f"{'task':44s}{'result':30s}{'first (ms)':>12s}{'warm (ms)':>12s}")
    tot1 = totw = 0.0
    for name, fn in tasks:
        runs, res = [], None
        for _ in range(reps):
            clear_cache()
            t = time.perf_counter()
            res = fn()
            runs.append(time.perf_counter() - t)
        w = min(runs[1:]) if reps > 1 else float("nan")
        tot1 += runs[0]
        totw += w
        if as_json:
            print("RESULT " + json.dumps({"task": name, "runs": runs, "result": str(res)}), flush=True)
        else:
            print(f"{name:44s}{str(res):30s}{1e3 * runs[0]:12.2f}{1e3 * w:12.2f}", flush=True)
    if not as_json:
        print(f"{'TOTAL':74s}{1e3 * tot1:12.1f}{1e3 * totw:12.1f}")

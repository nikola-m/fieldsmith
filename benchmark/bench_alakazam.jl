# Alakazam side of the Alakazam vs fieldsmith step benchmark (self-contained).
#
#   julia --project=. -e "using Pkg; Pkg.instantiate()"      # once
#   julia --project=. bench_alakazam.jl 5                      # 5 runs per task
#   options: --json (machine-readable), --only=K (task K only, 1-based)
#
# 18 tasks taken from fieldsmith's example scripts: the index-level identities and covariant fits, done
# here with Alakazam plus about 100 lines of benchmark helper code (zero_abs, fresh, avary, cnum, afit
# below; they are not Alakazam features). Each starts from the same Lagrangian as its counterpart in
# bench_fieldsmith.py and returns the same final result (fit coefficients, true, or delta L = 0).
#
# Alakazam has no sqrt(-g) object, so the two stress-tensor tasks assemble the canonical tensor
# dL/d(d_mu A) d_nu A - g L and the sqrt(-g) term of the Hilbert tensor 2 dL/dg - g L by hand, from
# derivatives Alakazam computes; bench_fieldsmith.py assembles the same formulas by hand too.
#
# A result of `false` (identities) or `nothing` (fits) means Alakazam did not reduce the expression, not
# that it proved it nonzero. "first" is run 1 (it includes Julia's JIT compilation); "warm" is the best of
# runs 2..N.
using Alakazam, Symbolics
const SU = Symbolics.SymbolicUtils

# ---------------------------------------------------------------- helpers
"""True if one of Alakazam's simplification entry points reduces the expression to zero."""
function zero_abs(e)
    e = apply_derivative(TensorExpression(e))
    isZeroTens(e) && return true
    for f in (simplify, hard_simplify, canonicalise)
        r = try
            f(e)
        catch
            continue
        end
        isZeroTens(r) && return true
    end
    false
end

const _FRESH = Ref(0)
"""A fresh index of `set` (unique name), for dummies inside definitions such as F^a_{μν}."""
fresh(set, base="i") = (_FRESH[] += 1; Index(base * string(_FRESH[]), set))

"""
First-order change δL = Σ_f [∂L/∂f δf + ∂L/∂(∂_τ f) ∂_τ δf (+ ∂L/∂(∂_σ∂_τ f) ∂_σ∂_τ δf)], from Alakazam's
VariationalDerivative. `vars` holds (field, δfield); δfield carries the field's free indices.
"""
function avary(L, vars, ∂, dset; order=1)
    L = TensorExpression(L)
    out = zero()
    for (f, δf) in vars
        out = out + apply_derivative(VariationalDerivative(L, f)) * δf
        τ = fresh(dset, "τ")
        P1 = apply_derivative(VariationalDerivative(L, ∂(τ => lower, f)))
        isZeroTens(P1) || (out = out + P1 * apply_derivative(∂(τ => lower, δf)))
        if order >= 2
            σ = fresh(dset, "σ")
            P2 = apply_derivative(VariationalDerivative(L, ∂(σ => lower, ∂(τ => lower, f))))
            isZeroTens(P2) || (out = out + P2 * apply_derivative(∂(σ => lower, ∂(τ => lower, δf))))
        end
    end
    simplify(apply_derivative(out))
end

# Alakazam coefficients may carry complex constants (i e ...); they are written a + b·𝕚 with a symbol 𝕚.
# Each coefficient is linear in 𝕚 because SymbolicUtils folds numeric constants before we see them.
const 𝕚 = Symbolics.variable(:𝕚)
cnum(x::Num) = x
cnum(x::Complex) = cnum(real(x)) + 𝕚 * cnum(imag(x))
cnum(x::Number) = Num(x)
cnum(x) = Symbolics.wrap(_realtree(x))
function _realtree(x)
    x isa SU.BasicSymbolic || return x isa Complex ? Symbolics.unwrap(cnum(x)) : x
    if SU.isconst(x)
        v = SU.unwrap_const(x)
        return v isa Complex ? Symbolics.unwrap(cnum(v)) : x
    end
    SU.iscall(x) || return x
    op = SU.operation(x)
    args = map(_realtree, SU.arguments(x))
    op === real && return args[1]
    op === imag && return 0
    op(args...)
end
simp(x) = Symbolics.simplify_fractions(Symbolics.expand(x))
isz(x) = (u = Symbolics.unwrap(simp(x)); u isa Number ? iszero(u) : (SU.isconst(u) && iszero(SU.unwrap_const(u))))

"""
Covariant fit: coefficients c_i with E = Σ c_i B_i, from the coefficients of the canonical terms of
E - Σ c_i B_i (Alakazam `simplify`), solved as a small linear system. Returns Dict(name => c) or nothing.
"""
function afit(E, basis::Vector)
    cs = [Symbolics.variable(Symbol("c_fit", k)) for k in eachindex(basis)]
    R = apply_derivative(TensorExpression(E))
    for (c, (_, B)) in zip(cs, basis)
        R = R - Symbolics.unwrap(c) * apply_derivative(TensorExpression(B))
    end
    R = simplify(apply_derivative(R))
    isZeroTens(R) && return nothing
    eqs = [cnum(c) for (c, _) in summands(R)]
    n = length(cs)
    M = Matrix{Num}(undef, length(eqs), n + 1)
    for (i, e) in enumerate(eqs)
        for (j, c) in enumerate(cs)
            M[i, j] = simp(Symbolics.derivative(e, c))
        end
        M[i, n+1] = -simp(Symbolics.substitute(e, Dict(c => 0 for c in cs)))
    end
    r = 0
    for c in 1:n                         # Gaussian elimination over rational functions
        p = findfirst(i -> i > r && !isz(M[i, c]), 1:size(M, 1))
        p === nothing && return nothing
        r += 1
        M[[r, p], :] = M[[p, r], :]
        M[r, :] = simp.(M[r, :] ./ M[r, c])
        for i in 1:size(M, 1)
            i == r || (M[i, :] = simp.(M[i, :] .- M[i, c] .* M[r, :]))
        end
    end
    all(i -> isz(M[i, end]), r+1:size(M, 1)) || return nothing
    Dict(name => simp(M[k, end]) for (k, (name, _)) in enumerate(basis))
end

# ---------------------------------------------------------------- tasks
function az_setup(d=4)
    clear_global!()
    st = IndexSet("spacetime", (1, d - 1), lower)
    idx = Index(["μ", "ν", "ρ", "σ", "τ", "λ", "κ", "α", "β", "ζ"], st)
    x = Coordinate("x", st)
    ∂(i, T) = PD("∂", T, x, i)
    (; st, idx, x, ∂)
end

# vector potential, field strength and its substitution F -> ∂A - ∂A
function maxwell(s)
    ∂ = s.∂
    A(i) = Tensor("A", [i], s.x)
    F(i, j) = AntisymmetricTensor("F", [i, j], s.x)
    pa, pb = DummyPatternIndex(["a", "b"], s.st)
    Fdn = ∂(pa => lower, A(pb => lower)) - ∂(pb => lower, A(pa => lower))
    Fup = ∂(pa => upper, A(pb => upper)) - ∂(pb => upper, A(pa => upper))
    defF(ex) = simplify(replace_pattern(replace_pattern(TensorExpression(ex), "F_{?a?b}", Fdn), "F^{?a?b}", Fup))
    (; A, F, defF)
end
fitstr(d) = d === nothing ? "nothing" : "{" * join(["$k: $(d[k])" for k in sort(collect(keys(d)))], ", ") * "}"
zerostr(e) = zero_abs(e) ? "0" : "not reduced by Alakazam"

function t01_eom()
    s = az_setup(); μ = s.idx[1]; ∂ = s.∂
    @syms m::Real
    φ = Tensor("φ", s.x)
    L = 1//2 * ∂(μ => lower, φ) * ∂(μ => upper, φ) - 1//2 * m^2 * φ * φ
    E = FD(L, φ)
    string(zero_abs(E + ∂(μ => lower, ∂(μ => upper, φ)) + m^2 * φ))
end

function t03_fit()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower))
    fitstr(afit(FD(L, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper)))]))
end

function t03_fit_source()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    J(i) = Tensor("J", [i], s.x)
    Ls = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower)) - J(μ => upper) * w.A(μ => lower)
    fitstr(afit(FD(Ls, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper))), "J" => J(ν => upper)]))
end

function t03_bianchi()
    s = az_setup(); μ, ν, ρ = s.idx[1:3]; ∂ = s.∂; w = maxwell(s)
    string(zero_abs(w.defF(∂(μ => lower, w.F(ν => lower, ρ => lower)) + ∂(ν => lower, w.F(ρ => lower, μ => lower)) +
                           ∂(ρ => lower, w.F(μ => lower, ν => lower)))))
end

function t03_ddF()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    string(zero_abs(w.defF(∂(ν => lower, ∂(μ => lower, w.F(μ => upper, ν => upper))))))
end

function t03_gauge()
    s = az_setup(); μ, ν, κ = s.idx[1], s.idx[2], s.idx[7]; ∂ = s.∂; w = maxwell(s)
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower))
    α = Tensor("α", s.x)
    zerostr(avary(L, [(w.A(κ => lower), ∂(κ => lower, α))], ∂, s.st))
end

function t04_fit()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    @syms m::Real
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) + 1//2 * m^2 * w.A(μ => lower) * w.A(μ => upper))
    fitstr(afit(FD(L, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper))), "mA" => m^2 * w.A(ν => upper)]))
end

function t04_stueckelberg()
    s = az_setup(); μ, ν, κ = s.idx[1], s.idx[2], s.idx[7]; ∂ = s.∂; w = maxwell(s)
    @syms m::Real
    θ = Tensor("θ", s.x)
    α = Tensor("α", s.x)
    mA(i) = m * w.A(i) - ∂(i, θ)
    Ls = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) + 1//2 * mA(μ => lower) * mA(μ => upper))
    zerostr(avary(Ls, [(w.A(κ => lower), ∂(κ => lower, α)), (θ, m * α)], ∂, s.st))
end

function t05_fit()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    @syms e::Real mu::Real lam::Real
    Φ = Tensor("Φ", s.x)
    Φb = Tensor("Φbar", s.x)
    DΦ(i) = ∂(i, Φ) - im * e * w.A(i) * Φ
    DΦb(i) = ∂(i, Φb) + im * e * w.A(i) * Φb
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) + DΦb(μ => lower) * DΦ(μ => upper) +
               mu^2 * Φb * Φ - lam * Φb * Φ * Φb * Φ)
    j = im * e * (Φb * DΦ(ν => upper) - DΦb(ν => upper) * Φ)
    fitstr(afit(FD(L, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper))), "j" => j]))
end

function t07_fit()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂
    col = IndexSet("colour", (0, 3), lower)
    @syms g::Real
    A(c, i) = Tensor("A", [c, i], s.x)
    f(i, j, k) = AntisymmetricTensor("f", [i, j, k])
    function F(c, i, j)
        local b = fresh(col, "b"); local d = fresh(col, "d")
        ∂(i, A(c, j)) - ∂(j, A(c, i)) + g * f(c, b => upper, d => upper) * A(b => lower, i) * A(d => lower, j)
    end
    function Dadj(i, Y, c)
        local b = fresh(col, "b"); local d = fresh(col, "d")
        apply_derivative(∂(i, Y(c))) + g * f(c, b => upper, d => upper) * A(b => lower, i) * Y(d => lower)
    end
    a = Index("a", col)
    d = Index("d", col)
    L = simplify(-1//4 * F(a => upper, μ => lower, ν => lower) * F(a => lower, μ => upper, ν => upper))
    E = FD(L, A(d => lower, ν => lower))
    fitstr(afit(E, ["DF" => Dadj(μ => lower, c -> F(c, μ => upper, ν => upper), d => upper)]))
end

function qed()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    sp = IndexSet("spinor", (0, 4), lower)
    @syms e::Real m::Real
    ψ(i) = Tensor("ψ", [i], s.x)
    ψb(i) = Tensor("ψbar", [i], s.x)
    γ(mm, i, j) = Tensor("γ", [mm, i, j])
    function Dirac(i)
        local j = fresh(sp, "j"); local n = fresh(s.st, "n")
        im * γ(n => upper, i, j => lower) * (∂(n => lower, ψ(j => upper)) - im * e * w.A(n => lower) * ψ(j => upper)) - m * ψ(i)
    end
    k = Index("k", sp)
    q = Index("q", sp)
    L = w.defF(ψb(k => lower) * Dirac(k => upper) - 1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower))
    (; s, w, sp, ψ, ψb, γ, k, q, L, e)
end

function t09_gauge()
    Q = qed(); s = Q.s; κ = s.idx[7]; ∂ = s.∂; e = Q.e
    α = Tensor("α", s.x)
    zerostr(avary(Q.L, [(Q.ψ(Q.k => upper), im * α * Q.ψ(Q.k => upper)), (Q.ψb(Q.k => lower), -im * α * Q.ψb(Q.k => lower)),
                        (Q.w.A(κ => lower), 1 / e * ∂(κ => lower, α))], ∂, s.st))
end

function t09_fit()
    Q = qed(); s = Q.s; μ, ν = s.idx[1:2]; ∂ = s.∂; w = Q.w
    fitstr(afit(FD(Q.L, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper))),
                                          "j" => Q.e * Q.ψb(Q.k => lower) * Q.γ(ν => upper, Q.k => upper, Q.q => lower) * Q.ψ(Q.q => upper)]))
end

function t10_belinfante()
    s = az_setup(); μ, ν, ρ, σ, λ, κ, α_, β_ = s.idx[[1, 2, 3, 4, 6, 7, 8, 9]]; ∂ = s.∂; w = maxwell(s)
    g(i, j) = Metric([i, j])
    Lraw = -1//4 * g(μ => upper, ρ => upper) * g(ν => upper, σ => upper) * w.F(μ => lower, ν => lower) * w.F(ρ => lower, σ => lower)
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower))
    P = simplify(apply_derivative(VariationalDerivative(L, ∂(α_ => lower, w.A(κ => lower)))))
    Tc = simplify(P * ∂(β_ => upper, w.A(κ => lower)) - g(α_ => upper, β_ => upper) * L)
    Pg = w.defF(apply_derivative(VariationalDerivative(Lraw, g(κ => upper, λ => upper))))
    Th = simplify(contract_metrics(g(α_ => upper, κ => upper) * g(β_ => upper, λ => upper) * (2 * Pg - g(κ => lower, λ => lower) * L)))
    fitstr(afit(Th - Tc, ["belinfante" => w.defF(∂(λ => lower, w.F(α_ => upper, λ => upper) * w.A(β_ => upper))),
                          "eom" => w.defF(∂(λ => lower, w.F(α_ => upper, λ => upper))) * w.A(β_ => upper)]))
end

function t10_improved_trace()
    s = az_setup(); μ, ν, ρ, λ, κ, α2, β2 = s.idx[[1, 2, 3, 6, 7, 8, 9]]; ∂ = s.∂
    φ = Tensor("φ", s.x)
    g2(i, j) = Metric([i, j])
    Lsr = 1//2 * g2(μ => upper, ν => upper) * ∂(μ => lower, φ) * ∂(ν => lower, φ)
    Ls = simplify(Lsr)
    Pg = simplify(apply_derivative(VariationalDerivative(Lsr, g2(κ => upper, λ => upper))))
    T = simplify(contract_metrics(g2(α2 => upper, κ => upper) * g2(β2 => upper, λ => upper) * (2 * Pg - g2(κ => lower, λ => lower) * Ls)))
    imp = apply_derivative(∂(α2 => upper, ∂(β2 => upper, φ * φ))) - g2(α2 => upper, β2 => upper) * apply_derivative(∂(ρ => lower, ∂(ρ => upper, φ * φ)))
    trimp = simplify(contract_metrics(g2(α2 => lower, β2 => lower) * (T - 1//6 * imp)))
    string(zero_abs(trimp - φ * ∂(ρ => lower, ∂(ρ => upper, φ))))
end

function t12_axion_eq()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    @syms m_a::Real g::Real
    ax = Tensor("a", s.x)
    Ft(i, j) = (local r = fresh(s.st, "r"); local q = fresh(s.st, "q");
                1//2 * EpsilonTensor([i, j, r => upper, q => upper]) * w.F(r => lower, q => lower))
    FFt() = (local i = fresh(s.st, "i"); local j = fresh(s.st, "j"); w.defF(w.F(i => lower, j => lower) * Ft(i => upper, j => upper)))
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) + 1//2 * ∂(μ => lower, ax) * ∂(μ => upper, ax) -
               1//2 * m_a^2 * ax * ax - g / 4 * ax * FFt())
    Ea = FD(L, ax)
    string(zero_abs(Ea + ∂(μ => lower, ∂(μ => upper, ax)) + m_a^2 * ax + g / 4 * FFt()))
end

function podolsky()
    s = az_setup(); μ, ν = s.idx[1:2]; ∂ = s.∂; w = maxwell(s)
    @syms a::Real
    m1, m2 = fresh(s.st, "m"), fresh(s.st, "m")
    jj = ∂(m1 => upper, w.F(m1 => lower, ν => lower)) * ∂(m2 => lower, w.F(m2 => upper, ν => upper))
    L = simplify(apply_derivative(w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) + a^2 / 2 * jj)))
    (; s, w, L)
end

function t14_gauge()
    P = podolsky(); s = P.s; κ = s.idx[7]; ∂ = s.∂
    α = Tensor("α", s.x)
    zerostr(avary(P.L, [(P.w.A(κ => lower), ∂(κ => lower, α))], ∂, s.st; order=2))
end

function t14_fit()
    P = podolsky(); s = P.s; μ, ν, ρ = s.idx[1:3]; ∂ = s.∂; w = P.w
    dF = w.defF(∂(μ => lower, w.F(μ => upper, ν => upper)))
    fitstr(afit(FD(P.L, w.A(ν => lower)), ["dF" => dF, "box" => apply_derivative(∂(ρ => lower, ∂(ρ => upper, dF)))]))
end

function t15_fit()
    s = az_setup(3); μ, ν, ρ, σ = s.idx[1:4]; ∂ = s.∂; w = maxwell(s)
    @syms mu::Real
    ε3(i, j, k) = EpsilonTensor([i, j, k])
    L = w.defF(-1//4 * w.F(μ => upper, ν => upper) * w.F(μ => lower, ν => lower) +
               mu / 4 * ε3(μ => upper, ν => upper, ρ => upper) * w.A(μ => lower) * w.F(ν => lower, ρ => lower))
    fitstr(afit(FD(L, w.A(ν => lower)), ["dF" => w.defF(∂(μ => lower, w.F(μ => upper, ν => upper))),
                                        "cs" => w.defF(ε3(ν => upper, ρ => upper, σ => upper) * w.F(ρ => lower, σ => lower))]))
end

const TASKS = [("01 Klein-Gordon field equation check", t01_eom),
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

jstr(s) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"") * "\""
function main(args)
    json = "--json" in args
    only = [parse(Int, split(a, "=")[2]) for a in args if startswith(a, "--only=")]
    nums = filter(a -> !startswith(a, "--"), args)
    reps = isempty(nums) ? 5 : parse(Int, nums[1])
    tasks = isempty(only) ? TASKS : [TASKS[only[1]]]
    json || println(rpad("task", 44), rpad("result", 30), lpad("first (ms)", 12), lpad("warm (ms)", 12))
    tot1 = totw = 0.0
    for (name, fn) in tasks
        runs = Float64[]
        res = ""
        for _ in 1:reps
            t0 = time_ns()
            res = fn()
            push!(runs, (time_ns() - t0) / 1e9)
        end
        w = reps > 1 ? minimum(runs[2:end]) : NaN
        tot1 += runs[1]
        totw += w
        if json
            println("RESULT {\"task\":", jstr(name), ",\"runs\":[", join(runs, ","), "],\"result\":", jstr(res), "}")
        else
            println(rpad(name, 44), rpad(res, 30), lpad(round(1e3 * runs[1], digits=2), 12), lpad(round(1e3 * w, digits=2), 12))
        end
        flush(stdout)
    end
    json || println(rpad("TOTAL", 74), lpad(round(1e3 * tot1, digits=1), 12), lpad(round(1e3 * totw, digits=1), 12))
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

"""LaTeX output for field expressions and tensors."""
from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef

from .tensors import Tensor


def _symbolize(model, expr):
    """Replace field functions / derivatives by Symbols with LaTeX-friendly names."""
    expr = sp.sympify(expr)
    rep = {}
    names = model.latex_names
    coord_tex = {c: c.name for c in model.coords}
    for d in sorted(expr.atoms(sp.Derivative), key=lambda x: -sum(c for _, c in x.variable_count)):
        f = d.expr
        if not isinstance(f, AppliedUndef):
            continue
        base = names.get(f.func.__name__, f.func.__name__)
        ops = "".join(
            r"\partial_{%s}%s" % (coord_tex.get(v, v), f"^{{{c}}}" if c > 1 else "")
            for v, c in d.variable_count)
        rep[d] = sp.Symbol(r"\left(%s %s\right)" % (ops, base))
    expr = expr.xreplace(rep)
    rep = {}
    for f in expr.atoms(AppliedUndef):
        rep[f] = sp.Symbol(names.get(f.func.__name__, f.func.__name__))
    return expr.xreplace(rep)


def latex(model, obj):
    if isinstance(obj, Tensor):
        lines = []
        for k, v in sorted(obj.data.items()):
            lines.append(f"{k}: {latex(model, v)}")
        return "\n".join(lines) if lines else "0"
    if isinstance(obj, sp.MatrixBase):
        return sp.latex(obj.applyfunc(lambda e: _symbolize(model, e)))
    return sp.latex(_symbolize(model, obj))

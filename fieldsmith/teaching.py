"""Helpers for assigning and auto-grading classical-field-theory homework.

Typical use: the instructor fixes a model and a Lagrangian; students submit LaTeX for the field
equation, a conserved current, a stress tensor or an identity. These functions answer
"is the submitted expression *equivalent* to the correct one?" symbolically.

    >>> grade_field_equation(M, L, r"\\partial_\\mu F^{\\mu\\nu} = 0", field="A")
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .analysis import fit
from .tensors import Tensor


@dataclass
class Grade:
    correct: bool
    message: str
    factor: object = None

    def __bool__(self):
        return self.correct

    def __repr__(self):
        return f"{'CORRECT' if self.correct else 'INCORRECT'}: {self.message}"


def _side(model, text):
    t = model.parse(text, as_tensor=True)
    return t if isinstance(t, Tensor) else Tensor.scalar(model, t)


def grade_identity(model, text) -> Grade:
    """Grade a claimed identity ``lhs = rhs`` written in LaTeX."""
    try:
        ok = model.check(text)
    except Exception as exc:              # parse / index errors are reported, not raised
        return Grade(False, f"could not evaluate: {exc}")
    return Grade(ok, "identity holds" if ok else "the two sides are not equal")


def grade_field_equation(model, L, submitted, field, allow_overall_factor=True, labels=None) -> Grade:
    """Compare a submitted field equation to the Euler-Lagrange equation of ``L`` for ``field``.

    ``submitted`` is LaTeX of either ``expr = 0``, ``lhs = rhs`` or just ``expr``; the free indices
    must match the field's (e.g. ``\\nu`` for ``A_\\nu``). A nonzero overall factor (and sign) is
    accepted by default, since equations of motion are only defined up to normalisation.
    """
    try:
        E = model.euler_lagrange(L).tensor(field)
        if "=" in submitted:
            lhs, rhs = submitted.split("=", 1)
            sub = _side(model, lhs) - _side(model, rhs)
        else:
            sub = _side(model, submitted)
        labs = labels or sub.labels
        res = fit(model, E, {"s": sub}, labels=labs)
    except Exception as exc:
        return Grade(False, f"could not evaluate: {exc}")
    if res is None:
        return Grade(False, "not equivalent to the Euler-Lagrange equation of this Lagrangian")
    c = res["s"]
    if c == 0:
        return Grade(False, "the submitted expression is identically zero")
    if not allow_overall_factor and c != 1:
        return Grade(False, f"correct up to the factor {c}, but an exact match was requested", c)
    return Grade(True, "equivalent to the Euler-Lagrange equation" + ("" if c == 1 else f" (up to factor {1 / c})"), c)


def grade_conserved_current(model, L, variations, submitted, K=None) -> Grade:
    """Grade a submitted Noether current ``J^\\mu`` (LaTeX, free upper index) for a symmetry."""
    from .noether import noether_current
    try:
        nc = noether_current(model, L, variations, K)
        sub = model.parse(submitted, as_tensor=True)
        res = fit(model, nc.tensor(), {"s": sub}, labels=sub.labels)
    except Exception as exc:
        return Grade(False, f"could not evaluate: {exc}")
    if res is None or res["s"] == 0:
        return Grade(False, "not proportional to the Noether current of this symmetry")
    return Grade(True, "matches the Noether current" + ("" if res["s"] == 1 else f" (factor {res['s']})"), res["s"])

from fieldsmith import Model
from fieldsmith.teaching import grade_field_equation, grade_identity, grade_conserved_current
from fieldsmith.noether import complex_variation


def _maxwell():
    M = Model()
    M.parameters("m", positive=True)
    M.vector("A")
    M.define(r"F_{\mu\nu}", r"\partial_\mu A_\nu - \partial_\nu A_\mu")
    return M


def test_grading_field_equations():
    M = _maxwell()
    L = r"-\frac14 F_{\mu\nu}F^{\mu\nu} + \frac12 m^2 A_\mu A^\mu"
    assert grade_field_equation(M, L, r"\partial_\mu F^{\mu\nu} + m^2 A^\nu = 0", "A")
    assert grade_field_equation(M, L, r"-\partial_\mu F^{\mu\nu} = m^2 A^\nu", "A")          # sign/normalisation
    assert not grade_field_equation(M, L, r"\partial_\mu F^{\mu\nu} - m^2 A^\nu = 0", "A")    # wrong mass sign
    assert not grade_field_equation(M, L, r"\partial_\mu F^{\mu\nu} = 0", "A")                # forgot the mass term
    assert grade_field_equation(M, L, r"\partial_\mu F^{\mu\nu} + m^2 A^\nu = 0", "A", allow_overall_factor=False)
    assert not grade_field_equation(M, L, r"-\partial_\mu F^{\mu\nu} = m^2 A^\nu", "A", allow_overall_factor=False)


def test_grading_identities_and_currents():
    M = _maxwell()
    assert grade_identity(M, r"\partial_\nu\partial_\mu F^{\mu\nu} = 0")
    assert not grade_identity(M, r"\partial_\mu F^{\mu\nu} = 0")
    S = Model()
    S.parameters("m", positive=True)
    S.scalar("phi", complex=True)
    L = r"\partial_\mu\phi^*\partial^\mu\phi - m^2\phi^*\phi"
    var = complex_variation(S, "phi", r"i\phi")
    assert grade_conserved_current(S, L, var, r"i(\phi\,\partial^\mu\phi^* - \phi^*\partial^\mu\phi)")
    assert not grade_conserved_current(S, L, var, r"i(\phi\,\partial^\mu\phi^* + \phi^*\partial^\mu\phi)")

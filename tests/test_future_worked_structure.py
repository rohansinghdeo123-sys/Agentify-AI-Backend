"""Questions and descriptive examples must not be mislabeled solved work."""
from services.notes_material_service import _is_worked_solution


def test_unsolved_question_remains_a_question():
    assert not _is_worked_solution('Given m = 2 kg and F = 6 N, find acceleration.')
    assert not _is_worked_solution('Solve x² − 5x + 6 = 0.')
    assert not _is_worked_solution('A book at rest has balanced upward and downward forces.')


def test_complete_source_solutions_are_recognized_without_changing_them():
    assert _is_worked_solution('Given m = 2 kg and F = 6 N, use a = F/m = 6/2 = 3 m s⁻².')
    assert _is_worked_solution('Given x² − 5x + 6 = 0, factor to (x − 2)(x − 3) = 0. The roots are 2 and 3.')

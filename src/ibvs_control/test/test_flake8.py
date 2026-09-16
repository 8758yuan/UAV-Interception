import pytest
from ament_flake8.main import main_with_errors


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    """Check Python source style."""
    rc, errors = main_with_errors(argv=[])
    assert rc == 0, 'Found %d code style errors:\n' % len(errors) + '\n'.join(
        errors
    )

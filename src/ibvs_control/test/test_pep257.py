import pytest
from ament_pep257.main import main


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257() -> None:
    """Check Python docstring style."""
    assert main(argv=['.']) == 0

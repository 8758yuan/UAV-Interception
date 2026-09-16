from pathlib import Path


def test_source_files_do_not_contain_todo_copyright() -> None:
    """Keep the initial package free of generated TODO copyright headers."""
    package_dir = Path(__file__).parents[1] / 'ibvs_control'

    for source_file in package_dir.glob('*.py'):
        assert 'TODO: Copyright' not in source_file.read_text()

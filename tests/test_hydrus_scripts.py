import pytest

from hydrus_scripts import TagChanger


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("t1\nt2\nt3\n\nt4\nt5", {"t1": {"t2", "t3"}, "t4": {"t5"}}),
        ("t1\nt2\nt3  \n  \n  t4  \n  t5", {"t1": {"t2", "t3"}, "t4": {"t5"}}),
        ("t1\nt2\nt3\n\n\nt4\nt5", {"t1": {"t2", "t3"}, "t4": {"t5"}}),
    ],
)
def test_text_to_dict(test_input, expected):
    assert TagChanger.text_to_dict(test_input) == expected

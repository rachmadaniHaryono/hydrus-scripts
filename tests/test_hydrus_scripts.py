import pytest

import hydrus_scripts as hs
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


@pytest.mark.vcr()
@pytest.mark.golden_test("data/test_get_4chan_archive_data*.yaml")
def test_get_4chan_archive_data(golden):
    kwargs = {}
    if exclude_video := golden.get("exclude_video"):
        kwargs["exclude_video"] = exclude_video
    assert [
        [x[0], list(sorted(x[1]))]
        for x in sorted(x for x in hs.get_4chan_archive_data(golden["input"], **kwargs))
    ] == golden.out["output"]

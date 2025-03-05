import pytest

from mellow_db.utils import count_different, is_list_compatible


@pytest.mark.parametrize(
    "input, is_compatible",
    [
        ([1, 2, 3], True),
        ("string", False),
        ((1, 2, 3), True),
        ({"key", "value"}, True),
        (b"bytes", False),
        ({1: "abc", 2: "def", 3: "tt"}, False)
    ]
)
def test_is_list_compatible(input, is_compatible):
    assert is_list_compatible(input) is is_compatible


@pytest.mark.parametrize(
    "true_scores, predicted_scores, tolerance, expected",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 1e-6, 0),
        ([0.1, 0.2, 0.3], [0.1, 0.200001, 0.3], 1e-6, 1),
        ([0.1, 0.2, 0.3], [0.11, 0.21, 0.31], 0.001, 3),
        ([0.1, 0.2, 0.3], [0.1, 0.1999999, 0.4], 1e-6, 1),
        ([], [], 1e-6, 0),
    ],
)
def test_count_different(true_scores, predicted_scores, tolerance, expected):
    assert count_different(true_scores, predicted_scores, tolerance) == expected

import pytest

import mellow_db.protocols.service_pb2 as pb2
from mellow_db.converter import _get_field_name, from_flex_item, to_flex_item


@pytest.mark.parametrize(
    "value, expected_field, expected_value",
    [
        (None, "is_null", None),
        (True, "bool_value", True),
        (False, "bool_value", False),
        (123, "int_value", 123),
        (123.45, "float_value", 123.45),
        ("test", "str_value", "test"),
        ("", "str_value", ""),
        ([1, 2, 3], "int_list_value", [1, 2, 3]),
        ([1.1, 2.2], "float_list_value", [1.1, 2.2]),
        (["a", "b"], "str_list_value", ["a", "b"]),
        # other iterable types converted to list
        ({1, 2, 3}, "int_list_value", [1, 2, 3]),
        ((1, 2, 3), "int_list_value", [1, 2, 3]),
        # none types
        (["a", None], "str_list_value", ["a", None]),
        (["a", None, "b"], "str_list_value", ["a", None, "b"]),
        (("a", "b", None, "c"), "str_list_value", ["a", "b", None, "c"]),
        ([None], "str_list_value", [None]),
        ([None, None], "str_list_value", [None, None]),
    ]
)
def test_get_field_name(value, expected_field, expected_value):
    assert _get_field_name(value) == (expected_field, expected_value)


@pytest.mark.parametrize(
    "value",
    [
        ([1, "a"],),
        ({1, 2},),
        ([1, None, 2]),
        ([1.2, 2.5, None]),
        ({1: "abc", 2: "def", 3: "tt"}),
    ]
)
def test_get_field_name_invalid(value):
    with pytest.raises(ValueError):
        _get_field_name(value)


@pytest.mark.parametrize(
    "value, expected_flex_item",
    [
        (None, pb2.FlexItem(is_null=True)),
        (True, pb2.FlexItem(bool_value=True)),
        (123, pb2.FlexItem(int_value=123)),
        (123.45, pb2.FlexItem(float_value=123.45)),
        ("test", pb2.FlexItem(str_value=pb2.NullableString(str_value="test"))),
        ("", pb2.FlexItem(str_value=pb2.NullableString(str_value=""))),
        ([1, 2, 3], pb2.FlexItem(int_list_value=pb2.IntListValue(values=[1, 2, 3]))),
        ([1.1, 2.2], pb2.FlexItem(float_list_value=pb2.FloatListValue(values=[1.1, 2.2]))),
        (["a", "b"], pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value="a"),
                    pb2.NullableString(str_value="b")]))),
        (["a", ""], pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value="a"),
                    pb2.NullableString(str_value="")]))),
        (["a", "b", None], pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value="a"),
                    pb2.NullableString(str_value="b"),
                    pb2.NullableString(is_null=True)]))),
        # other iterable types converted to list
        ({1, 2, 3}, pb2.FlexItem(int_list_value=pb2.IntListValue(values=[1, 2, 3]))),
        ((1, 2, 3), pb2.FlexItem(int_list_value=pb2.IntListValue(values=[1, 2, 3]))),
    ]
)
def test_to_flex_item(value, expected_flex_item):
    assert to_flex_item(value) == expected_flex_item


@pytest.mark.parametrize(
    "flex_item, expected_value",
    [
        (pb2.FlexItem(is_null=True), None),
        (pb2.FlexItem(bool_value=True), True),
        (pb2.FlexItem(int_value=123), 123),
        (pb2.FlexItem(float_value=123.45), 123.45),
        (pb2.FlexItem(str_value=pb2.NullableString(str_value="test")), "test"),
        (pb2.FlexItem(str_value=pb2.NullableString(str_value="")), ""),
        (pb2.FlexItem(int_list_value=pb2.IntListValue(values=[1, 2, 3])), [1, 2, 3]),
        (pb2.FlexItem(float_list_value=pb2.FloatListValue(values=[1.1, 2.2])), [1.1, 2.2]),
        (pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value="a"),
                    pb2.NullableString(str_value="b")])),
            ["a", "b"]),
        (pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value=""),
                    pb2.NullableString(is_null=True),
                    pb2.NullableString(str_value="b")])),
            ["", None, "b"]),
        (pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value=""),
                    pb2.NullableString(str_value="a")])),
            ["", "a"]),
        (pb2.FlexItem(str_list_value=pb2.StrListValue(
            values=[pb2.NullableString(str_value="a"),
                    pb2.NullableString(is_null=True),
                    pb2.NullableString(str_value="b")])),
            ["a", None, "b"]),
    ]
)
def test_from_flex_item(flex_item, expected_value):
    assert from_flex_item(flex_item) == expected_value

import mellow_db.protocols.service_pb2 as pb2
from mellow_db.utils import is_list_compatible

null_field = "is_null"
nullable_str_field = "str_value"
nullable_str_list_field = "str_list_value"
list_fields = {nullable_str_list_field, "int_list_value", "float_list_value"}
fields = {null_field, "bool_value", nullable_str_field, "int_value", "float_value"} | list_fields


def to_flex_item(value):
    """
    Convert a Python value to a pb2.FlexItem object.

    Args:
        value (Any): The value to convert. Can be None, a primitive type, or a list of compatible types.

    Returns:
        pb2.FlexItem: A protobuf FlexItem object representing the input value.

    Raises:
        ValueError: If the value type is not supported.
    """
    field_name, field_value = _get_field_name(value)
    if field_name == null_field:
        return pb2.FlexItem(**{field_name: True})
    elif field_name == nullable_str_field:
        return pb2.FlexItem(
            **{field_name: pb2.NullableString(
                is_null=True if field_value is None else False,
                str_value=field_value
            )}
        )
    # if type is list, make the list_class initialization
    elif field_name in list_fields:
        list_class = getattr(pb2, field_name.title().replace("_", ""))
        if field_name == nullable_str_list_field:
            return pb2.FlexItem(
                **{field_name: list_class(
                    values=[
                        pb2.NullableString(
                            is_null=True if val is None else False,
                            str_value=val
                        )
                        for val in field_value
                    ]
                )}
            )
        else:
            return pb2.FlexItem(**{field_name: list_class(values=field_value)})
    else:
        return pb2.FlexItem(**{field_name: field_value})


def from_flex_item(flex_item):
    """
    Convert a pb2.FlexItem object back to its original Python value.

    Args:
        flex_item (pb2.FlexItem): The protobuf FlexItem object to convert.

    Returns:
        Any: The corresponding Python value.
    """
    value_field = flex_item.WhichOneof("value")
    if value_field == null_field:
        return None
    value = getattr(flex_item, value_field)
    if value_field == nullable_str_field:
        return None if value.is_null else value.str_value
    if value_field == nullable_str_list_field:
        return [
            None if val.is_null else val.str_value
            for val in value.values
        ]
    elif value_field in list_fields:
        return list(value.values)
    else:
        return value


def _get_field_name(value):
    """
    Determine the appropriate field name for a given value.

    Args:
        value (Any): The value to analyze.

    Returns:
        tuple: (field_name, processed_value) where field_name is the corresponding protobuf field name,
               and processed_value is the value converted to the appropriate type if needed.

    Raises:
        ValueError: If the value contains mixed types or is unsupported.
    """
    if value is None:
        field_name = null_field
    elif is_list_compatible(value):
        element_types = set(map(type, value))
        # if empty list, or a list of strings, or list of none values or a mixture list of none values and strings
        if not element_types.difference({type(None), type("")}):
            field_name = "str_list_value"
        elif len(element_types) == 1:
            field_name = f"{element_types.pop().__name__}_list_value"
        else:
            raise ValueError(f"List contains mixed types: {element_types}")
        value = list(value)  # convert other iterable types to list
    else:
        field_name = f"{type(value).__name__}_value"
    if field_name not in fields:
        raise ValueError(f"Unsupported type for field '{field_name}' (type='{type(value)}')")
    return field_name, value

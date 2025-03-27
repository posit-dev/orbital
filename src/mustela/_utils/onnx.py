import onnx
import onnx.helper

ListVariableTypes = list[int] | list[float] | list[str]
VariableTypes = float | int | str | ListVariableTypes


def get_initializer_data(var: onnx.TensorProto) -> VariableTypes:
    """Given a constant initializer, return its value"""
    attr_name = onnx.helper.tensor_dtype_to_field(var.data_type)
    values = list(getattr(var, attr_name))
    dimensions = getattr(var, "dims", None)

    if not dimensions and len(values) == 1:
        # If there are no dimensions, it's a scalar
        # and we should return the single value
        return values[0]
    return values


def get_attr_value(attr: onnx.AttributeProto) -> VariableTypes:
    """Given an attribute, return its value"""
    # TODO: Check if it can be replaced with onnx.helper.get_attribute_value
    if attr.type == attr.INTS:
        return list(attr.ints)
    elif attr.type == attr.FLOATS:
        return list(attr.floats)
    elif attr.type == attr.STRINGS:
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]
    elif attr.type == attr.INT:
        return attr.i
    elif attr.type == attr.FLOAT:
        return attr.f
    elif attr.type == attr.STRING:
        return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
    else:
        raise ValueError(f"Unsupported attribute type: {attr.type}")

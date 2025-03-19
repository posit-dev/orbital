import onnx
import onnx.helper
import onnx.numpy_helper


def get_initializer_data(var):
    if var is None:
        raise ValueError("Expected a variable, got None")
    
    attr_name = onnx.helper.tensor_dtype_to_field(var.data_type)
    values = list(getattr(var, attr_name))
    dimensions = getattr(var, "dims", None)

    if not dimensions and len(values) == 1:
        # If there are no dimensions, it's a scalar
        # and we should return the single value
        return values[0]
    return values


def get_attr_value(attr):
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
    elif attr.type == attr.TENSOR:
        return onnx.numpy_helper.to_array(attr.t)
    elif attr.type == attr.GRAPH:
        return attr.g
    elif attr.type == attr.SPARSE_TENSOR:
        return onnx.numpy_helper.to_array(attr.sparse_tensor)
    else:
        raise ValueError(f"Unsupported attribute type: {attr.type}")
import onnx

TYPE_TO_DATA_FIELD = {
    onnx.TensorProto.FLOAT: "float_data",
    onnx.TensorProto.UINT8: "int32_data",
    onnx.TensorProto.INT8: "int32_data",
    onnx.TensorProto.UINT16: "int32_data",
    onnx.TensorProto.INT16: "int32_data",
    onnx.TensorProto.INT32: "int32_data",
    onnx.TensorProto.INT64: "int64_data",
    onnx.TensorProto.STRING: "string_data",
    onnx.TensorProto.BOOL: "int32_data",
    onnx.TensorProto.FLOAT16: "int32_data",
    onnx.TensorProto.DOUBLE: "double_data",
    onnx.TensorProto.UINT32: "int64_data",
    onnx.TensorProto.UINT64: "int64_data",
}


def get_variable_data(var):
    if var is None:
        raise ValueError("Expected a variable, got None")
    
    attr_name = TYPE_TO_DATA_FIELD[var.data_type]
    values = getattr(var, attr_name)
    dimensions = getattr(var, "dims", None)

    if not dimensions and len(values) == 1:
        return values[0]
    return values


def get_attr_value(attr):
    # TODO: Check if it can be replaced with onnx.numpy_helper.get_attribute_value
    #       for some reason it doesn't behave as expected, so we wrote our own function.
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
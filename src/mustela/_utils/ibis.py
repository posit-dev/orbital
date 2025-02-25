def empty_memtable(onnx_model):
    """
    Create an empty ibis.memtable that has the same
    columns as the ONNX model input.
    """
    schema = ibis.schema(
        {
            inp.name: ONNX_TYPES_TO_IBIS[inp.type.tensor_type.elem_type]
            for inp in onnx_model.graph.input
        }
    )
    return ibis.memtable(
        {inp.name: [] for inp in onnx_model.graph.input}, schema=schema
    )



def debug_expr(expr, indent=0):
    """
    Traverses an Ibis expression in pre-order, printing debug information
    for each node before processing its children.
    """
    prefix = " " * indent

    if isinstance(expr, dict):
        print(f"{prefix}Dictionary with keys: {list(expr.keys())}")
        for key, value in expr.items():
            print(f"{prefix} Key: {key}")
            debug_expr(value, indent=indent + 2)
        return

    if isinstance(expr, Field):
        print(f"{prefix}Field: {expr.name}")
        return

    if hasattr(expr, "to_expr"):
        expr = expr.to_expr()

    if hasattr(expr, "op") and callable(expr.op):
        try:
            op = expr.op()
            if isinstance(op, Literal):
                print(f"{prefix}Literal: {op.value}")
                return
            else:
                print(f"{prefix}{type(op).__name__}: {op}")
        except Exception as e:
            print(f"{prefix}Error retrieving op() from {expr}: {e}")
            return
    else:
        # Expression is already an operation
        op = expr

    if isinstance(op, (float, int, bool, str)):
        print(f"{prefix}PyLiteral: {op}")
        return
    elif isinstance(op, tuple):
        for item in op:
            debug_expr(item, indent=indent + 2)
        return

    if isinstance(op, PhysicalTable):
        print(f"{prefix}Table: {op.name}")
        return

    try:
        for arg in op.args:
            try:
                debug_expr(arg, indent=indent + 2)
            except Exception:
                print(f"{prefix}Error printing {arg}")
                import traceback

                traceback.print_exc()
    except AttributeError as e:
        print(
            f"{prefix}Error traversing args of {type(op).__name__}: {[type(a) for a in op.args]}"
        )


def display_pipeline(pipeline):
    from onnxscript import ir

    onnx_model = pipeline._model
    script_model = ir.from_proto(onnx_model)
    script_model.graph.display(page=False)


def predict_with_onnxruntime(pipeline):
    import numpy
    import onnxruntime as rt
    import pandas

    onnx_model = pipeline._model
    with open("temp.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    sess = rt.InferenceSession("temp.onnx")
    input_name = sess.get_inputs()[0].name
    
    # TODO: This needs to be generated based on pipeline features.
    pred_onx = sess.run(
        None, {input_name: X["feature1"].to_numpy(dtype=numpy.float32).reshape(-1, 1)}
    )[0]
    return pandas.DataFrame({"feature1": X["feature1"], "prediction": pred_onx.flatten()}).head(5)


def map_sklearn2onnx_names(sklearn_feature_names, onnx_column_names):
    categories_map = {}
    for colname in sklearn_feature_names:
        varname, cat = colname.split("_")
        categories_map.setdefault(cat, []).append(varname)

    variables = []
    seen_cats = []
    name_maps = {}
    for colname in onnx_column_names:
        varname, cat = colname.split(".", 1)
        if '.' in cat:
            varname, cat = cat.split(".", 1)

        if variables and varname != variables[-1][0]:
            oldcandidate = variables[-1][1]
            # print("Removing", oldcandidate, "from", seen_cats)
            for seen_cat in seen_cats:
                categories_map[seen_cat] = [c for c in categories_map[seen_cat] if c != oldcandidate]
            seen_cats = []

        seen_cats.append(cat)
        candidates = categories_map[cat]
        best_candidate = candidates[0]
        if len(candidates) > 1:
            oldvarname, mappedname = variables[-1]
            # print("Conflict", colname, varname, oldvarname, cat, candidates)
            if oldvarname == varname:
                # print("\tAssuming", variables[-1], "for", oldvarname, varname, "conflict")
                best_candidate = mappedname
            else:
                # Don't know, assume the columns are in the same order.
                # print("\tBest Guess", candidates[0], "for", oldvarname, varname, "conflict")
                best_candidate = candidates[0]
        
        name_maps[colname] = best_candidate + "_" + cat
        # print("Mapped", colname, "->", name_maps[colname])
        variables.append((varname, best_candidate))

    return name_maps
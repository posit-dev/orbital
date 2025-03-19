import ibis

from ...translator import Translator
from .tree import build_tree, mode_to_condition


class TreeEnsembleRegressorTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html
    # This is deprecated in ONNX but it's what skl2onnx uses.

    def process(self):
        input_exr = self._variables.consume(self.inputs[0])
        prediction_expr = self.build_regressor(input_exr)
        self.set_output(prediction_expr)

    def build_regressor(self, input_expr):
        optimizer = self._optimizer
        ensemble_trees = build_tree(self)

        if isinstance(input_expr, dict):
            ordered_features = list(input_expr.values())
        else:
            ordered_features = [input_expr]
        ordered_features = [
            feature.name(self.variable_unique_short_alias("tclass"))
            for feature in ordered_features
        ]
        ordered_features = self.preserve(*ordered_features)

        def build_tree_value(node):
            # Leaf node, should return the prediction weight
            if node["mode"] == "LEAF":
                return ibis.literal(node["weight"])

            # BRANCH node, should return a CASE statement
            feature_expr = ordered_features[node["feature_id"]]
            condition = mode_to_condition(node, feature_expr)

            if node["missing_tracks_true"]:
                raise NotImplementedError("Missing value tracks true not supported")

            true_val = build_tree_value(node["true"])
            false_val = build_tree_value(node["false"])
            case_expr = optimizer.fold_case(
                ibis.case().when(condition, true_val).else_(false_val).end()
            )
            return case_expr

        # Build results from each tree and sum them
        tree_values = []
        for tree in ensemble_trees.values():
            tree_values.append(build_tree_value(tree))
        total_value = ibis.literal(0.0)
        for val in tree_values:
            total_value = optimizer.fold_operation(total_value + val)

        # According to ONNX doc: can be left unassigned (assumed 0)
        base_values = self._attributes.get("base_values", [0.0])
        if len(base_values) != 1:
            raise NotImplementedError("Base values with length != 1 not supported")
        total_value = optimizer.fold_operation(total_value + ibis.literal(base_values[0]))

        return total_value
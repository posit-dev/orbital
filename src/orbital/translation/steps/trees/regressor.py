"""Implement regression based on trees"""

import typing

import ibis

from ...translator import Translator
from ...variables import VariablesGroup
from .tree import BranchConditionCreator, build_tree


class TreeEnsembleRegressorTranslator(Translator):
    """Processes a TreeEnsembleClassifier node and updates the variables with the output expression.

    This node is foundational for most tree based models:
    - Gradient Boosted Trees
    - Decision Trees

    The parsing of the tree is done by the [orbital.translation.steps.trees.regressor.build_tree][] function,
    which results in a dictionary of trees.

    The class parses the trees to generate a set of `CASE WHEN THEN ELSE`
    expressions that are used to compute the prediction for each tree.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html
        # This is deprecated in ONNX but it's what skl2onnx uses.
        input_exr = self._variables.consume(self.inputs[0])
        if not isinstance(input_exr, (ibis.Expr, VariablesGroup)):
            raise ValueError(
                "TreeEnsembleRegressor: The first operand must be a column or a column group."
            )

        prediction_expr = self.build_regressor(input_exr)
        self.set_output(prediction_expr)

    def build_regressor(
        self, input_expr: typing.Union[VariablesGroup, ibis.Expr]
    ) -> ibis.Expr:
        """Build the regression expression"""
        optimizer = self._optimizer
        ensemble_trees = build_tree(self)

        condition_creator = BranchConditionCreator(self, input_expr)

        def build_tree_value(node: dict) -> ibis.Expr:
            # Leaf node, should return the prediction weight
            if node["mode"] == "LEAF":
                return ibis.literal(node["weight"])

            # BRANCH node, should return a CASE statement
            condition = condition_creator.create_condition(node)

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
        total_value: ibis.NumericValue = ibis.literal(0.0)
        for val in tree_values:
            total_value = optimizer.fold_operation(total_value + val)

        # According to ONNX doc: can be left unassigned (assumed 0)
        base_values = typing.cast(
            list[float], self._attributes.get("base_values", [0.0])
        )
        if len(base_values) != 1:
            raise NotImplementedError("Base values with length != 1 not supported")
        total_value = optimizer.fold_operation(
            total_value + ibis.literal(base_values[0])
        )

        return total_value

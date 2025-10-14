"""Prase tree definitions and return a graph of nodes."""

import typing

import ibis

from ...translator import Translator
from ...variables import VariablesGroup


class BranchConditionCreator:
    """Factory for branch conditions that reuses preserved feature columns."""

    def __init__(
        self,
        translator: Translator,
        input_expr: typing.Union[ibis.Expr, VariablesGroup],
    ) -> None:
        """Set up reusable branch conditions for tree traversal.

        :param translator: The [orbital.translation.translator.Translator][] instance
            that preserves intermediate expressions and holds ONNX node metadata.
        :param input_expr: The expression or
            [orbital.translation.variables.VariablesGroup][] providing the features
            consumed by the tree.
        """
        self._translator = translator
        self._ordered_features = self._prepare_features(input_expr)
        self._conditions = self._prepare_conditions()

    def _prepare_features(
        self, input_expr: typing.Union[ibis.Expr, VariablesGroup]
    ) -> list[ibis.Expr]:
        if isinstance(input_expr, VariablesGroup):
            features: list[ibis.Expr] = input_expr.values_value()
        else:
            features = typing.cast(list[ibis.Expr], [input_expr])

        renamed_features = [
            feature.name(self._translator.variable_unique_short_alias("cnf"))
            for feature in features
        ]
        # `node["feature_id"]` is a 0-based index into this preserved list, so we
        # return it directly instead of building a mapping keyed by feature id.
        return self._translator.preserve(*renamed_features)

    def _condition_expression(
        self,
        feature_expr: ibis.Expr,
        threshold_expr: ibis.Expr,
        mode: str,
    ) -> ibis.Expr:
        if mode == "BRANCH_LEQ":
            return feature_expr <= threshold_expr
        if mode == "BRANCH_LT":
            return feature_expr < threshold_expr
        if mode == "BRANCH_GTE":
            return feature_expr >= threshold_expr
        if mode == "BRANCH_GT":
            return feature_expr > threshold_expr
        if mode == "BRANCH_EQ":
            return feature_expr == threshold_expr
        if mode == "BRANCH_NEQ":
            return feature_expr != threshold_expr

        raise NotImplementedError(f"Unsupported node mode: {mode}")

    def _prepare_conditions(self) -> dict[tuple[int, str, float], ibis.Expr]:
        node_modes = typing.cast(list[str], self._translator._attributes["nodes_modes"])
        node_featureids = typing.cast(
            list[int], self._translator._attributes["nodes_featureids"]
        )
        node_thresholds = typing.cast(
            list[float], self._translator._attributes["nodes_values"]
        )

        keys_to_preserve: dict[tuple[int, str, float], ibis.Expr] = {}

        for feature_id, mode, threshold in zip(
            node_featureids, node_modes, node_thresholds
        ):
            if mode == "LEAF":
                continue
            key = (feature_id, mode, threshold)
            if key in keys_to_preserve:
                continue

            feature_expr = self._ordered_features[feature_id]
            threshold_expr = ibis.literal(threshold)
            raw_condition = self._condition_expression(
                feature_expr, threshold_expr, mode
            )
            named_expression = raw_condition.name(
                self._translator.variable_unique_short_alias("cnc")
            )
            keys_to_preserve[key] = named_expression

        if not keys_to_preserve:
            return {}

        preserved_exprs = self._translator.preserve(*keys_to_preserve.values())
        return dict(zip(keys_to_preserve.keys(), preserved_exprs))

    def create_condition(self, node: dict) -> ibis.Expr:
        """Return a preserved comparison expression for the given branch node."""
        key = (node["feature_id"], node["mode"], node["treshold"])
        return self._conditions[key]


def build_tree(translator: Translator) -> dict[int, dict[int, dict]]:
    """Build a tree based on nested dictionaries of nodes."""
    nodes_treeids = typing.cast(list[int], translator._attributes["nodes_treeids"])
    nodes_nodeids = typing.cast(list[int], translator._attributes["nodes_nodeids"])
    nodes_modes = typing.cast(list[str], translator._attributes["nodes_modes"])
    nodes_truenodeids = typing.cast(
        list[int], translator._attributes["nodes_truenodeids"]
    )
    nodes_falsenodeids = typing.cast(
        list[int], translator._attributes["nodes_falsenodeids"]
    )
    nodes_thresholds = typing.cast(list[float], translator._attributes["nodes_values"])
    nodes_featureids = typing.cast(
        list[int], translator._attributes["nodes_featureids"]
    )
    nodes_missing_value_tracks_true = typing.cast(
        list[int], translator._attributes["nodes_missing_value_tracks_true"]
    )
    node = translator._node

    # Assert a few things to ensure we don't ed up genearting a tree with wrong data
    # All entries related to branches should match in length
    assert (
        len(nodes_treeids)
        == len(nodes_nodeids)
        == len(nodes_modes)
        == len(nodes_truenodeids)
        == len(nodes_falsenodeids)
        == len(nodes_thresholds)
        == len(nodes_featureids)
    )

    # Weight could be a float or a dictionary of class labels weights
    weights: dict = {}
    if node.op_type == "TreeEnsembleClassifier":
        weights = typing.cast(
            dict[tuple[int, int], dict[typing.Union[str, int], float]], weights
        )
        # Weights for classifier, in this case the weights are per-class
        class_nodeids = typing.cast(list[int], translator._attributes["class_nodeids"])
        class_treeids = typing.cast(list[int], translator._attributes["class_treeids"])
        class_weights = typing.cast(
            list[float], translator._attributes["class_weights"]
        )
        weights_classid = typing.cast(list[int], translator._attributes["class_ids"])
        assert (
            len(class_treeids)
            == len(class_nodeids)
            == len(class_weights)
            == len(weights_classid)
        )
        classlabels = typing.cast(
            typing.Optional[list[typing.Union[str, int]]],
            translator._attributes.get("classlabels_strings")
            or translator._attributes.get("classlabels_int64s"),
        )
        if not classlabels:
            raise ValueError("Missing class labels when building tree")

        for tree_id, node_id, weight, weight_classid in zip(
            class_treeids, class_nodeids, class_weights, weights_classid
        ):
            node_weights = typing.cast(
                dict[typing.Union[str, int], float],
                weights.setdefault((tree_id, node_id), {}),
            )
            node_weights[classlabels[weight_classid]] = weight

    elif node.op_type == "TreeEnsembleRegressor":
        # Weights for the regressor, in this case leaf nodes have only 1 weight
        weights = typing.cast(dict[tuple[int, int], float], weights)
        target_weights = typing.cast(
            list[float], translator._attributes["target_weights"]
        )
        target_nodeids = typing.cast(
            list[int], translator._attributes["target_nodeids"]
        )
        target_treeids = typing.cast(
            list[int], translator._attributes["target_treeids"]
        )
        assert len(target_treeids) == len(target_nodeids) == len(target_weights)
        for tree_id, node_id, weight in zip(
            target_treeids, target_nodeids, target_weights
        ):
            weights[(tree_id, node_id)] = weight
    else:
        raise NotImplementedError(f"Unsupported tree node type: {node.op_type}")

    # Create all nodes for the trees
    trees: dict[int, dict[int, dict]] = {}
    for tree_id, node_id, mode, true_id, false_id, threshold, feature_id in zip(
        nodes_treeids,
        nodes_nodeids,
        nodes_modes,
        nodes_truenodeids,
        nodes_falsenodeids,
        nodes_thresholds,
        nodes_featureids,
    ):
        if tree_id not in trees:
            trees[tree_id] = {}

        node_dict = {
            "id": (tree_id, node_id),
            "mode": mode,
            "feature_id": feature_id,
            "missing_tracks_true": bool(
                nodes_missing_value_tracks_true[node_id]
                if nodes_missing_value_tracks_true
                else 0
            ),
        }
        if mode == "LEAF":
            node_dict["weight"] = weights[(tree_id, node_id)]
        else:
            node_dict["treshold"] = threshold

        trees[tree_id][node_id] = node_dict

    # Link nodes creating a tree structure
    for tree_id, node_id, true_id, false_id in zip(
        nodes_treeids,
        nodes_nodeids,
        nodes_truenodeids,
        nodes_falsenodeids,
    ):
        if node_id in trees[tree_id]:
            node_dict = trees[tree_id][node_id]
            if node_dict["mode"] == "LEAF":
                # Leaf nodes have no true or false branches
                # In the end they are leaves so they don't have branches
                continue
            if true_id in trees[tree_id]:
                node_dict["true"] = trees[tree_id][true_id]
            if false_id in trees[tree_id]:
                node_dict["false"] = trees[tree_id][false_id]

    return {tree_id: trees[tree_id][0] for tree_id in trees}

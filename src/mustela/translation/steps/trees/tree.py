"""Prase tree definitions and return a graph of nodes."""

import typing

import ibis

from ...translator import Translator


def build_tree(translator: Translator) -> dict[int, dict[int, dict]]:
    """Build a tree based on nested dictionaries of nodes.

    The tree is built based on the node and attributes of the translator.
    """
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
        weights = typing.cast(dict[tuple[int, int], dict[str | int, float]], weights)
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
            None | list[str | int],
            translator._attributes.get("classlabels_strings")
            or translator._attributes.get("classlabels_int64s"),
        )
        if not classlabels:
            raise ValueError("Missing class labels when building tree")

        is_binary = len(classlabels) == 2 and len(set(weights_classid)) == 1
        for tree_id, node_id, weight, weight_classid in zip(
            class_treeids, class_nodeids, class_weights, weights_classid
        ):
            node_weights = typing.cast(
                dict[str | int, float], weights.setdefault((tree_id, node_id), {})
            )
            node_weights[classlabels[weight_classid]] = weight

        if is_binary:
            # ONNX treats binary classification as a special case:
            # https://github.com/microsoft/onnxruntime/blob/5982430af66f52a288cb8b2181e0b5b2e09118c8/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h#L854C1-L871C4
            # https://github.com/microsoft/onnxruntime/blob/5982430af66f52a288cb8b2181e0b5b2e09118c8/onnxruntime/core/providers/cpu/ml/tree_ensemble_aggregator.h#L469-L494
            # In this case there is only one weight and it's the probability of the positive class.
            for node_weights in weights.values():
                assert len(node_weights) == 1, (
                    f"Binary classification expected to have only one class, got: {node_weights}"
                )
                score = list(node_weights.values())[0]
                if score > 0.5:
                    node_weights[classlabels[1]] = 1.0
                    node_weights[classlabels[0]] = 0.0
                else:
                    node_weights[classlabels[1]] = 0.0
                    node_weights[classlabels[0]] = 1.0

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


def mode_to_condition(node: dict, feature_expr: ibis.Expr) -> ibis.Expr:
    """Build a comparison expression for a branch node.

    The comparison is based on the mode of the node and the threshold
    for that noode. The feature will be compared to the threshold
    using the operator defined by the mode.
    """
    threshold = node["treshold"]
    if node["mode"] == "BRANCH_LEQ":
        condition = feature_expr <= threshold
    elif node["mode"] == "BRANCH_LT":
        condition = feature_expr < threshold
    elif node["mode"] == "BRANCH_GTE":
        condition = feature_expr >= threshold
    elif node["mode"] == "BRANCH_GT":
        condition = feature_expr > threshold
    elif node["mode"] == "BRANCH_EQ":
        condition = feature_expr == threshold
    elif node["mode"] == "BRANCH_NEQ":
        condition = feature_expr != threshold
    else:
        raise NotImplementedError(f"Unsupported node mode: {node['mode']}")
    return condition

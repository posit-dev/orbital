"""Implement classification based on trees"""

import typing

import ibis

from ...translator import Translator
from ...variables import ValueVariablesGroup, VariablesGroup
from ..linearclass import LinearClassifierTranslator
from ..softmax import SoftmaxTranslator
from .tree import build_tree, mode_to_condition


class TreeEnsembleClassifierTranslator(Translator):
    """Processes a TreeEnsembleClassifier node and updates the variables with the output expression.

    This node is foundational for most tree based models:
    - Random Forest
    - Gradient Boosted Trees
    - Decision Trees

    The parsing of the tree is done by the :func:`build_tree` function,
    which results in a dictionary of trees.

    The class parses the trees to generate a set of `CASE WHEN THEN ELSE`
    expressions that are used to compute the votes for each class.

    The class also computes the probability of each class by dividing
    the votes by the sum of all votes.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html
        # This is deprecated in ONNX but it's what skl2onnx uses.

        input_exr = self._variables.consume(self.inputs[0])
        if not isinstance(input_exr, (ibis.Expr, VariablesGroup)):
            raise ValueError(
                "TreeEnsembleClassifier: The first operand must be a column or a column group."
            )

        label_expr, prob_colgroup = self.build_classifier(input_exr)
        post_transform = typing.cast(
            str, self._attributes.get("post_transform", "NONE")
        )

        if post_transform != "NONE":
            if post_transform == "SOFTMAX":
                prob_colgroup = SoftmaxTranslator.compute_softmax(self, prob_colgroup)
            elif post_transform == "LOGISTIC":
                prob_colgroup = ValueVariablesGroup(
                    {
                        lbl: LinearClassifierTranslator._apply_post_transform(
                            prob_col, post_transform
                        )
                        for lbl, prob_col in prob_colgroup.items()
                    }
                )
            else:
                raise NotImplementedError(
                    f"Post transform {post_transform} not implemented."
                )

        self._variables[self.outputs[0]] = label_expr
        self._variables[self.outputs[1]] = prob_colgroup

    def build_classifier(
        self, input_expr: ibis.Expr | VariablesGroup
    ) -> tuple[ibis.Expr, VariablesGroup]:
        """Build the classification expression and the probabilities expressions

        Return the classification expression as the first argument and a group of
        variables (one for each category) for the probability expressions.
        """
        optimizer = self._optimizer
        ensemble_trees = build_tree(self)

        classlabels = self._attributes.get(
            "classlabels_strings"
        ) or self._attributes.get("classlabels_int64s")
        if classlabels is None:
            raise ValueError("Unable to detect classlabels for classification")
        output_classlabels = classlabels = typing.cast(
            list[str] | list[int], classlabels
        )

        # ONNX treats binary classification as a special case:
        # https://github.com/microsoft/onnxruntime/blob/5982430af66f52a288cb8b2181e0b5b2e09118c8/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h#L854C1-L871C4
        # https://github.com/microsoft/onnxruntime/blob/5982430af66f52a288cb8b2181e0b5b2e09118c8/onnxruntime/core/providers/cpu/ml/tree_ensemble_aggregator.h#L469-L494
        # In this case there is only one weight and it's the probability of the positive class.
        # So we need to check if we are in a binary classification case.
        weights_classid = typing.cast(list[int], self._attributes["class_ids"])
        is_binary = len(classlabels) == 2 and len(set(weights_classid)) == 1
        if is_binary:
            # In this case there is only one label, the first one
            # which actually acts as the score of the prediction.
            # When > 0.5 then class 1, when < 0.5 then class 0
            classlabels = typing.cast(list[str] | list[int], [classlabels[0]])

        if isinstance(input_expr, VariablesGroup):
            ordered_features = input_expr.values_value()
        else:
            ordered_features = typing.cast(list[ibis.Value], [input_expr])
        ordered_features = [
            feature.name(self.variable_unique_short_alias("tcl"))
            for feature in ordered_features
        ]
        ordered_features = self.preserve(*ordered_features)

        def build_tree_case(node: dict) -> dict[str | int, ibis.Expr]:
            # Leaf node, return the votes
            if node["mode"] == "LEAF":
                # We can assume missing class = weight 0
                # The optimizer will remove this if both true and false have 0.
                return {
                    clslabel: ibis.literal(node["weight"].get(clslabel, 0.0))
                    for clslabel in classlabels
                }

            # Branch node, build a CASE statement
            feature_expr = ordered_features[node["feature_id"]]
            condition = mode_to_condition(node, feature_expr)

            true_votes = build_tree_case(node["true"])
            false_votes = build_tree_case(node["false"])

            votes = {}
            for clslabel in classlabels:
                t_val = true_votes[clslabel]
                f_val = false_votes[clslabel]
                votes[clslabel] = optimizer.fold_case(
                    ibis.case().when(condition, t_val).else_(f_val).end()
                )
            return votes

        # Genera voti per ogni albero
        tree_votes = []
        for tree in ensemble_trees.values():
            tree_votes.append(build_tree_case(tree))

        # Aggregate votes from all trees.
        total_votes = {}
        for clslabel in classlabels:
            total_votes[clslabel] = ibis.literal(0.0)
            for votes in tree_votes:
                total_votes[clslabel] = optimizer.fold_operation(
                    total_votes[clslabel] + votes.get(clslabel, ibis.literal(0.0))
                )

        # Compute prediction of class itself.
        if is_binary:
            total_score = total_votes[classlabels[0]]
            label_expr = optimizer.fold_case(
                ibis.case()
                .when(total_score > 0.5, output_classlabels[1])
                .else_(output_classlabels[0])
                .end()
            )
            # The order matters, for ONNX the VariableGroup is a list of subvariables
            # the names are not important.
            prob_dict = ValueVariablesGroup(
                {
                    str(output_classlabels[0]): 1.0 - total_score,
                    str(output_classlabels[1]): total_score,
                }
            )
        else:
            candidate_cls = classlabels[0]
            candidate_vote = total_votes[candidate_cls]
            for clslabel in classlabels[1:]:
                candidate_cls = optimizer.fold_case(
                    ibis.case()
                    .when(total_votes[clslabel] > candidate_vote, clslabel)
                    .else_(candidate_cls)
                    .end()
                )
                candidate_vote = optimizer.fold_case(
                    ibis.case()
                    .when(total_votes[clslabel] > candidate_vote, total_votes[clslabel])
                    .else_(candidate_vote)
                    .end()
                )

            label_expr = ibis.case()
            for clslabel in classlabels:
                label_expr = label_expr.when(candidate_cls == clslabel, clslabel)
            label_expr = label_expr.else_(ibis.null()).end()
            label_expr = optimizer.fold_case(label_expr)

            post_transform = typing.cast(
                str, self._attributes.get("post_transform", "NONE")
            )
            if post_transform == "SOFTMAX":
                # Use softmax as an hint that we are doing a gradient boosted tree,
                # thus the probability is the same as the score and should not be normalized
                prob_dict = ValueVariablesGroup(
                    {str(clslabel): total_votes[clslabel] for clslabel in classlabels}
                )
            else:
                # Compute probability to return it too.
                sum_votes = sum(total_votes[clslabel] for clslabel in classlabels)
                prob_dict = ValueVariablesGroup(
                    {
                        str(clslabel): total_votes[clslabel] / sum_votes
                        for clslabel in classlabels
                    }
                )

        return label_expr, prob_dict

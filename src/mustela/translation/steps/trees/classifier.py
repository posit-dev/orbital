"""Implement classification based on trees"""

import typing

import ibis

from ...translator import Translator
from ...variables import VariablesGroup
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
                prob_colgroup = SoftmaxTranslator.compute_softmax(prob_colgroup)
            elif post_transform == "LOGISTIC":
                prob_colgroup = VariablesGroup(
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
        classlabels = typing.cast(list[str] | list[int], classlabels)

        if isinstance(input_expr, VariablesGroup):
            ordered_features = input_expr.values_value()
        else:
            ordered_features = typing.cast(list[ibis.Value], [input_expr])
        ordered_features = [
            feature.name(self.variable_unique_short_alias("tclass"))
            for feature in ordered_features
        ]
        ordered_features = self.preserve(*ordered_features)

        def build_tree_case(node: dict) -> dict[str | int, ibis.Expr]:
            # Leaf node, return the votes
            if node["mode"] == "LEAF":
                votes = {}
                for clslabel in classlabels:
                    # We can assume missing class = weight 0
                    # The optimizer will remove this if both true and false have 0.
                    votes[clslabel] = ibis.literal(node["weight"].get(clslabel, 0))
                return votes

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

        # Compute probability to return it too.
        sum_votes = None
        for clslabel in classlabels:
            if sum_votes is None:
                sum_votes = total_votes[clslabel]
            else:
                sum_votes = optimizer.fold_operation(sum_votes + total_votes[clslabel])

        # FIXME: Probabilities are currently broken for gradient boosted trees.
        prob_dict = VariablesGroup()
        for clslabel in classlabels:
            prob_dict[str(clslabel)] = total_votes[clslabel] / sum_votes

        return label_expr, prob_dict

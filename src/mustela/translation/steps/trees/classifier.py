import ibis

from ...translator import Translator
from .tree import build_tree, mode_to_condition


class TreeEnsembleClassifierTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html
    # This is deprecated in ONNX but it's what skl2onnx uses.

    def process(self):
        input_exr = self._variables.consume(self.inputs[0])
        label_expr, prob_expr = self.build_classifier(input_exr)
        
        self._variables[self.outputs[0]] = label_expr
        self._variables[self.outputs[1]] = prob_expr

    def build_classifier(self, input_expr):
        optimizer = self._optimizer
        ensemble_trees = build_tree(self)
        classlabels = self._attributes.get("classlabels_strings") or self._attributes(
            "classlabels_int64s"
        )

        if isinstance(input_expr, dict):
            ordered_features = list(input_expr.values())
        else:
            ordered_features = [input_expr]
        ordered_features = [
            feature.name(self.variable_unique_short_alias("tclass"))
            for feature in ordered_features
        ]
        self.preserve(*ordered_features)

        def build_tree_case(tree, node):
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

            true_votes = build_tree_case(tree, node["true"])
            false_votes = build_tree_case(tree, node["false"])

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
        for treeid, tree in ensemble_trees.items():
            tree_votes.append(build_tree_case(tree, tree))

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
        label_expr = label_expr.else_("unknown").end()
        label_expr = optimizer.fold_case(label_expr)

        # Compute probability to return it too.
        sum_votes = None
        for clslabel in classlabels:
            if sum_votes is None:
                sum_votes = total_votes[clslabel]
            else:
                sum_votes = optimizer.fold_operation(sum_votes + total_votes[clslabel])

        prob_dict = {}
        for clslabel in classlabels:
            prob_dict[clslabel] = total_votes[clslabel] / sum_votes
        prob_expr = ibis.struct(prob_dict)

        return label_expr, prob_expr

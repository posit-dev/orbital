"""Translation options used to customize the ONNX→SQL conversion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranslationOptions:
    """Configuration knobs that affect translation behaviour."""

    allow_text_tensors: bool = False
    """Allow numeric or boolean tensors that ONNX coerced to text to remain as
    text columns in the generated SQL.

    Defaults to ``False`` so encoded columns stay numeric, which matches the
    behaviour most downstream SQL engines expect.
    """

    separate_trees: bool = False
    """Materialise each tree in an ensemble as its own SQL column before summing.

    When ``True``, tree ensembles (Gradient Boosted Trees, Random Forests) are
    emitted as a subquery that exposes one ``CASE`` expression per tree, with
    the final prediction computed by summing those columns plus the base score.
    When ``False`` (the default), every tree is inlined into a single large
    summed expression.

    Columnar engines such as DuckDB can evaluate independent tree columns in
    parallel and benefit from this layout; row-oriented engines such as SQLite
    or PostgreSQL see no runtime difference. The trade-off is a modest increase
    in generated SQL size (on the order of 7%), which is why the flag defaults
    to ``False``.

    See the R ``orbital`` project's separate trees article at
    <https://orbital.tidymodels.org/articles/separate-trees.html> for a
    complementary discussion.
    """

import onnx

import mustela
from mustela import types
from mustela.ast import ParsedPipeline

BASIC_FEATURES = {
    "sepal_length": types.FloatColumnType(),
    "sepal_width": types.FloatColumnType(),
    "petal_length": types.FloatColumnType(),
    "petal_width": types.FloatColumnType(),
}
BASIC_MODEL = onnx.helper.make_model(
    onnx.parser.parse_graph("""
agraph (double[?,1] sepal_length, double[?,1] sepal_width, double[?,1] petal_length, double[?,1] petal_width) => (double[?,1] variable) 
   <double[4] Su_Subcst =  {5.84333,3.05733,3.758,1.19933}, double[4,1] coef =  {-0.111906,-0.0400795,0.228645,0.609252}, double[1] intercept =  {1}, int64[2] shape_tensor =  {-1,1}>
{
   merged_columns = Concat <axis: int = 1> (sepal_length, sepal_width, petal_length, petal_width)
   variable1 = Sub (merged_columns, Su_Subcst)
   multiplied = MatMul (variable1, coef)
   resh = Add (multiplied, intercept)
   variable = Reshape (resh, shape_tensor)
}
""")
)


class TestSQLExport:
    def test_sql(self):
        parsed_pipeline = ParsedPipeline._from_onnx_model(BASIC_MODEL, BASIC_FEATURES)
        sql = mustela.export_sql("DATA_TABLE", parsed_pipeline, dialect="duckdb")
        assert sql == (
            'SELECT ("t0"."sepal_length" - 5.84333) * -0.111906 + 1.0 + '
            '("t0"."sepal_width" - 3.05733) * -0.0400795 + '
            '("t0"."petal_length" - 3.758) * 0.228645 + '
            '("t0"."petal_width" - 1.19933) * 0.609252 '
            'AS "variable" FROM "DATA_TABLE" AS "t0"'
        )

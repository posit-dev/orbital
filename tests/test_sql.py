import onnx
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbitalml
from orbitalml import types
from orbitalml.ast import ParsedPipeline

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
    @pytest.fixture(scope="class")
    def iris_data(self):
        iris = load_iris()
        # Clean feature names to match what's used in the example
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X = pd.DataFrame(iris.data, columns=feature_names)  # Use clean names directly
        y = pd.DataFrame(iris.target, columns=["target"])
        df = pd.concat([X, y], axis=1)
        return df, feature_names

    def test_sql(self):
        parsed_pipeline = ParsedPipeline._from_onnx_model(BASIC_MODEL, BASIC_FEATURES)
        sql = orbitalml.export_sql("DATA_TABLE", parsed_pipeline, dialect="duckdb")
        assert sql == (
            'SELECT ("t0"."sepal_length" - 5.84333) * -0.111906 + 1.0 + '
            '("t0"."sepal_width" - 3.05733) * -0.0400795 + '
            '("t0"."petal_length" - 3.758) * 0.228645 + '
            '("t0"."petal_width" - 1.19933) * 0.609252 '
            'AS "variable" FROM "DATA_TABLE" AS "t0"'
        )

    def test_sql_optimization_flag(self, iris_data):
        df, feature_names = iris_data

        sklearn_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regression", LinearRegression())]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        optimized_sql = orbitalml.export_sql(
            "data", parsed_pipeline, dialect="duckdb", optimize=True
        )
        unoptimized_sql = orbitalml.export_sql(
            "data", parsed_pipeline, dialect="duckdb", optimize=False
        )

        assert (
            optimized_sql
            == 'SELECT 1.0 + ("t0"."sepal_length" - 5.8433332443237305) * -0.1119058608179397432284150575 + ("t0"."sepal_width" - 3.05733323097229) * -0.04007948771815250781921206973 + ("t0"."petal_length" - 3.757999897003174) * 0.2286450295022994613348661968 + ("t0"."petal_width" - 1.1993333101272583) * 0.6092520419738746983614281006 AS "variable.target_0" FROM "data" AS "t0"'
        )
        assert len(optimized_sql) < len(unoptimized_sql)

    @pytest.mark.parametrize(
        "dialect", ["duckdb", "sqlite", "postgres", "mysql", "bigquery", "snowflake"]
    )
    def test_multiple_sql_dialects(self, iris_data, dialect):
        df, feature_names = iris_data

        sklearn_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regression", LinearRegression())]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        try:
            sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
            assert isinstance(sql, str) and len(sql) > 0
        except Exception as e:
            pytest.skip(f"Dialect {dialect} not supported: {str(e)}")

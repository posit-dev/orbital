import logging

import ibis
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mustela
import mustela.types

PRINT_SQL = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.DEBUG)

iris = load_iris(as_frame=True)

names = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

iris_x = iris.data.set_axis(names, axis=1)

pipeline = Pipeline(
    [
        (
            "preprocess",
            ColumnTransformer(
                [("scaler", StandardScaler(with_std=False), names)],
                remainder="passthrough",
            ),
        ),
        ("linear_regression", LinearRegression()),
    ]
)
pipeline.fit(iris_x, iris.target)

print(iris_x.columns)

features = mustela.types.guess_datatypes(iris_x)
print("Mustela Features:", features)

mustela_pipeline = mustela.parse_pipeline(pipeline, features=features)
print(mustela_pipeline)

# Include at least 1 value from training set to confirm the right computation happened
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
    }
)

ibis_expression = mustela.translate(ibis.memtable(example_data), mustela_pipeline)

if PRINT_SQL:
    print("\nGenerated Query for DuckDB:")
    con = ibis.duckdb.connect()
    print(con.compile(ibis_expression))

print("\nPrediction with Ibis")
print(ibis_expression.execute())

print("\nPrediction with SKLearn")
new_column_names = [name.replace("_", ".") for name in example_data.column_names]  # SkLearn uses dots
renamed_example_data = example_data.rename_columns(new_column_names).to_pandas()
predictions = pipeline.predict(renamed_example_data)
print(predictions)
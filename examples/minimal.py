from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import duckdb

import orbital
import orbital.types

COLUMNS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

iris = load_iris(as_frame=True)
iris_x = iris.data.set_axis(COLUMNS, axis=1)

# SQL and orbital don't like dots in column names, replace them with underscores
iris_x.columns = COLUMNS = [cname.replace(".", "_") for cname in COLUMNS]

X_train, X_test, y_train, y_test = train_test_split(
    iris_x, iris.target, test_size=0.2, random_state=42
)


pipeline = Pipeline(
    [
        ("preprocess", ColumnTransformer([("scaler", StandardScaler(with_std=False), COLUMNS)],
                                         remainder="passthrough")),
        ("linear_regression", LinearRegression()),
    ]
)
pipeline.fit(X_train, y_train)


orbital_pipeline = orbital.parse_pipeline(pipeline, features={
    "sepal_length": orbital.types.DoubleColumnType(),
    "sepal_width": orbital.types.DoubleColumnType(),
    "petal_length": orbital.types.DoubleColumnType(),
    "petal_width": orbital.types.DoubleColumnType(),
})
print(orbital_pipeline)

sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, projection=orbital.ResultsProjection(["sepal_width"]), dialect="duckdb")
print("\nGenerated Query for DuckDB:")
print(sql)
print("\nPrediction with SQL")
duckdb.register("DATA_TABLE", X_test)
result = duckdb.sql(sql).df()
print(result.head())
print("---")
print(result["variable"][:5].to_numpy())
print("\nPrediction with SciKit-Learn")
print(pipeline.predict(X_test)[:5])

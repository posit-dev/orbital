# Getting Started

Install orbital:

```bash
$ pip install orbital
```

Prepare some data:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

COLUMNS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

iris = load_iris(as_frame=True)
iris_x = iris.data.set_axis(COLUMNS, axis=1)

# SQL and orbital don't like dots in column names, replace them with underscores
iris_x.columns = COLUMNS = [cname.replace(".", "_") for cname in COLUMNS]

X_train, X_test, y_train, y_test = train_test_split(
    iris_x, iris.target, test_size=0.2, random_state=42
)
```

Define a Scikit-Learn pipeline and train it:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline(
    [
        ("preprocess", ColumnTransformer([("scaler", StandardScaler(with_std=False), COLUMNS)],
                                        remainder="passthrough")),
        ("linear_regression", LinearRegression()),
    ]
)
pipeline.fit(X_train, y_train)
```

Convert the pipeline to orbital:

```python
import orbital
import orbital.types

orbital_pipeline = orbital.parse_pipeline(pipeline, features={
    "sepal_length": orbital.types.DoubleColumnType(),
    "sepal_width": orbital.types.DoubleColumnType(),
    "petal_length": orbital.types.DoubleColumnType(),
    "petal_width": orbital.types.DoubleColumnType(),
})
```

You can print the pipeline to see the result:

```python
>>> print(orbital_pipeline)

ParsedPipeline(
    features={
        sepal_length: DoubleColumnType()
        sepal_width: DoubleColumnType()
        petal_length: DoubleColumnType()
        petal_width: DoubleColumnType()
    },
    steps=[
        merged_columns=Concat(
            inputs: sepal_length, sepal_width, petal_length, petal_width,
            attributes: 
             axis=1
        )
        variable1=Sub(
            inputs: merged_columns, Su_Subcst=[5.809166666666666, 3.0616666666666665, 3.7266666666666666, 1.18333333...,
            attributes: 
        )
        multiplied=MatMul(
            inputs: variable1, coef=[-0.11633479416518255, -0.05977785171980231, 0.25491374699772246, 0.5475959...,
            attributes: 
        )
        resh=Add(
            inputs: multiplied, intercept=[0.9916666666666668],
            attributes: 
        )
        variable=Reshape(
            inputs: resh, shape_tensor=[-1, 1],
            attributes: 
        )
    ],
)
```

Now we can generate the SQL from the pipeline:

```python
sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect="duckdb")
```

And check the resulting query:

```python
>>> print(sql)

SELECT ("t0"."sepal_length" - 5.809166666666666) * -0.11633479416518255 + 0.9916666666666668 +  
       ("t0"."sepal_width" - 3.0616666666666665) * -0.05977785171980231 + 
       ("t0"."petal_length" - 3.7266666666666666) * 0.25491374699772246 + 
       ("t0"."petal_width" - 1.1833333333333333) * 0.5475959809777828 
AS "variable" FROM "DATA_TABLE" AS "t0"
```

Once the SQL is generate, you can use it to run the pipeline on a
database. From here on the SQL can be exported and reused in other
places:

```python
>>> print("\nPrediction with SQL")
>>> duckdb.register("DATA_TABLE", X_test)
>>> print(duckdb.sql(sql).df()["variable"][:5].to_numpy())

Prediction with SQL
[ 1.23071715 -0.04010441  2.21970287  1.34966889  1.28429336]
```

We can verify that the prediction matches the one done by Scikit-Learn
by running the scikitlearn pipeline on the same set of data:

```python
>>> print("\nPrediction with SciKit-Learn")
>>> print(pipeline.predict(X_test)[:5])

Prediction with SciKit-Learn
[ 1.23071715 -0.04010441  2.21970287  1.34966889  1.28429336 ]
```

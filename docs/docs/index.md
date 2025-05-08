# Welcome to Orbital

![Orbital](images/orbital.png){ align=left width=52 }

Orbital lets you convert scikit-learn pipelines into pure SQL, 
allowing execution of machine learning models directly in your database, 
without requiring a Python environment.

### Why Orbital?

In regulated, secure, or resource-constrained environments, deploying Python code may be undesirable or even impossible. 
Orbital provides a lightweight alternative: it translates trained sklearn pipelines into SQL expressions that can be audited, 
versioned, and executed entirely inside the database engine.

This enables:

* In-database predictions: avoid moving data or running external inference services
* No dependencies: models run as standard SQL, with no Python runtime required
* Interpretable results: clear formulas for linear models and structured logic for trees
* Security and compliance: auditable, deterministic, dependency-free

### Example

Take a look at the [Examples](https://github.com/posit-dev/orbital/tree/main/examples) 
or follow the [Getting Started](getstarted.md) Guide

```python
# Create a SciKit Learn Pipeline and Train it
pipeline = Pipeline(
    [
        ("preprocess", ColumnTransformer([("scaler", StandardScaler(with_std=False), COLUMNS)],
                                        remainder="passthrough")),
        ("linear_regression", LinearRegression()),
    ]
)
pipeline.fit(X_train, y_train)

# Convert it to an Orbital Pipeline
orbitalml_pipeline = orbitalml.parse_pipeline(pipeline, features={
    "sepal_length": orbitalml.types.DoubleColumnType(),
    "sepal_width": orbitalml.types.DoubleColumnType(),
    "petal_length": orbitalml.types.DoubleColumnType(),
    "petal_width": orbitalml.types.DoubleColumnType(),
})

# Generate SQL
sql = orbitalml.export_sql("DATA_TABLE", orbitalml_pipeline, dialect="duckdb")
```
```
>>> print(sql)
SELECT ("t0"."sepal_length" - 5.809166666666666) * -0.11633479416518255 + 0.9916666666666668 +  
       ("t0"."sepal_width" - 3.0616666666666665) * -0.05977785171980231 + 
       ("t0"."petal_length" - 3.7266666666666666) * 0.25491374699772246 + 
       ("t0"."petal_width" - 1.1833333333333333) * 0.5475959809777828 
AS "variable" FROM "DATA_TABLE" AS "t0"
```

This SQL produces the same predictions as `pipeline.predict(...)` from SciKit-Lear

## Supported Models

OrbitalML currently supports the following models:

-   Linear Regression
-   Logistic Regression
-   Lasso Regression
-   Elastic Net
-   Decision Tree Regressor
-   Decision Tree Classifier
-   Random Forest Classifier
-   Gradient Boosting Regressor
-   Gradient Boosting Classifier


import logging
import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mustela
import mustela.types

PRINT_SQL = False

np.set_printoptions(precision=6, suppress=True)
pd.set_option('display.float_format', '{:.6f}'.format)
#pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.DEBUG)

# Carichiamo il dataset iris e creiamo un DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Creiamo una colonna categorica "petal_width_cat" per usare OneHotEncoder
df["petal_width_cat"] = np.where(df["petal_width"] < 1.0, "narrow", "wide")

# Introduciamo alcuni valori mancanti nella colonna "sepal_width"
df.loc[[0, 10, 20], "sepal_width"] = np.nan

# Usiamo direttamente iris.target come target (multiclasse: 0,1,2)
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_width_cat"]]
y = iris.target

# Costruiamo la pipeline con preprocessamento numerico e categorico
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]), ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["petal_width_cat"])
        ]
    )),
    ("logreg", LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42))
])

pipeline.fit(X, y)

features = mustela.types.guess_datatypes(X)
print("Mustela Features:", features)

mustela_pipeline = mustela.parse_pipeline(pipeline, features=features)
print(mustela_pipeline._model)

# Dati di test
example_data = pa.table({
    "sepal_length": [5.0, 6.1, 7.2, 5.843333],
    "sepal_width": [3.2, 2.8, 3.0, 3.057333],
    "petal_length": [1.2, 4.7, 6.1, 3.758000],
    "petal_width": [0.2, 1.2, 2.3, 1.199333],
    "petal_width_cat": ["narrow", "wide", "wide", "wide"]
})

con = ibis.duckdb.connect()
ibis.set_backend(con)

ibis_expression = mustela.translate(ibis.memtable(example_data), mustela_pipeline)

if PRINT_SQL:
    print("\nGenerated Query:")
    print(con.compile(ibis_expression))

print("\nPrediction with Ibis")
print(ibis_expression.execute())

print("\nPrediction with SKLearn")
test_df = example_data.to_pandas()
pred = pipeline.predict(test_df)
print(pred)
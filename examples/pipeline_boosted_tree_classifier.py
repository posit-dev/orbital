import logging
import ibis
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mustela
import mustela.types

PRINT_SQL = False
logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.DEBUG)

# Carica il dataset Ames Housing e adatta per la classificazione
ames = fetch_openml(name="house_prices", as_frame=True)
ames = ames.frame

# SQL non consente nomi di colonne che iniziano con numeri
ames.columns = ["_" + col if col[0].isdigit() else col for col in ames.columns]

# Selezione delle feature numeriche e categoriche
numeric_features = [
    col
    for col in ames.select_dtypes(include=["int64", "float64"]).columns
    if col != "SalePrice"
]
categorical_features = ames.select_dtypes(include=["object", "category"]).columns

# Conversione delle feature numeriche in float per garantire compatibilità con Mustela
ames[numeric_features] = ames[numeric_features].astype(np.float64)

# Riempimento dei valori mancanti nelle feature categoriche
ames[categorical_features] = ames[categorical_features].fillna("missing")

# Creazione della variabile target classificatoria basata sul prezzo
# Dividiamo le case in 3 fasce di prezzo
def categorize_price(price: float) -> str:
    if price < 130000:
        return "low"
    elif price < 250000:
        return "medium"
    else:
        return "high"

ames["price_category"] = ames["SalePrice"].apply(categorize_price)

# Separa le feature dalla target
X = ames.drop(columns=["SalePrice", "price_category"])
y = ames["price_category"]

# Definizione delle pipeline di trasformazione
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant", missing_values="missing", fill_value="missing"
            ),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Creazione del modello Gradient Boosting Classifier con pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("var_threshold", VarianceThreshold()),
        ("pca", PCA(n_components=5)),
        ("classifier", GradientBoostingClassifier()),
    ]
)

# Addestramento del modello
model.fit(X, y)

# Identificazione delle feature per Mustela
features = mustela.types.guess_datatypes(X)
print("Mustela Features:", features)

# Creazione di un sottoinsieme di dati per la predizione
data_sample = X.head(5)

# Conversione del modello in pipeline SQL con Mustela
mustela_pipeline = mustela.parse_pipeline(model, features=features)
print(mustela_pipeline)

# Generazione della query SQL con Mustela
ibis_expression = mustela.translate(ibis.memtable(data_sample), mustela_pipeline)
con = ibis.sqlite.connect()

if PRINT_SQL:
    print("\nGenerated Query for SQLite:")
    print(con.compile(ibis_expression))

# Predizione con SKLearn
print("\nPrediction with SKLearn")
target = model.predict(data_sample)
print(target)

# Predizione con Ibis
print("\nPrediction with Ibis
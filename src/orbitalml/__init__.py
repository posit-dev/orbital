"""OrbitalML, translate scikit-learn pipelines into SQL queries

OrbitalML is a library for translating **scikit-learn** pipelines
into **SQL queries** and **Ibis expressions**.

It provides a way to execute machine learning models on databases without
the need for a python runtime environment.
"""

from .ast import parse_pipeline
from .sql import export_sql
from .translate import ResultsProjection, translate

__all__ = ["parse_pipeline", "translate", "export_sql", "ResultsProjection"]

"""Translators for trees based models."""

from .classifier import TreeEnsembleClassifierTranslator
from .regressor import TreeEnsembleRegressorTranslator

__all__ = ["TreeEnsembleClassifierTranslator", "TreeEnsembleRegressorTranslator"]

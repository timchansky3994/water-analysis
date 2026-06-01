"""Scope selection and analytical engines."""

from water_analysis.analysis.correlations import CorrelationAnalysis, CorrelationDiagnostic, run_correlation_analysis
from water_analysis.analysis.feature_selection import FeatureSelectionResult, prepare_modeling_frame, select_predictors
from water_analysis.analysis.scopes import ScopeSlice, build_scope_slices

__all__ = [
    "CorrelationAnalysis",
    "CorrelationDiagnostic",
    "FeatureSelectionResult",
    "ScopeSlice",
    "build_scope_slices",
    "prepare_modeling_frame",
    "run_correlation_analysis",
    "select_predictors",
]

"""PinchPy: Python library for Pinch Analysis and Heat Exchanger Network analysis."""

from .models.stream import Stream, StreamTypeError, StreamValidationError, StreamType
from .models.exchanger import HeatExchanger
from .models.network import HeatExchangerNetwork
from .analysis.pinch_point import utility_targets, pinch_temperature, pinch_temperatures
from .analysis.heat_cascade import heat_cascade
from .analysis.benchmarks import BENCHMARK_CASES, get_benchmark
from .synthesis.network_design import design_network
from .synthesis.network_evaluation import evaluate_network, validate_network_pinch_rules
from .synthesis.pinch_split import split_streams_at_pinch
from .economics.cost import ExchangerCostModel, UtilityCostModel, cost_analysis
from .economics.emissions import EmissionsModel, emissions_analysis, FUEL_EMISSION_FACTORS
from .economics import economic_and_environmental_analysis
from .retrofit.diagnosis import diagnose_network, PinchViolation
from .retrofit.recommendations import plan_retrofit, RetrofitProject, RetrofitModification
from .retrofit import retrofit_analysis
from .reporting.report import (
    generate_report,
    export_report_html,
    export_report_markdown,
    export_report_json,
)
from .utils.io import (
    load_streams_from_csv,
    save_streams_to_csv,
    load_streams_from_json,
    save_streams_to_json,
)
from .ml.dataset import generate_case, generate_dataset
from .ml.train import train_ann, build_ann
from .ml.predict import predict_targets, predict_case
from .ml.evaluate import evaluate_regression

__version__ = "0.1.0"

__all__ = [
    "Stream",
    "StreamType",
    "StreamValidationError",
    "StreamTypeError",
    "HeatExchanger",
    "HeatExchangerNetwork",
    "utility_targets",
    "pinch_temperature",
    "pinch_temperatures",
    "heat_cascade",
    "BENCHMARK_CASES",
    "get_benchmark",
    "design_network",
    "evaluate_network",
    "validate_network_pinch_rules",
    "split_streams_at_pinch",
    "ExchangerCostModel",
    "UtilityCostModel",
    "cost_analysis",
    "EmissionsModel",
    "emissions_analysis",
    "FUEL_EMISSION_FACTORS",
    "economic_and_environmental_analysis",
    "diagnose_network",
    "PinchViolation",
    "plan_retrofit",
    "RetrofitProject",
    "RetrofitModification",
    "retrofit_analysis",
    "generate_report",
    "export_report_html",
    "export_report_markdown",
    "export_report_json",
    "load_streams_from_csv",
    "save_streams_to_csv",
    "load_streams_from_json",
    "save_streams_to_json",
    "generate_case",
    "generate_dataset",
    "train_ann",
    "build_ann",
    "predict_targets",
    "predict_case",
    "evaluate_regression",
    "__version__",
]

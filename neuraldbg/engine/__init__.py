"""
NeuralDBG Causal Inference Engine.
Bundled with NeuralDBG. MIT License.
"""

from .gradient import GradientAnalyzer
from .activation import ActivationAnalyzer
from .data import DataAnalyzer
from .coupling import CouplingDetector
from .explain import Explanator


class CausalEngine:
    def __init__(self, dbg):
        self.dbg = dbg
        self.gradient = GradientAnalyzer(dbg)
        self.activation = ActivationAnalyzer(dbg)
        self.data = DataAnalyzer(dbg)
        self.coupling = CouplingDetector(dbg)
        self.explain = Explanator(dbg)

    def detect_gradient_transition(self, prev_norm, current_norm):
        return self.gradient.detect_transition(prev_norm, current_norm)

    def classify_gradient_health(self, norm):
        return self.gradient.classify_health(norm)

    def classify_activation_health(self, stats):
        return self.activation.classify_health(stats)

    def classify_data_health(self, tensor):
        return self.data.classify_health(tensor)

    def check_data_anomaly(self, tensor, layer_name):
        return self.data.check_anomaly(tensor, layer_name)

    def detect_coupled_failures(self, window=5):
        return self.coupling.detect(window)

    def explain_failure(self, failure_type="vanishing_gradients"):
        return self.explain.explain(failure_type)

    def get_root_causes(self):
        return self.explain.get_root_causes()

    def trace_causal_chain(self, event_type):
        return self.explain.trace_causal_chain(event_type)

    def get_causal_hypotheses(self):
        return self.explain.get_causal_hypotheses()

    def export_aquarium_package(self, package_path):
        return self.explain.export_aquarium_package(package_path)

    def export_mermaid_causal_graph(self):
        return self.explain.export_mermaid_causal_graph()

    def collapse_events(self):
        return self.explain.collapse_events()

    def track_first_occurrence(self, failure_type, layer_name):
        self.dbg._track_first_occurrence(failure_type, layer_name)

    def event_matches_failure_key(self, event, failure_key):
        return self.explain.event_matches_failure_key(event, failure_key)

    def detect_activation_shift(self, prev_stats, current_stats):
        return self.activation.detect_shift(prev_stats, current_stats)

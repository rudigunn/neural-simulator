"""Graph conversion utilities."""

from neural_simulator.graphs.graph_dict import scenario_to_graph_dict
from neural_simulator.graphs.template import build_template, template_to_pyg_data

__all__ = ["build_template", "scenario_to_graph_dict", "template_to_pyg_data"]

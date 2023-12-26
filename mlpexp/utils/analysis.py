import plotly.graph_objects as go

from plotly.subplots import make_subplots
from torch.nn import Linear, Module
from torch import Tensor

from typing import List, Tuple


def get_MLP_neuron_values(model: Module, X: Tensor) -> Tuple[List, List]:
    pre_activation, post_activation = [], []
    for layer in model.layers:
        X = layer(X)
        if isinstance(layer, Linear):
            pre_activation.append(X.detach())
        else:
            post_activation.append(X.detach())
    return pre_activation, post_activation

def display_MLP_neuron_values(neuron_values: List[Tensor]) -> None:
    fig = make_subplots(
        rows=len(neuron_values), 
        cols=1)
    
    for i, output in enumerate(neuron_values):
        np_out = output.detach().cpu().numpy()
        normalized = (np_out - np_out.min()) / (np_out.max() - np_out.min())
        trace = go.Heatmap(
            z=normalized,
            colorscale='Viridis')
        fig.add_trace(trace, row=i+1, col=1)

    return fig


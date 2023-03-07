"""A module that contains visualisation functions and helpers"""

import numpy as np
import plotly.graph_objects as go
import torch

from . import radial_spherical_tensor


def visualise(
        rst: radial_spherical_tensor.RadialSphericalTensor,
        rcut: float,
        signal: torch.tensor,
        peak_points=None
):
    with torch.no_grad():
        sample_points_linear, signal_on_grid = rst.signal_on_grid(signal, rcut, 50, crop_bases=True)

        X, Y, Z = np.meshgrid(sample_points_linear, sample_points_linear, sample_points_linear, indexing='ij')
        max_val = torch.max(torch.abs(signal_on_grid)).item()

        layout = go.Layout(width=500, height=500, margin=dict(l=0, r=0, t=10, b=0), )

        data = []
        if peak_points is not None:
            trace_points = go.Scatter3d(
                x=peak_points[:, 0],
                y=peak_points[:, 1],
                z=peak_points[:, 2],
                mode='markers',
            )
            data.append(trace_points)

        trace_volume = go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=signal_on_grid.flatten(),
            isomin=max_val * 0.25,
            isomax=max_val,
            opacity=0.3,
            surface_count=3,
        )
        data.append(trace_volume)

        fig = go.Figure(data=data, layout=layout)
        fig.show()

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_target_label(bin_edges: np.array, y_train: pd.Series):
    # Create histogram with manually set bins
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=y_train,
        xbins=dict(
            start=bin_edges[0],  # Ensure bins start from 0
            end=bin_edges[-1],  # Maximum bin range
            size=bin_edges[1] - bin_edges[0]   # Bin width
        ),
        marker=dict(color='blue', line=dict(width=1)),
        opacity=0.75
    ))

    # Update layout and fix hover text formatting
    fig.update_layout(
        title="Distribution of Target Variable in Training Set",
        xaxis_title="Target Variable",
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.1,
        xaxis=dict(
            tickmode='array',
            tickvals=bin_edges[:-1],  # Place ticks at the bin starts
        )
    )

    fig.show()

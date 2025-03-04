import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

def plot_corr_with_label(df: pd.DataFrame, target_label: pd.Series, top_n_features: int=10) -> None:
    # Compute correlation matrix with the target variable
    correlation_matrix = df.copy()
    correlation_matrix["target"] = target_label  # Add target for correlation check
    corr_with_target = np.round(correlation_matrix.corr()["target"].drop("target").sort_values(ascending=False),2).head(10)

    # Create correlation heatmap using Plotly
    fig = ff.create_annotated_heatmap(
        z=[corr_with_target.values],
        x=corr_with_target.index.tolist(),
        y=["Correlation with Target"],
        colorscale="Viridis",
        showscale=True,
    )

    # Show the figure
    fig.show()


def plot_features_dist(df: pd.DataFrame) -> None:
    # Create a dropdown menu with all features
    features = df.columns

    # Create the figure
    fig = go.Figure()

    # Add an initial trace (first feature)
    fig.add_trace(go.Histogram(x=df[features[0]], name=features[0], nbinsx=50))

    # Update layout with dropdown
    fig.update_layout(
        title=f"Feature Distribution: {features[0]}",
        xaxis_title=features[0],
        yaxis_title="Count",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [{"x": [df[feature]], "name": feature}],
                        "label": feature,
                        "method": "restyle",
                    }
                    for feature in features
                ],
                "direction": "down",
                "showactive": True,
            }
        ],
    )

    # Show the figure
    fig.show()

def plot_target_label(bin_edges: np.array, y_train: pd.Series) -> None:
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

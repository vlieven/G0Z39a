from typing import cast

import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html
from flask import Flask

from .components import county_choropleth

app: Dash = Dash(__name__)
server: Flask = cast(Flask, app.server)

colors = {"background": "#111111", "text": "#7FDBFF"}

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Covid evolution predictor",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="An interactive app to predict Covid evolution based on government response.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Graph(id="choropleth_map"),
    ],
)


@app.callback(  # type: ignore[misc]
    Output("graph", "figure"),
    Input("candidate", "value"),
)
def display_choropleth() -> go.Figure:
    df = pd.DataFrame()
    fig = county_choropleth(df)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)

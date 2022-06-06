from typing import cast

import pandas as pd
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from flask import Flask

from nepal.app.components import county_choropleth, slider, navbar
from nepal.app.data import MasterData, Predictions
from nepal.ml.transformers import TargetTransform

app: Dash = Dash(
    __name__,
    title="Covid-19 Dashboard",
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)
server: Flask = cast(Flask, app.server)

transform: TargetTransform = TargetTransform()
master: MasterData = MasterData(transform)
predictions: Predictions = Predictions(transform)


def load_predictions(
    stringency_index: float,
    gov_response_index: float,
    containment_health_index: float,
    economic_support_index: float,
) -> pd.DataFrame:
    exogenous: pd.DataFrame = master.exogenous()
    exogenous["StringencyIndex"] = stringency_index
    exogenous["GovernmentResponseIndex"] = gov_response_index
    exogenous["ContainmentHealthIndex"] = containment_health_index
    exogenous["EconomicSupportIndex"] = economic_support_index

    return predictions.load(
        endogenous=master.target(),
        exogenous=exogenous
    )


controls = dbc.Card(
    [
        slider("Stringency Index", id="stringency_index"),
        slider("Government Response Index", id="gov_response_index"),
        slider("Containment Health Index", id="containment_health_index"),
        slider("Economic Support Index", id="economic_support_index"),
    ],
    body=True,
)


app.layout = html.Div(
    children=[
        navbar("Covid-19 simulation"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(controls, md=4),
                        dbc.Col(
                            dcc.Graph(
                                id="choropleth_graph",
                                figure=county_choropleth(load_predictions(0.5, 0.5, 0.5, 0.5)),
                            ),
                            md=8,
                        ),
                    ],
                    align="center",
                ),
            ],
            fluid=True,
        ),
    ]
)


@app.callback(  # type: ignore[misc]
    Output("choropleth_graph", "figure"),
    [
        Input("stringency_index", "value"),
        Input("gov_response_index", "value"),
        Input("containment_health_index", "value"),
        Input("economic_support_index", "value"),
    ],
)
def display_choropleth(
    stringency_index: float,
    gov_response_index: float,
    containment_health_index: float,
    economic_support_index: float,
) -> go.Figure:
    df = load_predictions(stringency_index, gov_response_index, containment_health_index, economic_support_index)
    fig = county_choropleth(df)

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


if __name__ == "__main__":
    app.run(debug=True)

import dash_bootstrap_components as dbc
from dash import dcc, html


def slider(label: str, *, id: str) -> html.Div:
    return html.Div(
        [
            dbc.Label(label),
            dcc.Slider(
                id=id,
                min=0,
                max=100,
                step=5,
                value=50,
            ),
        ]
    )

import json
from functools import lru_cache
from typing import Any, Mapping
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


@lru_cache(maxsize=None)
def load_geojson_counties() -> Mapping[str, Any]:
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        counties: Mapping[str, Any] = json.load(response)
    return counties


def state_choropleth(df: pd.DataFrame) -> go.Figure:
    fig: go.Figure = px.choropleth(
        df,
        scope="usa",
        locationmode="USA-states",
        locations=df.index.get_level_values(level="state"),
        color="infections_per_10000",
        color_continuous_scale="orrd",
        range_color=[0, 35],
        labels={"infections_per_10000": "Active cases per 10.000"},
        animation_frame=df.index.get_level_values(level="date"),
        hover_name="name",
    ).update_layout(sliders=[{"currentvalue": {"prefix": "Date: "}}])

    return fig

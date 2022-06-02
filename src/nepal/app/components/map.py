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


def county_choropleth(df: pd.DataFrame) -> go.Figure:
    counties: Mapping[str, Any] = load_geojson_counties()
    fig: go.Figure = px.choropleth(
        df,
        geojson=counties,
        locations="fips",
        color="unemp",
        color_continuous_scale="Viridis",
        range_color=(0, 12),
        scope="usa",
        labels={"unemp": "unemployment rate"},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

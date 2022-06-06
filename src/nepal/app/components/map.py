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
        # fitbounds="geojson",
        locations=df.index.get_level_values(level="fips"),
        color="new_cases",
        color_continuous_scale="orrd",
        scope="usa",
        labels={"new_cases": "Additional Covid cases"},
        animation_frame=df.index.get_level_values(level="date"),
        # animation_group=df.index.get_level_values(level="fips"),
    ).update_layout(sliders=[{"currentvalue": {"prefix": "Date: "}}])

    return fig

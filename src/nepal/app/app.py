from typing import cast

import plotly
from dash import Dash
from flask import Flask

app: Dash = Dash(__name__)
server: Flask = cast(Flask, app.server)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)

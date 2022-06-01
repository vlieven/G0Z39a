import dash
import plotly


def load_app() -> dash.Dash:
    return dash.Dash(__name__)


if __name__ == "__main__":
    app: dash.Dash = load_app()
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)

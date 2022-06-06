import dash_bootstrap_components as dbc


def navbar(brand: str) -> dbc.NavbarSimple:
    return dbc.NavbarSimple(
        brand=brand,
        brand_href="#",
        color="primary",
        dark=True,
        sticky="top",
        fluid=True,
        key="simple_navigation_bar",
    )

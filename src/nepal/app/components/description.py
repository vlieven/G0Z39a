from dash import html


def descriptions() -> html.Div:
    return html.Div(
        [
            html.H4("Policy Indices"),
            html.H5("Government Response Index"),
            html.P(
                "The overall response of governments, becoming stronger or weaker over the course of the outbreak."
            ),
            html.H5("Containment and Health Index"),
            html.P(
                "The index combines ‘lockdown’ restrictions and closures with measures such as testing policy and contact tracing, short term investment in healthcare, as well investments in vaccines."
            ),
            html.H5("Stringency Index"),
            html.P(
                "The index records the strictness of ‘lockdown style’ policies that primarily restrict people’s behaviour, along with public information campaigns."
            ),
            html.H5("Economic Support Index"),
            html.P("The index records measures such as income support and debt relief."),
        ]
    )

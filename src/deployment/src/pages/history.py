import dash
from dash import html

dash.register_page(__name__, path  = "/history", name = "Historial")

layout = html.Div([
    html.H1('This is our Archive page'),
    html.Div('This is our Archive page content.'),
])
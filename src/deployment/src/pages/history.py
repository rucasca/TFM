# pages/about.py
import dash
from dash import html

dash.register_page(__name__, path="/about")

layout = html.Div([
    html.H1("About Page"),
    html.P("This is the about page.")
])

# pages/contact.py
import dash
from dash import html

dash.register_page(__name__, path="/contact")

layout = html.Div([
    html.H1("Contact Page"),
    html.P("Get in touch with us!")
])

import dash
from dash import html

dash.register_page(__name__, path  = "/results", name = "Ejemplos")

path = r"C:\Users\ruben\Desktop\code_tfm\src\deployment\src\outputs"

layout = html.Div([
    html.H1('This is our Archive page'),
    html.Div('This is our Archive page content.'),
])
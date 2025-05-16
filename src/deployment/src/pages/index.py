# index.py
from dash import html, dcc, page_container, page_registry
import dash
from app import app

app.layout = html.Div([
    html.Nav([
        dcc.Link("Home", href="/", style={"marginRight": "15px"}),
        dcc.Link("About", href="/about", style={"marginRight": "15px"}),
        dcc.Link("Contact", href="/contact"),
    ], style={"padding": "10px", "backgroundColor": "#f0f0f0"}),

    html.Hr(),
    page_container  # This displays the current page
])

if __name__ == "__main__":
    app.run_server(debug=True)

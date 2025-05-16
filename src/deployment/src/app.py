import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)

navbar =   html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"]),
            style={"margin": "10px 0"}
        ) for page in dash.page_registry.values()
    ], className= "navbar"),


app.layout = html.Div([
    navbar, 
    html.Div(dash.page_container , className= "content")
    
], style = "web-layout")

if __name__ == '__main__':
    app.run(debug=True)
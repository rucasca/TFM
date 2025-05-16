import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)
print("loaded pages are", dash.page_registry.values())


icons = ["fas fa-robot","fas fa-clock-rotate-left"]

navbar =   html.Div([
        html.Div(

            html.Img(src='/assets/image.jpg', style={
                'width': '80%',  
                'display': 'block',  
                'margin': '0 auto' 
            }),

            dcc.Link(
                     html.Div([
                         html.I(className= icons[i]),
                         html.Span(page['name'])
                     ], className= "navbar-entry"),
                    href=page["relative_path"], className="navbar-link"),
            style={"margin": "10px 0"}
        ) for i, page in enumerate(dash.page_registry.values())
    ], className= "navbar"),


app.layout = html.Div([
    navbar, 
    html.Div(dash.page_container , className= "content")
    
], style = "web-layout")

if __name__ == '__main__':
    app.run(debug=True)
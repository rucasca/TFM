import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import io
from PIL import Image
import numpy as np

dash.register_page(__name__, path  = "/", name = "Inferencia")

models = ["Retinanet + SAM", "UNET"]


layout = html.Div([
    html.H1('Generador de segmentaciones'),
    dcc.Store(id='image-container'),
    html.Div([
        html.P('Generador automático de segmentaciones semánticas mediante modelos tipo ensemble '),
        html.P("Indique la configuración a aplicar:"),
        dbc.Container(
            dbc.Card(
                dbc.CardBody(
                    dbc.Row([

                        dbc.Col([
                            html.Label("Modelo seleccionado", className="fw-bold mb-2"),
                            dcc.Dropdown(id="dropdown-model-selected", label="Clases base", value=True)
                        ], width=4),          


                        dbc.Col([
                            html.Label("Clases base", className="fw-bold mb-2"),
                            dbc.Switch(id="switch-is-default-class", label="Clases base", value=True)
                        ], width=4),

                        # dbc.Col([
                        #     html.Label("Sensibilidad", className="fw-bold mb-2"),
                        #     dcc.Slider(id="slider-sensibility", label="Sensibilidad del ", value = 50,min = 0.1, max = 99.9, step= 0.1, tooltip={"placement": "bottom", "always_visible": True})
                        # ], width=4),

                        dbc.Col([
                            dbc.Button(id="buttoon-inference", label="Activo", value=True, style={"display":"None"}, className="button-inference")
                        ], width=4),
                    ])
                ),
                className="shadow-sm rounded",
                style={"width": "90%", "margin": "2rem auto", "backgroundColor": "white"}
            ),
            fluid=True,
            className="d-flex justify-content-center align-items-center vh-100"
        ),
        html.Div([
        dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Arrastre o ',
                    html.A('seleccione una imagen'),
                    "(formatos soportados: png y jpg)"
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ])
    ], id = "layout")

])



@callback(Output('output-data-upload', 'children'),
          Output('buttoon-inference', 'style'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
        )
def allow_inference(contents, filename):
    if contents is None:
        return "No file uploaded", {"display": "none"}

    # Extract file extension and validate
    if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
        return "Only PNG and JPG files are supported.", {"display": "none"}
    try:
        # Decode base64 content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Load image and convert to numpy array
        image = Image.open(io.BytesIO(decoded)).convert('RGB')
        np_array = np.array(image)

        # Store the array using the filename as a key (can use something else if needed)
        image_store[filename] = np_array

        return f"Uploaded image: {filename}", {"display": "inline-block"}

    except Exception as e:
        return f"Error processing image: {str(e)}", {"display": "none"}

@callback(Output('layout', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
        )
def generate_inference(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



# <button><i class="fas fa-upload"></i> Upload</button> fa-cloud-upload-alt
# <button><i class="fas fa-image"></i> Picture</button>
# <button><i class="fas fa-floppy-disk"></i> Save</button> fas fa-floppy-disk
# <button><i class="fas fa-clock-rotate-left"></i> History</button> fa-history
# <button><i class="fas fa-plus"></i> Add</button> fa-circle-plus
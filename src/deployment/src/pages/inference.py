import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from PIL import Image
import numpy as np
import os
import dash_daq as daq
from utils_dash.model_inference import inference_model_pipeline
import json

dash.register_page(__name__, path  = "/", name = "Inferencia")

MODELS = {"Base U-Net":"unet_base",
          "RetinaNet + SAM": "retinanet_sam",
          "Yolov8 + SAM": "yolo_sam",
          "CLIP + SAM": "sam_clip",
          "RetinaNet + SAM + U-Net": "final_model"}



STORE_IMG = None
RESULT_INFERENCE = None
FILENAME = None

DIR_CONSTANTS = r"inputs\constants.json"

with open(DIR_CONSTANTS, 'r') as file:
    CONSTANTS =json.load(file)
    CONSTANTS["cons_threshold"] = 0.5


OBJECTIVES = CONSTANTS["objetives"]
CATEGORIES = CONSTANTS["categories"]

ID_OBJECTIVES = CONSTANTS["id_objetives"]
CATEGORY_INFO_OBJECTIVE = CONSTANTS["category_info_objetive"]

DICT_CLASS_INDEX = CONSTANTS["dict_class_index"]
CONS_TRHESHOLD = CONSTANTS["cons_threshold"]


#####   DEFAULT VALUES   #######

CONS_CAT_INDEX_BY_NAME= {'background': 0,
 'person': 1,
 'car': 2,
 'motorcycle': 3,
 'bus': 4,
 'traffic light': 5,
 'backpack': 6,
 'handbag': 7,
 'chair': 8,
 'dining table': 9,
 'cell phone': 10}

CONS_INFO_OBJ = {1: 'person',
 3: 'car',
 4: 'motorcycle',
 6: 'bus',
 10: 'traffic light',
 27: 'backpack',
 31: 'handbag',
 77: 'cell phone',
 62: 'chair',
 67: 'dining table',
 0: 'background'}


CONS_DIV_ERROR_CASE1 = html.Div("❌ Formato no soportado (admite .png y .jpg)", style={
                'color': 'rgb(211, 47, 47)',
                'fontWeight': 'bold',
                'padding': '10px',
                'border': '1px solid rgb(211, 47, 47)',
                'borderRadius': '5px',
                'backgroundColor': 'rgb(255, 235, 238)',
                "margin-top": "10px",
            })

CONS_DIV_ERROR_CASE2 = html.Div("❌ Error en el procesamiento del fichero", style={
                'color': 'rgb(211, 47, 47)',
                'fontWeight': 'bold',
                'padding': '10px',
                'border': '1px solid rgb(211, 47, 47)',
                'borderRadius': '5px',
                'backgroundColor': 'rgb(255, 235, 238)',
                "margin-top": "10px",
            })


layout = html.Div([
    html.H1('Generador de segmentaciones'),
    dcc.Store(id='image-container'),
    dcc.Loading(
        type="default",
        children = html.Div([

        html.P('Generador automático de segmentaciones semánticas mediante modelos tipo ensemble ', className="p-info" ),
        
        dbc.Container([
            

            dbc.Card([

                html.P("Configuración a aplicar:", className= "p-info", style = {"margin":" 0px 0px 20px 0px","font-size": "14px"}),
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Modelo seleccionado", className="fw-bold mb-2"),
                                    dcc.Dropdown(id="dropdown-model-selected", value=list(MODELS.keys())[0], options=list(MODELS.keys()), style={"minWidth": "200px", "maxWidth": "300px"})
                                ],
                                style = {"maxWidth": "300px"},
                                className="d-flex flex-column justify-content-center me-1"
                            ),
                            dbc.Col([
                                html.Label("Segmentación ligera", className="fw-bold mb-2"),
                                daq.BooleanSwitch(
                                    id="switch-is-default-class", 
                                    on=False,
                                    className="my-auto",
                                    color ="#1E1E1E",
                                )],
                                style = {"maxWidth": "300px"},
                                className="d-flex align-items-center me-1"
                            ),
                            dbc.Col(
                                dbc.Button(
                                    html.Div([html.I(className= "fas fa-microchip"), html.Span("Calcular Inferencia")]),
                                    id="buttoon-inference",
                                    style={"display": "none"},
                                    className="button-inference"
                                ),
                                style = {"maxWidth": "300px"},
                                className="d-flex align-items-center"
                            ),
                        ],
                        className="d-flex flex-row align-items-center mb-3",
                        style={"display": "flex","justify-content": "space-between","align-items": "center"}
                    )
                )],
                className="shadow-sm rounded",
                
            )],
            fluid=True,
            className="d-flex justify-content-center align-items-center vh-100 container-settings",
            
        ),
        html.Div([
            html.Div(id='output-data-upload'),
            dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastre o ',
                        html.A('seleccione una imagen'),
                        " (formatos soportados: png y jpg)"
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        "margin": "20px 0px"
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),

                html.Div(

                    className= "button_save", id = "save-inference", style = {"display": "none"}
                )


            
        ])
    ], id = "layout")

    )

])



@callback(Output('output-data-upload', 'children'),
          Output('buttoon-inference', 'style'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
        )
def allow_inference(contents, filename):


    print("image trying to be saved")

    global STORE_IMG
    global FILENAME

    if contents is None:
        return dash.no_update, {"display": "none"}

    if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
        return CONS_DIV_ERROR_CASE1, {"display": "none"}
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        image = Image.open(io.BytesIO(decoded)).convert('RGB')
        np_array = np.array(image)

        STORE_IMG = np_array
        FILENAME = filename

        log_success = html.Div(f"✅ Imagen cargada: {filename}", style={
        'color': '#155724',
        'backgroundColor': '#d4edda',
        'border': '1px solid #c3e6cb',
        'padding': '10px',
        'borderRadius': '5px',
        'fontWeight': 'bold',
        'marginTop': '10px'
    })

        return log_success, {"display": "inline-block"}

    except Exception as e:
        return CONS_DIV_ERROR_CASE2,  {"display": "none"}

@callback(Output('layout', 'children'),
              Input('buttoon-inference', 'n_clicks'),
              State('dropdown-model-selected', 'value'),
              State('switch-is-default-class', 'on'),

        )
def generate_inference(n_clicks, model, has_all_classes):

    if(n_clicks == 0  or n_clicks == None):
        return dash.no_update

    global STORE_IMG
    

    fig1, fig2 =  get_plots_inference(STORE_IMG , model, has_all_classes, class_names=None)


    return  [
        html.H2(f"Imagen y resultado de la inferencia empleando {model}"),
        html.Div([
            dcc.Graph(figure=fig1, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig2, style={'width': '48%', 'display': 'inline-block'})
        ]),

        html.Div(
            html.I(id = "save", className = "fa-floppy-disk" ),
            html.Span("Guardar resultados"),
            className= "button_save", id = "save-inference"
        )


    ]   



def get_plots_inference(image, selected_model, has_all_classes, class_names):
    
    fig_rgb = px.imshow(image)
    fig_rgb.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    #print("processing inference")
    class_map = inference_model_pipeline(image = image, model = MODELS[selected_model])

    ## TODO: include inference with more models
    #print("generating plot output")
    hover_text = np.vectorize(class_names.get)(class_map)

    # Create class map figure with hover
    fig_class = go.Figure(data=go.Heatmap(
        z=class_map,
        text=hover_text,
        hoverinfo='text',
        colorscale='Viridis',
        colorbar=dict(title='Class')
    ))
    fig_class.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    

    global RESULT_INFERENCE
    RESULT_INFERENCE = class_map


    return fig_rgb,fig_class



@dash.callback(
    Output("url", "pathname"),
    Input('save-inference', 'n_clicks')

)
def save_results(n_clicks):



    global STORE_IMG
    global RESULT_INFERENCE
    global FILENAME

    if(n_clicks == 0 or n_clicks == None):
        return dash.no_update

    output_path = r"C:\Users\ruben\Desktop\code_tfm\src\deployment\src\outputs"
    filename = filename.split(".")[0] + ".npz"
    full_path = os.path.join(output_path, filename)
    np.savez(full_path, image=STORE_IMG, inference=RESULT_INFERENCE)



    return "/history"


# <button><i class="fas fa-upload"></i> Upload</button> fa-cloud-upload-alt
# <button><i class="fas fa-image"></i> Picture</button>
# <button><i class="fas fa-floppy-disk"></i> Save</button> fas fa-floppy-disk
# <button><i class="fas fa-clock-rotate-left"></i> History</button> fa-history
# <button><i class="fas fa-plus"></i> Add</button> fa-circle-plus

# loaded = np.load(full_path)
# array1_loaded = loaded['array1']
# array2_loaded = loaded['array2']







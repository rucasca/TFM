import dash
from dash import html
import os
import dash_bootstrap_components as dbc
from dash import html, Output, Input, dcc, ctx
import plotly.graph_objects as go
import datetime

import numpy as np
from PIL import Image
import io
import base64
import plotly.express as px
import json

dash.register_page(__name__, path  = "/history", name = "Historial")


path_input = r"outputs"
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



def numpy_to_base64(img_array):
    image = Image.fromarray(img_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def create_card(image_data, title ,id_value ):

    image = Image.fromarray(image_data.astype('uint8'))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    data_url = f"data:image/png;base64,{encoded_image}"


    # card =  dbc.Card(
    #     [
    #         dbc.CardImg(src=data_url, top=True),
    #         dbc.CardBody(
    #             html.H5(title, className="card-title")
    #         ),
    #     ],
    #     style={"width": "100%", "margin-bottom": "20px"},
    #     id={"type": "selected-card", "index": id},
    #     n_clicks = 0
    # )

    result =  html.Div(
        dbc.Card(
            [
                dbc.CardImg(src=data_url, top=True, style={"width": "100%",
                    "aspect-ratio": "4 / 3",
                    "object-fit": "cover",
                    "border-top-left-radius": "0.5rem",
                    "border-top-right-radius": "0.5rem",
                },),
                dbc.CardBody(html.H5(title, className="card-title", style={"margin": "0","text-align": "center"}), style={"padding": "1rem"},),
            ],
            style={
            "width": "100%",
            "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "border-radius": "0.5rem",
            "overflow": "hidden",
            "background-color": "#ffffff",
            "margin-bottom": "20px",
            "transition": "transform 0.2s",
        },
        ),
        id=id_value,
        n_clicks=0,
        style={"cursor": "pointer",
            "width": "100%",
            "padding": "10px",
            "transition": "transform 0.2s",}
    )


    return result




def load_saved_inferences():
    images = []
    titles = []
    for i, archivo in enumerate(os.listdir(path_input)):
        if archivo.endswith(".npz"):
            ruta = os.path.join(path_input, archivo)
            dict_np_array = np.load(ruta)

            imagen = dict_np_array['image']
            #encoded_image = numpy_to_base64(imagen)
            images.append(imagen)


            model_name = archivo.split("_")[0]

            date_str = archivo.split("_")[1]
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%d/%m/%Y") 


            titles.append(f"Inferencia de {model_name} ({formatted_date})")

    rows = []
    # for i in range(0, len(images), 3):
    #     row = dbc.Row(
    #         [
    #             dbc.Col(create_card(img, title = titles[i+j],   id_value = {"type": "selected-card", "index": i+j}), width=4)
    #             for j, img in enumerate(images[i:i+3])
    #         ],
    #         className="mb-4",
    #     )
    #     rows.append(row)

    rows = []

    for i in range(0, len(images), 3):
        row_cards = [
            html.Div(
                create_card(
                    img,
                    title=titles[i + j],
                    id_value={"type": "selected-card", "index": i + j}
                ),
                style={
                    "flex": "1 1 30%",
                    "margin": "10px",
                    "boxSizing": "border-box",
                    "maxWidth": "30%",
                },
            )
            for j, img in enumerate(images[i:i + 3])
        ]

        row = html.Div(
            row_cards,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "space-between",
                "marginBottom": "20px",
            }
        )
        rows.append(row)

    return html.Div(rows, style={"width": "100%"})


def generate_plots_modal(image, inference):
    fig_rgb = px.imshow(image)
    fig_rgb.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    hover_text = np.vectorize(lambda x: f"{CATEGORY_INFO_OBJECTIVE.get(x, 'None')}")(inference)

    fig_class = go.Figure(data=go.Heatmap(
        z=inference,
        text=hover_text,
        hoverinfo='text',
        colorscale='Viridis',
        colorbar=dict(title='Clase:')
    ))
    fig_class.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig_rgb, fig_class



layout = html.Div([
    html.H1('Resultados guardados'),
    html.P("El histórico de resultados ha sido:"),
    load_saved_inferences(),
    dbc.Modal(id = "modal-img", size="xl",is_open=False,backdrop="static" )
])




@dash.callback(
    Output("modal-img", "is_open"),
    Output("modal-img", "children"),
    Input({"type": "selected-card", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def on_card_click(n_clicks_list):

    if(not(any(n_clicks_list))):
        return False,dash.no_update
        
    triggered_id = ctx.triggered_id
    if triggered_id is None:
        return False, dash.no_update
    
    
    # print(f"{triggered_id}")
    index = triggered_id["index"]
    files = sorted(os.listdir(path_input))
    # print("index", index)
    # print("files", files)
    target_file = files[index]

    ruta = os.path.join(path_input, target_file)
    # print(f"{np.load(ruta).files=}")
    dict_np_array = np.load(ruta)
    imagen = dict_np_array['image']
    inference = dict_np_array['result']

    model_name = target_file.split('_')[0]

    plot1, plot2 = generate_plots_modal(imagen, inference)

    results =  [
    dbc.ModalHeader(
        dbc.ModalTitle("Resultados de la inferencia en {}".format(target_file), className="w-100 text-center"),
        close_button=True
    ),
    dbc.ModalBody([
        html.H4(f"Resultados del modelo {model_name}", className="text-center mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=plot1), width=6),
            dbc.Col(dcc.Graph(figure=plot2), width=6),
        ])
    ])
]

    return True, results

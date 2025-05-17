import dash
from dash import html
import os
import dash_bootstrap_components as dbc
from dash import html, Output, Input, dcc, ctx
import plotly.graph_objects as go

from dash import html
import numpy as np
from PIL import Image
import io
import base64
import plotly.express as px


dash.register_page(__name__, path  = "/history", name = "Historial")


path_input = r"C:\Users\ruben\Desktop\code_tfm\src\deployment\src\outputs"




def numpy_to_base64(img_array):
    image = Image.fromarray(img_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def create_card(image_data, id ):
    card =  dbc.Card(
        [
            dbc.CardImg(src=numpy_to_base64(image_data["array"]), top=True),
            dbc.CardBody(
                html.H5(image_data["title"], className="card-title")
            ),
        ],
        style={"width": "100%", "margin-bottom": "20px"},
    )


    return html.Div(card, id =  id, n_clicks= 0 )




def load_saved_inferences():
    images = []
    for archivo in os.listdir(path_input):
        if archivo.endswith(".npz"):
            ruta = os.path.join(path_input, archivo)
            dict_np_array = np.load(ruta)

            imagen = dict_np_array['image']
            encoded_image = numpy_to_base64(imagen)
            images.append(encoded_image)

    rows = []
    for i in range(0, len(images), 3):

        row = dbc.Row(
            [
                dbc.Col(create_card(img, id = {"type": "selected-card", "index": i}), width=4)
                for img in images[i:i+3]
            ],
            className="mb-4",
        )
        rows.append(row)
    return rows


def generate_plots_modal(image, inference):
    fig_rgb = px.imshow(image)
    fig_rgb.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )


    hover_text = np.vectorize(inference.get)(inference)

    # Create class map figure with hover
    fig_class = go.Figure(data=go.Heatmap(
        z=inference,
        text=hover_text,
        hoverinfo='text',
        colorscale='Viridis',
        colorbar=dict(title='Class')
    ))
    fig_class.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig_rgb, fig_class



layout = html.Div([
    html.H1('Resultados guardados'),
    html.P("Los resultados guardados hasta el momento han sido los siguientes:"),
    load_saved_inferences(),
    dbc.Modal(id = "modal-img", size="xl",is_open=False,backdrop="static" )
])




@dash.callback(
    Output("modal-img", "is_open"),
    Output("modal-img", "children"),
    Input({"type": "selected-card", "index": dash.ALL}, "n_clicks")
)
def on_card_click(n_clicks_list):
    triggered_id = ctx.triggered_id
    if triggered_id is None:
        return False, dash.no_update
    

    files =  os.listdir(path_input)
    target_file = files[triggered_id]
    ruta = os.path.join(path_input, target_file)

    dict_np_array = np.load(ruta)
    imagen = dict_np_array['image']
    inference = dict_np_array['inference']

    plot1, plot2 = generate_plots_modal(imagen, inference)

    results =  [
    dbc.ModalHeader(
        dbc.ModalTitle("🧠 Resultados de la inferencia en {}".format(target_file), className="w-100 text-center"),
        close_button=True
    ),
    dbc.ModalBody([
        html.H4("Visualizaciones de Inferencia", className="text-center mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=plot1), width=6),
            dbc.Col(dcc.Graph(figure=plot2), width=6),
        ])
    ])
]

    return results

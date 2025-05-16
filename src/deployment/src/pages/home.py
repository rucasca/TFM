import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1('This is our Archive page'),
    html.Div('This is our Archive page content.'),
])

# <button><i class="fas fa-upload"></i> Upload</button> fa-cloud-upload-alt
# <button><i class="fas fa-image"></i> Picture</button>
# <button><i class="fas fa-floppy-disk"></i> Save</button> fas fa-floppy-disk
# <button><i class="fas fa-clock-rotate-left"></i> History</button> fa-history
# <button><i class="fas fa-plus"></i> Add</button> fa-circle-plus
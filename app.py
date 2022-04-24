import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow as tf

import keras

import gan_conv

orc_model_path = './modeles/test_orcs2'
pokemon_model_path = './modeles/test_pokemon5'
manga_model_path = './modeles/test_manga2'

img = np.random.randint(255, size=(120, 120), dtype=np.uint8)
img = cv2.resize(img, (250,250))
noise_str = array_to_data_url(img)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(children="Application web de génération d'images", style={"text-align": "center", "font-size":"350%","background-color": "#757473", "color":"#E7E6E3", "height":"75px"}),

        html.Br(), #Retour à la ligne

        html.Div([
        html.P(
            children="Choisissez le type d'image à générer",
            style={"text-align": "center", "font-size":"150%", "color":"#E7E6E3"}
        ),
        dcc.Dropdown(options = [{'label': 'orc', 'value' : 'orc'},{'label' : 'pokemon', 'value' : 'pokemon'}, {'label' : 'manga', 'value' : 'manga'}], value = 'pokemon', id='model', style={"border-radius":"5px", "background-color": "#DAD9D6", "border": "None"}),
        ],
        style={"text-align":"center"}),

        html.Br(),

        html.Div([
        html.Button('Générer image', id='button', n_clicks=0, style={'marginRight':'40px',"height":"40px", "border-radius":"5px", "width":"30%", "background-color": "#DAD9D6", "border": "None"})], style={"text-align":"center"}),

        html.Br(),

        html.Div([
            html.Img(id='img', src='src', style={"center": "center"})
            ],style={"text-align": "center"}),

    ], style={"background-color": "#ABA9A6", 'height' : '700px'}
)

@app.callback(
     Output('img', 'src'),
    [Input('button', 'n_clicks'),
     Input('model', 'value'),
    ]
)

def affiche_img(n_clicks, model):
    if not n_clicks:
        return dash.no_update

    if model == 'orc' :
        return show_image(orc_model_path)

    elif model == 'pokemon' :
        return show_image(pokemon_model_path)

    elif model == 'manga':
        return show_image(manga_model_path)


def show_image(model_path):
    img = gan_conv.generate_image(model_path)
    x = img * 255
    x = cv2.resize(x, (250, 250))
    x = x.astype(np.uint8)
    return array_to_data_url(x)


if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=False)
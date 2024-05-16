import dash
from dash import dcc, html, Dash, Input, Output
from PIL import Image
import io
import numpy as np
import keras
import tensorflow as tf
from constants import MODEL_DIR, APP_IMAGES
from medical_assistant import chatbot
from dash.exceptions import PreventUpdate

from dotenv import load_dotenv # used to load env variables 
load_dotenv()

# loaded trained model
alexnet_model = keras.saving.load_model(MODEL_DIR+"DenseNet121_model.keras")
# instantiate chatbot
chat = chatbot()

# Define CSS styles
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

def apply_thresh(data): 
    threshold = 0.36770377
    return (data > threshold).astype("int32")

def get_prediction(filename):
        # Read the uploaded image into PIL Image object
        img_pil = Image.open(APP_IMAGES+filename[0])

        # Convert PIL Image to numpy array
        img_np = np.array(img_pil)

        # Convert numpy array to TensorFlow tensor with dims (1, 300, 300, 3)
        img_tensor = tf.convert_to_tensor(img_np)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        prediction = alexnet_model.predict(img_tensor)
        label = apply_thresh(prediction[0][0])
        if label == 1: 
            return prediction[0][0], label, "Malignant"
        else:
            return 1. - prediction[0][0], label, "Benign"


def parse_contents(contents):
    return html.Div([
        html.Img(src=contents, style={'margin': '5px', 'width': '250px', 'height': '250px'}),
    ])

# Define upload button
upload_button = dcc.Upload(
    html.Button('Upload File', style={'color': 'white', 'border': '2px solid white'}),
    id='upload-image',
    multiple=True
)

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [parse_contents(c) for c in list_of_contents]
        return children
    
@app.callback(Output('model_predictions', 'children'),
              [Input('upload-image', 'filename')])   
def display_prediction(filename):
    
    if filename is not None:
        # i.e. (0.97, 1, "Malignant")
        prob, prediction, label = get_prediction(filename)

        # Placeholder for prediction logic
        return html.Div([
            #html.H3('Model Predictions'),
            html.P('Image is classified as {0} with a probability of {1:.3}%'.format(label, prob*100), style={'font-size': '20px'})
        ])

    return html.Div()


@app.callback(Output("display_chat", "children"),
            [Input("input", "value")],
            prevent_initial_call=True)
def medical_chat(value):
    # Check if the input is empty
    print("Value:", value) 
    return chat.get_retrieval_augmented_answer(value.strip())


top_left =  html.Div(style={'flex': '1', 
                'border': '5px solid teal',
                'margin-left': '10px',
                'margin-right': '2px', 
                'margin-bottom': '2px',
                'backgroundColor': 'black'},
                children=[
                    html.H3('Upload Your Image', style={'text-align': 'center', 'color': 'white'}),
                    # top left quadrant
                    html.Div([
                        # Upload button
                        upload_button,
                        # Uploaded image
                        html.Div(id='output-image-upload'),
                            ]),     
                ])

top_right =     html.Div(style={'flex': '1',
                'border': '5px solid teal', 
                'margin-right': '10px',
                'margin-bottom': '2px',
                'margin-left': '2px', 
                'backgroundColor': 'black'}, 
                children=[
                    html.H3('Model Prediction', style={'text-align': 'center', 'color': 'white'}),
                    #html.P('Content of Top Right Quadrant'),
                    html.Div(id='model_predictions')
                ])

bottom_left =   html.Div(style={'flex': '1', 
                                'border': '5px solid orange', 
                                'margin-left': '10px', 
                                'margin-bottom': '15px',
                                'margin-right': '2px', 
                                'margin-top': '2px',
                                'backgroundColor': 'black', 
                                'display': 'flex',
                                'flex-direction': 'column' }, 
                            children=[
                                html.H3('Benign and Malignant Examples', style={'text-align': 'center', 'color': 'white'}),
                                html.P('Benign'), 
                                
                                    # Top row of images
                            html.Div([
                                html.Img(src=dash.get_asset_url("melanoma_59_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_60_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_61_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_718_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_719_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_720_b.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                            ], style={'display': 'flex', 'justify-content': 'center'}),  # Center images horizontally

                            html.P('Malignant'),
                            # Bottom row of images
                            html.Div([

                                html.Img(src=dash.get_asset_url("melanoma_5019_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_5020_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_5021_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_6923_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_6924_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                                html.Img(src=dash.get_asset_url("melanoma_6925_m.jpg"), style={'margin': '5px', 'width': '100px', 'height': '100px'}),
                            ], style={'display': 'flex', 'justify-content': 'center'}),  # Center images horizontally                           
                                
                                
                                
                                    ])


bottom_right =  html.Div(
                    style={'flex': '1', 
                                'border': '5px solid orange',
                                'margin-right': '10px', 
                                'margin-bottom': '15px', 
                                'margin-left': '2px', 
                                'margin-top': '2px',
                                'backgroundColor': 'black',
                                #'display': 'flex', 
                                'height': 'calc(50vh - 20px)', 
                                'width': 'calc(50vw - 20px)' 
                    }, 
                 
                 children=[
                    html.H3('AI Assistant (Powered by ChatGPT)', style={'text-align': 'center', 'color': 'white'}),
                    html.P('Ask any questions you have about melanoma.'), 
                    dcc.Input(
                        id="input",
                        type='text',
                        placeholder="Input Text",
                        value='', 
                        debounce=True,
                        style={'width': '100%', 'height': 50, 'background-color': 'black', 'color': 'white'}
                        ), 
                    html.Div(id="display_chat", style={'overflowY': 'scroll', 'height': 200})
                ])

# Define app layout
app.layout = html.Div(style={'backgroundColor': 'black', 
                             'padding': '10px', 
                             'width': '100vw', 
                             'height': '100vh', 
                             'color': 'white', 
                             'overflow': 'hidden'},
                      
                      children=[
                          
                            html.H1('Medical AI: Skin Care Evolved', 
                                    style={'text-align': 'center', 
                                           'color': 'white', 
                                           'backgroundColor': 'black'}
                                    ),
                                                
            
                            html.Div(style={'display': 'flex', 
                                            'height': 'calc(50vh - 20px)'}, 
                                    children=[
                                
                                # top left quadrant
                                top_left,
                                # top right quadrant
                                top_right
                                ]),
    
                            html.Div(style={'display': 'flex',
                                            'height': 'calc(50vh - 20px)'},
                                            # 'width': 'calc(50vw - 20px)'},
                                            
                                    children=[
                                        # bottom left quadrant
                                        bottom_left, 
                                        # bottom right quadrant
                                        bottom_right
                                        ])
                                ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8050, use_reloader=False)
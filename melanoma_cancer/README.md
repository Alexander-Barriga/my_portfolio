# Melanoma Detection and ChatGPT Assistant

This project is a web application built using Dash, a trained CNN model for classifying skin marks as benign or malignant Melanoma, and a custom ChatGPT medical assistant to answer melanoma-related questions. 

You can read the Medium article I wrote explaining the app [**here**](https://towardsdatascience.com/create-an-a-i-driven-product-with-computer-vision-and-chatgpt-070a34ab9877?sk=41cb5af971a780c5a366d4b4308761b3).


## Demo 


https://github.com/DataBeast03/Portfolio/assets/10015949/efcaad0a-7662-4c5b-ba53-e2faf4ed5ada



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection.git
   cd melanoma-detection


## Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv cancer_venv
source cancer_venv/bin/activate  # For Linux/Mac
.\cancer_venv\Scripts\activate    # For Windows
```

## Install the required libraries:

```bash
pip install -r melanoma_cancer/requirements.txt
```

## Download Data

#### Download Data Directory to Laptop
The images used in this application are taken from the Melanoma Dataset provided on Kaggle. You can download the dataset [**here**](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data).


#### Download Data to Google Colab notebook
IF you desire to re-train the models on Google Colab (not necessary in order to run app), you will need to follow a different set of steps. You'll need to run the following code in notebook cells as is: 

1. You will need to create a Kaggle account (if you don't already have one).

2. You need to create a Kaggle access token to download data using the Kaggle API (See the [**Authentication**](https://www.kaggle.com/docs/api) section for instructions). Keep the access token file on your local machine. 

3. Launch a google colab notebook [**here**](https://colab.research.google.com/).

4. Run this code in a notebook cell. You will be prompted to upload the Kaggle access token file. Do so. 
```bash
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
```

5. Run this in a notebook cell in order to download and unzip the data directory. 
```bash
!kaggle datasets download -d hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
!unzip melanoma-skin-cancer-dataset-of-10000-images.zip
```

## Run App

### 1. Train the models (NOT Necessary, only if desired):

Because Github does not allow uploading files larger than 100 MB I am not able to upload all 4 trained models used in this project. 
However, I am able to upload the DenseNet model which fortunately is the best performing model and the one used in the app. 

DenseNet model has already been trained but you can retain it and others (only if desired).
    - Use notebook Train_Models_Colab.ipynb to train the models using the data in the data/train directory.
    - It is recommended to use GPU hardware to significantly speed up training time


### 2. Build the Docker image:

```bash
docker build -t melanoma-app .
```

### 3. Run the Docker container:

```bash
docker run -p 8050:8050 melanoma-app
```

### 4. Access the web app:

Open your browser and go to http://localhost:8050 to use the Melanoma Detection app.


## Project Files and Directories 
       Melanoma-Questions-and-Answers-Booklet.pdf: Melanoma-related questions and answers booklet.
       data/: Directory containing test and train data.
       test/: Test data directory.
       train/: Training data directory.
       melanoma_cancer/: Python package for the web app.
       Dockerfile: Docker configuration file.
       __pycache__/: Cached Python files.
       app.py: Dash web app code.
       app_images/: Uploaded images storage directory.
       assets/: Static assets for the web app.
       cancer_venv/: Virtual environment directory.
       constants.py: Constants used in the app.
       medical_assistant.py: ChatGPT assistant for melanoma-related questions.
       requirements.txt: List of required Python libraries.
       trained_models_cancer/: Trained models directory.
       notebooks/: Jupyter notebooks for model training and evaluation.
       Model_Evaluation.ipynb: Notebook for evaluating the trained models.
       Train_Models.ipynb: Notebook for training the models.
       Train_Models_Colab.ipynb: Notebook for training the models in Google Colab.


## License
This project is licensed under the MIT License

Copyright 2024 Alexander Barriga

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Welcome

Here you'll find some personal data science projects that I've worked on. 

### Note
I've recently had to migrate a subset of my personal projects to a completely new GitHub account. My previous, and long-standing, GitHub account (DataBeast03) got blocked by GitHub (other than myself no one else could access it). I don't know why. I've reported this to GitHub but I have yet to hear back. 


# Projects
------
# EV Energy Cost by Performance Tier Analysis 

### Cluster EVs by performance metrics, create an energy cost model, and compare monthly energy costs between brands.

![Image](./EV_project/plots/Cost_Model/Monthly%20Energy%20Cost%20Distribution%20by%20EV%20Tier.png)

#### TL;DR
- Understand the distributions of EV performance metrics and driver behavior at EV charging stations
- Run statistical hypothesis testing on EV driver charging behavior
- Create correlation model between EV performance metrics and price
- Cluster EVs based on performance metrics and create two tiers
- Create EV energy price model and compare results between EV tiers and brands


-----

# Melanoma Detection and ChatGPT Assistant

This project is a web application built using Dash, a trained CNN model for classifying skin marks as benign or malignant Melanoma, and a custom ChatGPT medical assistant to answer melanoma-related questions. 

You can read the Medium article I wrote explaining the app [**here**](https://towardsdatascience.com/create-an-a-i-driven-product-with-computer-vision-and-chatgpt-070a34ab9877?sk=41cb5af971a780c5a366d4b4308761b3).

## Demo 


https://github.com/DataBeast03/Portfolio/assets/10015949/efcaad0a-7662-4c5b-ba53-e2faf4ed5ada


# Machine Learning Pipeline for Bank Fraud 

![](https://github.com/Alexander-Barriga/my_portfolio/blob/main/Fraud_Detection/atm_fraud.png)

## Summary 

This software is designed to load, clean, and perform a series data transformations in preperation for the ML pipeline that trains a series of models for detection fradulent banking activity. 

## Data Preperation 

The `prep_data` class performs the following data transformations:
- Drops low variance features
- Distinguishes between categorical and numerical features
- Identifies and removes any outliers
- Balances imbalanced labels 

Check out the `EDA` notebook for a detailed overview.


## ML Pipeline 

The `train_models` class performs the following: 
- create_transform_portion_of_pipeline
- Creates a data transform pipeline that handels categorical and numerical features differently
- Performs a grid search for each pipeline object
- Trains a list of default ML models but allows the user to pass in a custom list 
- Logs the performance of each pipeline
- Saves best pipeline to file

Check out the `Modeling` notebook for a detailed overview.

## Set up
1) Download and install [Anaconda](https://www.anaconda.com/)
2) Run `pip install -r requirements.txt` 
3) In the terminal, navigate over to the `Scripts` directory
4) Run `python -m run` 

Click [**here**](https://github.com/DataBeast03/Portfolio/tree/master/Fraud_Detection) to be directed over to the project's code. 

-----

## Dashboard: Scalable Machine Learning Pipeline 

![](https://github.com/Alexander-Barriga/my_portfolio/blob/main/Dash_MLTool/pipeline_img.png)

#### This is a complete pipeline: 
1. Runs Classification and Regression data through an ETL pipeline
2. Parallelizable model building tool trains, scores agaisnt several metrics, and stores results
3. Model's metrics socres are displayed on learning curves. 
4. The interactive dashboard allows the user to select among several different models for both Classiciation and Regression. 

-----

#### Scalability and Performance Boost

![](https://github.com/Alexander-Barriga/my_portfolio/blob/main/Dash_MLTool/runtime_reg.png)

This pipeline was tested on a 8 core laptop. The chart shows that speed increases are achieved as the number of cores increases. 
The limiting factor for the performance boost being, of course, the run time of a single model's trian time and the number of cores. 

----

#### Technology

- The front end was built using a Python library called [**Dash**](https://plot.ly/products/dash/)
- The Scalable model building tool was built by me and can be found [**here**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/cross_val_tool_with_PARALLEL.py)
- The machine learning models are taken from [**Scikit-Learn**](http://scikit-learn.org/stable/)
- The pipeline is being deployed on [**AWS EC2**](https://aws.amazon.com/ec2/) 

------

Check out the [**Live Dashboard Here**](http://54.215.234.117/)

Check out the [**Dash Script**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/ml_pipeline.py)


------

## Analytical Dashboard 

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/dashboard_screenshot.png)

This is a prototype analytical dashboard for solar energy consumers. 

This is our scenario: imagine that one of Google's locations (there are many in the USA) has 4 buildings, each with solar panel installations. They want to keep track of 3 very importannt trends: 

1. Energy Consumption by each building
2. Energy Production by each building
3. Energy cost/profit by each building

The numbers will be tracked monthly. The cost is the energy bill for each building, so that means that the building has consumed more energy than its solar panels produced. The profit is the money made by selling excess energy back to the energy grid. In the end, we will have one years worth of data for each building. 

Check out the [**LIVE DASHBOARD HERE**](http://54.153.32.166/)

Check out the [**DASH SCRIPT**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/kwh_analytics.py)

Check out the [**JUPYTER NOTEBOOK**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/Dashboard.ipynb) where the models were built. 


------

## Content Based Recommender for New York Times Articles 

![](https://blog.gdeltproject.org/wp-content/uploads/2014-new-york-times-logo.png)

In this notebook, I create a content based recommender for New York Times articles. This recommender is an example of a very simple data product. I follow the same proceedure outlined in this [Medium article](https://medium.com/data-lab/how-we-used-data-to-suggest-tags-for-your-story-a120076d0bb6#.4vu7uby9z).

![](https://cdn-images-1.medium.com/max/1600/1*3BP9i12zmh99F4fyjUdi3w.png)


However, we will not be recommending tags. Instead we'll be recommending new articles that a user should read based on the article that they are currently reading.

Check out the [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/NYT_Recommender/Content_Based_Recommendations.ipynb)


-----


## Machine Learning Tool

The focus of this tool is to make the machine learning model building and validation workflow very fast and easy. 

This is done by abstracting away all the cross validation and plotting functionality with a reusable class. 

This class also allows us to train and score these models in parallel. 

It also has built in learning curve plotting functionality to assess model performance.   

As a case study, we use a Cellular Service Provider data set where we are tasked with building a model that can identify users 
who are predicted to churn. Naturally in subscription based services, these data sets are unbalanced since most users 
don't cancel their subscription on any given month. 

Let's see how this tool can help us achieve our goal!

Check out the [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/ML_Tool/ML_Tool.ipynb)

```python

# create model
rfc = RandomForestClassifier(n_estimators=100, 
                             criterion='entropy', 
                             n_jobs=-1)
# initialize ml tool 
cv_rfc = cross_validation(rfc, 
                      X_churn, 
                      Y_churn, 
                      average='binary',
                      init_chunk_size=100, 
                      chunk_spacings=100,
                      n_splits=3)

# call method for model training
cv_rfc.train_for_learning_curve()

# call method for ploting model results
cv_rfc.plot_learning_curve(image_name="Learning_Curve_Plot_RF", save_image=True)

```


![](https://github.com/DataBeast03/DataBeast/blob/master/ML_Tool/Learning_Curve_Plot_RF.png)


----
## Classify Physical Activities with CNN Deep Learning Models 
<img src="https://github.com/Alexander-Barriga/my_portfolio/blob/main/DeepLearning/CNN_Activity_Classification/sport_watch_logos.png" width="400"><img src="http://adventuresinmachinelearning.com/wp-content/uploads/2017/04/CNN-example-block-diagram-1024x340.jpg" width="400">



Based on the research of the authors of this [whitepaper](https://arxiv.org/pdf/1610.07031.pdf), I trained a Convolutional Neural Network to classify the physical activities of users wearing wrist devices that contain sensors like an accelerometer and gyroscope. In order words, the CNN was trained on time-series data and not images and performed quite well. 



-----



## Entity Search Engine
<img src="http://www.listchallenges.com/f/lists/d7aacdae-74bd-42ff-b397-b73905b5867b.jpg" width="400">

I engineered a data product that allows the user to search for unassuming relationships bewteen entities in New York Times articles. The articles were scraped from the NYT api. I used Sklearn's implementation of Latent Dirichlet Allocation for Topic Modeling and the NLTK library for Entity Recognition. This data product is an excellent example of how Machine Learning and Natural Language Processing can be used to build an application to serve the needs of an end user. 

I wrote three object oriented classes for this project:

**topic_model_distributions** 
has methods to fit Latent Dirichlet Allocation (LDA) for topic modeling and methods to get certain distributions that are necessary to visualize the LDA results using the pyLDAvis data viz tool

**named_entity_extraction**
has methods to identify and extract named entities, the like that we observed in the police shooting article. It also has methods that count the number of entity appearances in each topic and the number of entity apperances in each article.

**entity_topic_occurances**
has methods to identify co-occurances of entities within the same topic and within the same document. 


------


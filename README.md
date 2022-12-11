# Diasaster_Response_Project
This is Team project. We have created ETL Pipeline to clean the data transform into database.  ML pipeline helps in transforming the data using Scikit learn we can implemetent the SVM classifer and train the data. 
# Disaster Response Pipelines

## Table of Content:
* [Project Overview](#project_overview)tl
* [Project Outline](#project_outline)
  * [Extract, Transform, and Load Pipeline](#ETL_pipline)
  * [Machine Learning Pipleline](#machine_learning_pipeline)
  * [Flask Web app](#flask_app)
* [Training Dataset](#dataset)
* [Machine Learning Model](#model)
* [Files Structure](#files)
* [Requirments](#requirments)
* [Running Process](#running)
  * [Process Data](#process_data)
  * [Train Classifier](#train_classifier)
  * [Run the flask web app](#run_flask_app)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

***
<a id='project_overview'></a>
## 1. Project Overview
The Disaster Watch Project helps disaster victims by minimizing potential losses and providing the right assistance. Governments, businesses, and civil society work continuously to prepare for and lessen the effects of disasters. At every stage of a crisis, taking the right action increases preparedness, improves warnings, and reduces susceptibility.

One of the best ways to receive a quick overview of what's happening globally is through social media tools, but it may be challenging to sift through everything online. The goal of this research is to assist governments in categorizing millions of social media messages using supervised machine learning. In order for the governments to promptly respond to disasters, the model is trained using the [Figure Eight](https://appen.com/) dataset to classify the messages into the appropriate categories.

<a id='project_outline'></a>
## 2. Project Outline
This section explains all three parts of this project from cleaning the data to deploying the model on the flask app

<a id='ETL_pipline'></a>
### 2.1 Extract, Transform, and Load Pipeline 
The Extract, Transform and Load (ETL) pipeline is responsible for preparing the dataset for the machine learning pipeline and it works as following:
* Extract the messages and their categories from the CSV files
* Clean and merge the messages and categories in one data frame
* Saves the data frame inside an SQLite database

<a id='machine_learning_pipeline'></a>
### 2.2 Machine Learning Pipleline 
A machine learning pipeline is the end-to-end construct that orchestrates the flow of data into, and output from, a machine learning model (or set of multiple models). It includes raw data input, features, outputs, the machine learning model and model parameters, and prediction outputs.
<a id='flask_app'></a>
###2.3 Flask Web App
Flask Web App is responsible for deploying the machine learning model on a website and allowing the user to use the trained model to do predictions.

![image]

<a id='dataset'></a>
## 3. Training Dataset
The cleaned training dataset contains more than 26K labeled messages and has 36 different classes such as related, offer, food, water, and electricity. The following photo shows how many classes the dataset has: ![image](

<a id='model'></a>
## 4. The machine learning model
The macinhe learning model was built using [SVClinear](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) from [scikit-learn](https://scikit-learn.org/) library. The model accruacy was calculated using [Numpy mean](https://github.com/Murtada-Altarouti/Disaster-Response-Pipelines/blob/1bd315ee829e9cda890a88b25fcce356198a1aa5/models/train_classifier.py#L101) which was equal to 95%. 

<a id='files'></a>
## 5. Files Structure
```
├── app #Website folder
│   ├── run.py #Responsible of running the website
│   └── templates
│       ├── go.html #Responsible of showing the results
│       └── master.html #The main page
|
├── data
│   ├── disaster_categories.csv #Categories dataset
│   ├── disaster_messages.csv #Messages dataset
│   ├── DisasterResponse.db #The cleaned dataset in SQLite database
│   └── process_data.py #Responsible for preparing the dataset 
|
├── models
│   ├── classifier.pkl #The SVM model
│   └── train_classifier.py #Responsible for creating the machine learning model
|
├── readme_images #This folder contains all images for the readme file
│   ├── dataset.png
│   └── website_example.png
└── README.md #Readme file 
```

<a id='requirments'></a>
## 6. Requirments
In order to run this project, you must have [Python3](https://www.python.org/) installed on your machine. You also must have all listed libraries inside the `requirments.txt` so run the following command to install them: 
```
pip3 install -r requirments.txt


<a id='conclusion'></a>
## 8. Conclusion
In the conclusion, catastrophes are horrible if we are not properly prepared to deal with them, thus having a system that consistently delivers correct warnings is helpful to get an early notice of a potential disaster and decrease potential losses. The system was built using scikit-learn and achieved 95% accuracy, but that does not mean it is the best model.

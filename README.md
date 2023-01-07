# Disaster_Response

This project allows us to read a tweet and label it as any of the below categories facilitating the identification of high priority messages which need immediate assistance

Just download the project folder and run it in the local conda environment. 
I have utilised the below packages and libraries

1. sys
2. pandas
3. sqlite3
4. re 
5. numpy
6. nltk
7. pickle
8. sklearn.feature_extraction.text TfidfVectorizer
9. sklearn.base BaseEstimator
10. sklearn.pipeline Pipeline
11. sklearn.multioutput MultiOutputClassifier
12. sklearn.model_selection train_test_split
13. sklearn.tree DecisionTreeClassifier
14. sklearn.metrics classification_report
15. warnings

**NOTE**: download nltk('punkt') into the runtime environment

Files in the repository

![image](https://user-images.githubusercontent.com/117662647/211130962-e9575423-375a-4546-8eaa-97a711afc6aa.png)

File Usage:

1. run.py - Launch application
2. process_data.py - process the data and insert into Database
3. train_classifier.py - train the model from data from db
4. disaster_messages.csv,disaster_categories.csv - input files to process_data.py
5. DisasterResponse.db - database name to which processed data is saved
6. master.html - render graphs on app home page
7. go.html - render model input and prediction on app

**RUN Application**

1. In order to run the application from terminal: 
cd app
python run.py

2. In order to run the data processing from terminal:
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

3. In order to train model from terminal:
cd model
python train_classifier.py ..\data\DisasterResponse.db classifier.pkl

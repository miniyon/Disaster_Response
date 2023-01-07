# Disaster_Response

This project allows us to read a tweet and label it as any of the below categories facilitating the identification of high priority messages which need immediate assistance

Just download the project folder and run it in the local conda environment. 
I have utilised the below packages and libraries

1. sys
2. pandas
3. sqlite3
4. re 
5. numpy
6. pickle
7. sklearn.feature_extraction.text TfidfVectorizer
8. sklearn.base BaseEstimator
9. sklearn.pipeline Pipeline
10. sklearn.multioutput MultiOutputClassifier
11. sklearn.model_selection train_test_split
12. sklearn.tree DecisionTreeClassifier
13. sklearn.metrics classification_report
14. warnings

Files in the repository

![image](https://user-images.githubusercontent.com/117662647/211130962-e9575423-375a-4546-8eaa-97a711afc6aa.png)

**RUN Application**

1. In order to run the application from terminal: 
cd app
python run.py

2. In order to run the data processing from terminal:
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db


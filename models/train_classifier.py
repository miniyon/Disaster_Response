import sys
import pandas as pd
import sqlite3
import re 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


class TextClean(BaseEstimator):

    def __init__(self):
        """constructor for TextClean class
        """        
        pass

    def fit(self, corpus, y=None):
        """
        Fit the train data to the model
        Args:
            corpus (Series): _description_
            y (Array, optional): _description_. Defaults to None.

        Returns:
            self: self object
        """        
        return self

    def transform(self, data):
        """
            Transform the data
        Args:
            data (Series): The predictor variable

        Returns:
            Series: Cleaned predictor 
        """ 
        data = pd.Series(data,name='message')       
        data = data.reset_index()
       
        # replace all characters other than text
        data['cleaned'] = data['message'].apply(lambda x : re.sub('\s+',' ',re.sub('[^a-z,A-Z]',' ',x)))
        
        # convert all words to lowercase
        data['cleaned'] = data['cleaned'].apply(lambda x : x.lower())
        return data['cleaned']

 
def load_data(database_filepath):
    """
    Loads the training data from the database and
    converts to training data , target data and labels

    Args:
        The database file path (string): Database file location

    Returns:
        Series,numpy array,list: Predictor Variable as a series,Target variable as numpy array and the list of labels 
    """    
    conn = sqlite3.connect(database_filepath)
    print(conn)
    try:
        df = pd.read_sql('''select * from disaster_response_tweets''',con=conn)
        X = df['message']
        y = df.drop(['id','message','genre'],axis=1)
        categories = y.columns.unique().to_list()
        y = y.astype(int)
        y = y.to_numpy()
        
        return X,y,categories
    except:
        print("Database could not be connected to")
        raise
    finally:
        conn.close()
    


def build_model():
    """
    Chains together the process of cleaning,
    vectorization and classification
    
    Returns:
        Pipeline object: the object allows for the cleaning, vectorization 
        and classification in sequence
    """    
    model_pipeline = Pipeline(steps=[('textclean', TextClean()),
                                 ('texttovector',TfidfVectorizer(stop_words = 'english')),
                                ('classifier',MultiOutputClassifier(DecisionTreeClassifier(max_depth=4,random_state=42)))])
    
    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model using classification report
    for each label
    Args:
        model (DecisionTreeClassifier Object): model that was trained for classification
        X_test (Series): The input message
        Y_test (Numpy Array): The actual value of the target
        category_names (list): All the labels associated with the target
    """    
    classification_report_dictionary ={}
    y_pred = model.predict(X_test) 
    for label_col in range(len(category_names)):
        try:
            y_true_label = Y_test[:, label_col]
            y_pred_label = y_pred[:, label_col]
            classification_report_dictionary[category_names[label_col]] = classification_report(y_pred=y_pred_label, y_true=y_true_label)
        except:
            continue
    
    for label, report in classification_report_dictionary.items():
        print("classification report for label {}:".format(label))
        print(report)
    


def save_model(model, model_filepath):
    """
    This function saves the trained pipeline as a pickle file

    Args:
        model (DecisionTreeClassifier Object): model that was trained for classification
        model_filepath (string): location with file name for the trained model
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
         
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
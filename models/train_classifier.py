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
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


def tokenize_lemmatize_text(x):
    tokens_list = []
    if (str(x)!='nan') or (str(x)!='None'):
        tokens = word_tokenize(x)
        if len(tokens)!=0:
            for token in tokens:
                tokens_list.append(WordNetLemmatizer().lemmatize(token))
        return " ".join(tokens_list)
    else:
        return 'empty'


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
        
        # tokenize & lemmatize text
        data['cleaned_x'] = data['message'].apply(lambda x : tokenize_lemmatize_text(x))
        
        # replace all characters other than text
        data['cleaned_y'] = data['cleaned_x'].apply(lambda x : re.sub('\s+',' ',re.sub('[^a-zA-Z]',' ',x)) if str(x)!='empty' else x)
        
        # convert all words to lowercase
        data['cleaned'] = data['cleaned_y'].apply(lambda x : x.lower() if ((str(x)!='empty') or (str(x) !='None')) else None)
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
    


def build_model(X_train,Y_train):
    """
    Chains together the process of cleaning,
    vectorization and classification
    
    Returns:
        Pipeline object: the object allows for the cleaning, vectorization 
        and classification in sequence
    """  

    parameters = {
        'criterion' : ['gini','entropy'],
        'max_depth' : [3,4,5,6],
        'min_samples_split':[2,3,4]
    }  
    gridcv = GridSearchCV(estimator=DecisionTreeClassifier(),
        param_grid=parameters,
        scoring='accuracy',
        cv=3)
    
    gridcv_pipeline = Pipeline(steps=[('textclean', TextClean()),
                                 ('texttovector',TfidfVectorizer(stop_words = 'english')),
                                ('gridcv',gridcv)])

    gridcv_pipeline.fit(X_train, Y_train)
    criterion = gridcv.best_params_['criterion']
    max_depth = gridcv.best_params_['max_depth']
    min_samples = gridcv.best_params_['min_samples_split']

    model_pipeline = Pipeline(steps=[('textclean', TextClean()),
                                ('texttovector',TfidfVectorizer(stop_words = 'english')),
                            ('classifier',MultiOutputClassifier(DecisionTreeClassifier(criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples,
                            random_state=42)))])

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
        X_grid=X_train.copy()
        Y_grid = Y_train.copy()
        
        print('Creating Pipeline ...')
        model = build_model(X_grid,Y_grid)
        
        print('Training Model ...')
        
        model.fit(X_train,Y_train)
         
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
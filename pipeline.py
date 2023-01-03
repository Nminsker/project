from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import datasets_utils
import base_models
from hpd import search
from ipdb import set_trace
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import math
import random


class Pipeline:


    def __init__(self):
        pass


    def apply(self, data, labels, model):

        """this function returns trained model after pipeline"""

        ## prepare data for training, cleaning, etc.
        clean_data = self.clean_data_pipeline(data)

        # transformed_data = self.SelectKBest_feature_selection_pipeline(clean_data, labels)
        # transformed_data = self.feature_selection_pipeline(clean_data, labels)

        ## train the model
        model = self.model_pipeline(model, clean_data, labels)

        return model


    def clean_data_pipeline(self, data):

        """clean and validate data"""

        print('starting data cleaning...')

        ## transform non numeric data to categorial numeric data:
        for col in data.dtypes[data.dtypes=='object'].index:
            data[col], _ = data[col].factorize() 
        print(f'data shape before removing Nons: {data.shape}')

        ## remove Nons
        data = data.dropna()
        print(f'data shape after removing Nons: {data.shape}')
        
        return data


    def feature_selection_pipeline(self, data, labels):
        return


    def SelectKBest_feature_selection_pipeline(self, data, labels):

        """perform feature selection on the data"""

        # feature extraction
        k = math.floor(0.8*len(data.columns))
        print(f"Number of selected features : {k}/ {len(data.columns)}")
        test = SelectKBest(score_func=f_regression, k=k)
        fit = test.fit(data, labels)
        set_printoptions(precision=3)
        # print(fit.scores_)
        return fit.transform(data)


    def model_pipeline(self, model, clean_data, labels):
        
        """train the model, and use pipeline methods"""

        train_data, test_data, train_labels, test_labels = \
                train_test_split(clean_data, labels, test_size = 0.3)

        ## train the model
        model.fit(train_data, train_labels)

        # create some predictions on train data (just for sanity)
        # preds_on_train_data = model.predict(train_data)
        # print_err(train_labels, preds_on_train_data, "Train")

        ## create some predictions on test data
        preds = model.predict(test_data)

        ## print results
        print_err(test_labels, preds, "Test")

        ## run HPD on data to find data slices that our model perform badly on
        test_data.reset_index(inplace=True)
        test_data.drop('index', axis=1, inplace=True)

        test_labels = test_labels.reset_index()
        test_labels.drop('index', axis=1, inplace=True)

        data_slices = search(test_data, test_labels, model)

        ## wrap the model with interval model
        #TODO model = 'replace this with something real'

        return model


def print_err(labels, pred, type):

    """Print predictions error"""

    # print('MAE:', metrics.mean_absolute_error(labels, pred))
    print(f'{type} MSE:{ metrics.mean_squared_error(labels, pred)}')
    # print('RMSE:', np.sqrt(metrics.mean_squared_error(labels, pred)))


def baseline(data, labels, model):

    """train the baseline model, and print results"""

    ## transform non numeric data to categorial numeric data:
    for col in data.dtypes[data.dtypes == 'object'].index:
        data[col], _ = data[col].factorize()

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.3)

    model.fit(train_data, train_labels)
    preds = model.predict(test_data)
    print_err(test_labels, preds, "Baseline")


def set_seed():
    seed_value = 678
    random.seed(seed_value)
    np.random.seed(seed_value)

if __name__ == "__main__":

    # set_seed()
    pipeline = Pipeline()

    print("\n*******************************************************")
    print("******************* French Motor Data *****************")
    print("*******************************************************\n")

    data_motor, labels_motor = datasets_utils.loadFrenchMotorData()
    baseline(data_motor, labels_motor, base_models.frenchBaseModel())
    trained_model_motor = pipeline.apply(data_motor, 
                                         labels_motor, 
                                         base_models.frenchBaseModel())

    print("\n*******************************************************")
    print("******************* Boston Data ***********************")
    print("*******************************************************\n")

    data_boston, labels_boston = datasets_utils.loadBostonData()
    baseline(data_boston, labels_boston, base_models.bostonBaseModel())
    trained_model_boston = pipeline.apply(data_boston, 
                                          labels_boston, 
                                          base_models.bostonBaseModel())





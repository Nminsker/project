import base_models
import datasets_utils

from hpd import search
from macest_utils import MacestModel
import numpy as np

import pandas as pd
import random
from sklearn import metrics
import sklearn.base as sklearn_base
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split


class Pipeline:
    """ Fuctionality for building and running the pipeine """

    def apply(self, data, labels, model):
        """this function returns trained model after pipeline"""

        ## prepare data for training, cleaning, etc.
        self.clean_data_pipeline(data)

        ## Split data into train and test
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=6)

        ## Perform feature selection
        ## This function returns train and test data data frames
        ## just with the 'good features' after the feature selection methods
        selected_features = self.feature_selection_pipeline(model, train_data, train_labels)

        transformed_train_data = train_data[selected_features]
        transformed_test_data = test_data[selected_features]

        ## Model pipeline
        model, data_slices = self.model_pipeline(model, transformed_train_data, train_labels, transformed_test_data, test_labels)

        self.predict_and_eval(model, test_data, test_labels)

        return model

    def clean_data_pipeline(self, data):
        """clean and validate data - work is done in place!"""

        print('starting data cleaning...')

        ## transform non numeric data to categorial numeric data:
        for col in data.dtypes[data.dtypes=='object'].index:
            data[col], _ = data[col].factorize() 
        print(f'data shape before removing Nons: {data.shape}')

        ## remove Nons
        data.dropna(inplace=True)
        print(f'data shape after removing Nons: {data.shape}\n')

    def SelectKBest_feature_selection(self, model, train_data, train_labels, test_data, test_labels, num_features):
        """ Preform feature selection on the data using SelectKBest with f_regression scoring func and k = num_features"""

        test = SelectKBest(score_func=f_regression, k=num_features)

        fit = test.fit(train_data, train_labels)

        model.fit(fit.transform(train_data), train_labels)
        pred = model.predict(fit.transform(test_data))
        # print(fit.scores_)

        err = metrics.mean_squared_error(pred, test_labels)
        print(f"Number of features :" 
              f"{num_features}/ {train_data.shape[1]} MSE : {err}")

        # MACest makes us use sklearn 0.22.1 which doesn't have the function get_feature_names_out
        # To work around that just do as the source code of 1.0.* does
        featureNames = np.array(train_data.columns)[test.get_support()]

        return featureNames, err

    def feature_selection_pipeline(self, model, data, labels):
        """ Preform feature selection on the data and return the transformed data"""

        print("***Running feature selection pipeline***\n")

        # Find best number of features
        res = []
        start = int(len(data.columns)/2)
        stop = len(data.columns) + 1

        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=6)

        for k in range(start, stop):
            selected_features, err = self.SelectKBest_feature_selection(sklearn_base.clone(model), train_data,
                                                                        train_labels, val_data, val_labels, k)

            res.append({'selected_features': selected_features, 'err': err})

        best = min(res, key=lambda x:x['err'])

        print("\n***End of feature selection phase***\n")
        print(f"Num of selected features ===> {len(best['selected_features'])} \n"
              f"Selected features : {best['selected_features']} "
              f"Validation MSE :{best['err']}")

        return best['selected_features']


    def model_pipeline(self, model, train_data, train_labels, test_data, test_labels):     
        """train the model, and use pipeline methods."""

        ## train the model
        model.fit(train_data, train_labels)

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
        model = MacestModel(model, train_data, train_labels)  

        return model, data_slices


    def predict_and_eval(self, model, test_data, test_labels):
        preds = model.predict(test_data)
        print_err(test_labels, preds, "Test")


def print_err(labels, pred, type):

    """Print predictions error"""

    # print('MAE:', metrics.mean_absolute_error(labels, pred))
    print(f'{type} MSE:{ metrics.mean_squared_error(labels, pred)}\n')
    # print('RMSE:', np.sqrt(metrics.mean_squared_error(labels, pred)))


def baseline(data, labels, model):
    """train the baseline model, and print results"""

    ## transform non numeric data to categorial numeric data:
    for col in data.dtypes[data.dtypes == 'object'].index:
        data[col], _ = data[col].factorize()

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=6)

    model.fit(train_data, train_labels)
    preds = model.predict(test_data)
    print_err(test_labels, preds, "Baseline model")


def set_seed():
    seed_value = 678
    random.seed(seed_value)
    np.random.seed(seed_value)


if __name__ == "__main__":
    set_seed()
    pipeline = Pipeline()

    print("\n*******************************************************")
    print("******************* French Motor Data *****************")
    print("*******************************************************\n")

    data_motor, labels_motor = datasets_utils.loadFrenchMotorData()
    baseline(data_motor.copy(), labels_motor.copy(), base_models.frenchBaseModel())
    trained_model_motor = pipeline.apply(data_motor.copy(), labels_motor.copy(), base_models.frenchBaseModel())

    print("\n*******************************************************")
    print("******************* Boston Data ***********************")
    print("*******************************************************\n")

    data_boston, labels_boston = datasets_utils.loadBostonData()
    baseline(data_boston.copy(), labels_boston.copy(), base_models.bostonBaseModel())
    trained_model_boston = pipeline.apply(data_boston.copy(), labels_boston.copy(), base_models.bostonBaseModel())





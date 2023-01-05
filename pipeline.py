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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn
import pandas as pd

class Pipeline:

    def __init__(self):
        pass

    def apply(self, data, labels, model):

        """this function returns trained model after pipeline"""

        ## prepare data for training, cleaning, etc.
        clean_data = self.clean_data_pipeline(data)

        train_data, test_data, train_labels, test_labels = train_test_split(clean_data, labels, test_size = 0.3, random_state=6)

        ## Perform feature selection
        t_train_data, t_test_data = self.feature_selection_pipeline(model, train_data, train_labels, test_data, test_labels)

        ## train & eval
        model = self.model_train_pipeline(model, t_train_data, train_labels)
        self.predict_and_eval(model, t_test_data, test_labels)

        return model

    def clean_data_pipeline(self, data):

        """clean and validate data"""

        ## transform non numeric data to categorial numeric data:
        for col in data.dtypes[data.dtypes=='object'].index:
            data[col], _ = data[col].factorize() 
        
        ## remove Nons
        # TODO

        return data

    def RFECV_feature_selection_pipeline(self, model, data, labels):
        min_features_to_select = 6  # Minimum number of features to consider

        rfecv = RFECV(
            estimator=RandomForestRegressor(),
            step=1,
            scoring="neg_mean_squared_error",
            min_features_to_select=min_features_to_select,
            n_jobs=2,
        )

        select = rfecv.fit(data, labels)
        mask = select.get_support()
        features = np.array(data.columns)
        best_features = features[mask]
        #plot_feature_selection_results(rfecv, min_features_to_select)

        print(f"Ranking :{rfecv.ranking_}")
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Best features: {best_features}")

        return data.filter(best_features)

    def SelectKBest_feature_selection(self, model, train_data, train_labels, test_data, test_labels, num_features):

        """perform feature selection on the data using SelectKBest with f_regression scoring func and k = num_features"""

        test = SelectKBest(score_func=f_regression, k=num_features)
        fit = test.fit(train_data, train_labels)
        t_train_data = fit.transform(train_data)

        model.fit(t_train_data, train_labels)
        t_test_data = fit.transform(test_data)
        pred = model.predict(fit.transform(test_data))
        # print(fit.scores_)

        err = metrics.mean_squared_error(pred, test_labels)
        print(f"Number of features : {num_features}/ {train_data.shape[1]} MSE : {err}")

        return t_train_data, t_test_data, test.get_feature_names_out(train_data.columns), err

    def feature_selection_pipeline(self, model, train_data, train_labels, test_data, test_labels):

        """perform feature selection on the data and return the transformed data"""

        print("***Running feature selection pipeline***\n")

        # Find best number of features
        res = []
        for k in range(int(len(train_data.columns)/2), len(train_data.columns)+1):
            t_train_data, t_test_data, selected_features, err = \
                self.SelectKBest_feature_selection(sklearn.base.clone(model), train_data, train_labels,
                                                   test_data, test_labels, k)
            t_train_data_df, t_test_data_df = pd.DataFrame(data=t_train_data, columns=selected_features), \
                                              pd.DataFrame(data=t_test_data, columns=selected_features)
            res.append({'transformed_data': [t_train_data_df, t_test_data_df],
                        'err': err,
                        'selected_features': selected_features})

        best = min(res, key=lambda x:x['err'])
        ret_data = best['transformed_data']
        print("\n***End of feature selection phase***\n")
        print(f"Num of selected features ===> {ret_data[0].shape[1]} \n Selected features : {best['selected_features']} err :{best['err']}")

        return ret_data

    def model_train_pipeline(self, model, t_trained_data, t_trained_labels):
        
        """train the model, and use pipeline methods"""

        ## train the model
        model.fit(t_trained_data, t_trained_labels)

        # create some predictions on train data (just for sanity)
        # preds_on_train_data = model.predict(train_data)
        # print_err(train_labels, preds_on_train_data, "Train")

        return model

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

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.3, random_state=6)

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
    baseline(data_motor, labels_motor, base_models.frenchBaseModel())
    trained_model_motor = pipeline.apply(data_motor, labels_motor, base_models.frenchBaseModel())

    print("\n*******************************************************")
    print("******************* Boston Data ***********************")
    print("*******************************************************\n")

    data_boston, labels_boston = datasets_utils.loadBostonData()
    baseline(data_boston, labels_boston, base_models.bostonBaseModel())
    trained_model_boston = pipeline.apply(data_boston, labels_boston, base_models.bostonBaseModel())





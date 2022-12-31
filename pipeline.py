from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
import numpy as np
import datasets_utils
import base_models
from hpd import search
from ipdb import set_trace


def _transformers(categorial_cols=()):
    if categorial_cols:
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown='ignore'))])
        return [("cat", categorical_transformer, categorial_cols)]


def build_pipeline(categorial_cols, model):
    _trans = _transformers(categorial_cols)
    if _trans:
        preProcessor = ColumnTransformer(transformers=_trans)
        return Pipeline(steps=[("preproces", preProcessor), ("model", model)])

    else:
        return model


def train(X, y, model):

    categorial_cols = tuple(colName for colName in X.columns if X[colName].dtype == 'O')
    finalModel = build_pipeline(categorial_cols=categorial_cols, model=model)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
    finalModel.fit(X_train, y_train)

    y_pred_on_train_data = finalModel.predict(X_train)

    y_pred = finalModel.predict(X_test)

    print('MAE:',metrics.mean_absolute_error(y_train, y_pred_on_train_data))
    print('MSE:',metrics.mean_squared_error(y_train, y_pred_on_train_data))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred_on_train_data)))

    print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:',metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return finalModel


class Pipeline:

    def __init__(self):
        pass

    def apply(self, data, model):

        """this function returns trained model after pipeline"""

        ## prepare data for training, cleaning, etc.
        clean_data = self.data_pipeline(data)

        ## train the model 
        model = self.model_pipeline(model, clean_data)

        return model

    def data_pipeline(self, data):

        """clean and validate data"""
        pass

    def model_pipeline(self, model, clean_data):
        
        """train the model, and use pipeline methods"""

        instances, labels = clean_data
        train_instances, test_instances, train_labels, test_labels = \
                train_test_split(instances, labels, test_size = 0.3)

        ## train the model
        model.fit(train_instances, train_labels)

        ## train again with feature selection
        model = self.feature_selection_training(model, clean_data)


if __name__ == "__main__":
    #X, y = datasets_utils.loadFrenchMotorData()
    #model = base_models.frenchBaseModel()
    #model = train(X, y, model)
    #search(X)
    

    pipeline = Pipeline()

    data_motor = datasets_utils.loadFrenchMotorData()
    set_trace()
    model_motor = base_models.frenchBaseModel()
    trained_model_motor = pipeline.apply(data_motor, model_motor)

    data_boston = 1
    model_boston = 1
    trained_model_boston = pipeline.apply(data_boston, model_boston)











    





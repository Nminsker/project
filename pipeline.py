from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import datasets_utils
import base_models
from hpd import search
from ipdb import set_trace


class Pipeline:

    def __init__(self):
        pass

    def apply(self, data, labels, model):

        """this function returns trained model after pipeline"""

        ## prepare data for training, cleaning, etc.
        clean_data = self.data_pipeline(data)

        ## train the model 
        model = self.model_pipeline(model, clean_data, labels)

        return model

    def data_pipeline(self, data):

        """clean and validate data"""

        ## transform non numeric data to categorial numeric data:
        for col in data.dtypes[data.dtypes=='object'].index:
            data[col], _ = data[col].factorize() 
        
        ## remove Nons
        # TODO

        return data

    def model_pipeline(self, model, clean_data, labels):
        
        """train the model, and use pipeline methods"""

        train_data, test_data, train_labels, test_labels = \
                train_test_split(clean_data, labels, test_size = 0.3)

        ## train the model
        model.fit(train_data, train_labels)

        ## create some predictions on train data (just for sanity)
        preds_on_train_data = model.predict(train_data) 

        ## create some predictions on test data
        preds = model.predict(test_data) 
        for pred, label in zip(preds,test_labels):
            print(pred, label)


        print('MAE:',metrics.mean_absolute_error(train_labels, preds_on_train_data))
        print('MSE:',metrics.mean_squared_error(train_labels, preds_on_train_data))
        print('RMSE:',np.sqrt(metrics.mean_squared_error(train_labels, preds_on_train_data)))

        print('MAE:',metrics.mean_absolute_error(test_labels, preds))
        print('MSE:',metrics.mean_squared_error(test_labels, preds))
        print('RMSE:',np.sqrt(metrics.mean_squared_error(test_labels, preds)))

        ## train again with feature selection
        #TODO model = self.feature_selection_training(model, clean_data)

        return model


if __name__ == "__main__":

    pipeline = Pipeline()

    data_motor, labels_motor = datasets_utils.loadFrenchMotorData()
    model_motor = base_models.frenchBaseModel()
    trained_model_motor = pipeline.apply(data_motor, 
                                         labels_motor, 
                                         model_motor)

    data_boston, labels_boston = datasets_utils.loadBostonData()
    model_boston = base_models.bostonBaseModel()
    trained_model_boston = pipeline.apply(data_boston, 
                                          labels_boston, 
                                          model_boston)





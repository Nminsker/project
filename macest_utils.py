import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from macest.regression import models as reg_mod
from macest.regression import plots as reg_plot
from typing import Union
import os


class MacestModel:
    def __init__(self, model, X, y):

        self.model = model

        self.X_pp_train, self.X_conf_train, self.X_cal, self.X_test,
        self.y_pp_train, self.y_conf_train, self.y_cal, self.y_test =\
            self.split_data(X, y)

        self.calibrate_macest()

    def split_data(self, X, y):

        X_pp_train, X_conf_train, y_pp_train, y_conf_train =\
                train_test_split(X, y, test_size=0.66, random_state=10)

        X_conf_train, X_cal, y_conf_train, y_cal =\
                train_test_split(X_conf_train, 
                                 y_conf_train,
                                 test_size=0.5, 
                                 random_state=42)

        X_cal, X_test, y_cal, y_test =\
            train_test_split(X_cal, y_cal, test_size=0.5, random_state=42)

        [X_pp_train, X_conf_train, X_cal, X_test] =\
                map(convert_to_np_array, 
                    [X_pp_train, X_conf_train, X_cal, X_test])

        return X_pp_train, X_conf_train, X_cal, X_test,\
                   y_pp_train, y_conf_train, y_cal, y_test


    def calibrate_macest(self):

        self.model.fit(self.X_pp_train, self.y_pp_train)
        model_preds = self.model.predict(self.X_conf_train)
        test_error = np.array(abs(model_preds - self.y_conf_train))
        search_params = reg_mod.HnswGraphArgs(query_kwargs={'ef': 1500})

        self.macest_model = reg_mod.ModelWithPredictionInterval(
                self.model, self.X_conf_train, test_error,
                search_method_args=search_params,
                dist_func="error_weighted_poly")

        self.macest_model.fit(
                self.X_cal, 
                self.y_cal, 
                param_range=reg_mod.SearchBounds(k_bounds=(2, 20)))


    def predict_with_interval(self, 
                              data=None, 
                              conf_level: Union[np.ndarray, int, float] = 90):

        data = self.X_test if data is None else data

        return self.macest_model.predict_interval(data, conf_level)


    def sample_prediction(self, data=None, nsamples=10**3):
        data = self.X_test if data is None else data
        return self.macest_model.sample_prediction(data, nsamples)


    def predict(self, data):
        return self.model.predict(data)


    # Still needs to be checked
    def create_plots(self):

        current_dir = os.getcwd()
        plots_dir = os.path.join(current_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        (test_size, *z) = self.X_test.shape
        test_size_condition = test_size <= 10**4

        plot_data =\
                self.X_test if test_size_condition else self.X_test[0:10**3]
        plot_true =\
                self.y_test if test_size_condition else self.y_test[0:10**3]

        reg_plot.plot_calibration(
                self.macest_model, 
                plot_data, 
                plot_true, 
                save=True, 
                save_dir=os.path.join(plots_dir, 'plot1'))

        # reg_plot.plot_pred_with_conf(
        #    self.macest_model, plot_data, plot_true, save=True,
        #    save_dir=os.path.join(plots_dir, 'plot2'))

        reg_plot.plot_predicted_vs_true(
                self.model, 
                self.macest_model, 
                plot_data, 
                plot_true, 
                save=True,
                save_dir=os.path.join(plots_dir, 'plot3'))

        reg_plot.plot_true_vs_predicted(
                self.model, 
                self.macest_model, plot_data, plot_true, save=True,
                save_dir=os.path.join(plots_dir, 'plot4'))


def convert_to_np_array(data):
    res = data.toarray() if type(data) == scipy.sparse.csr.csr_matrix else data
    return res

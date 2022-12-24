from feature_engine import enconding
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import datasets_utils
from base_models import bostonBaseModel

def customOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if any(X[colName].dtype == 'O' for colName in X.columns):
            ohe = enconding.OneHotEncoder()
            return ohe.fit(X,y)
        else:
            self


if __name__ == "__main__":
    model = bostonBaseModel()
    pipeline = Pipeline(steps=[
        ("categories", customOneHotEncoder),
        ("model", model)])

    X, y = datasets_utils.loadBostonData()
    pipeline.fit(X, y)

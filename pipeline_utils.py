from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import datasets_utils
import base_models

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


def run(X, y, model):
    categorial_cols = tuple(colName for colName in X.columns if X[colName].dtype == 'O')
    finalModel = build_pipeline(categorial_cols=categorial_cols, model=model)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
    finalModel.fit(X_train, y_train)
    predicts = finalModel.predict(X_test)
    for p, t in zip(predicts, y_test):
        print(p, t)


if __name__ == "__main__":
    X, y = datasets_utils.loadFrenchMotorData()
    model = base_models.frenchBaseModel()
    run(X, y, model)






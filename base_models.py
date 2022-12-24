from xgboost import XGBRegressor

def bostonBaseModel():
    """ return a the base model used for the Boston dataset """
    return XGBRegressor()


def fernchBaseModel():
    """ return a base model used for the french motor dataset """
    return XGBRegressor(eval_metric='poisson-nloglik',
                        objective="count:poisson", 
                        colsample_bytree=0.9, 
                        learning_rate=0.1,
                        max_depth=3,
                        min_child_weight=1,
                        reg_alpha=5,
                        subsample=0.9
                        )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

class Reccommender():
    
    def __init__(self, reg_price=None, clf_star=None,
                       clf_meal=None, reg_dur=None, cls_city=None,
                 is_prod=False):
        self.clf_star = clf_star
        self.clf_meal = clf_meal
        self.reg_price = reg_price
        self.reg_dur = reg_dur
        self.cls_city = cls_city
        self.is_prod = is_prod
        
        self.model_params = dict(
            thread_count=8,
            iterations=100,
#             loss_function=objective,
#             eval_metric=eval_metric,
            learning_rate=0.1,
#             depth=HrPrmOptChoise(6, list(range(2, 11))),
            bagging_temperature=0.8,
            rsm=0.9,
            allow_writing_files=False,
            save_snapshot=False
        )

    def get_training_params(self, df):
        _df = (df.dtypes == int).reset_index()
        cat_features = _df[_df[0] == True].index.tolist()
        return {
            'verbose': 50,
            'cat_features': cat_features
        }
        
    def fit_reg_price(self, train):
        X = train.loc[:, train.columns != 'Price']
        y = train['Price']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostRegressor(**self.model_params)
        training_params = self.get_training_params(X_train)
        model.fit(X_train, y_train, **training_params)
        self.reg_price = model
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test.values, preds))
        mae = mean_squared_error(y_test.values, preds)
        return rmse, mae

    def fit_reg_dur(self, train):
        X = train.loc[:, train.columns != 'N Nights']
        y = train['N Nights']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        
        model = CatBoostRegressor(**self.model_params)
        training_params = self.get_training_params(X_train)
        model.fit(X_train, y_train, **training_params)
        self.reg_dur = model
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test.values, preds))
        mae = mean_squared_error(y_test.values, preds)
        return rmse, mae

    def fit_clf_star(self, train):
        X = train.loc[:, train.columns != 'Star Rate']
        y = train['Star Rate']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(**self.model_params)
        training_params = self.get_training_params(X_train)
        model.fit(X_train, y_train, **training_params)
        self.clf_star = model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test.values, preds)
        return acc

    def fit_clf_meal(self, train):
        X = train.loc[:, train.columns != 'Meal Option']
        y = train['Meal Option']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(**self.model_params)
        training_params = self.get_training_params(X_train)
        model.fit(X_train, y_train, **training_params)
        self.clf_meal = model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test.values, preds)
        return acc
   
    def fit_city(self, train):
        X = train.loc[:, train.columns != 'city_to']
        y = train['city_to']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(**self.model_params)
        training_params = self.get_training_params(X_train)
        model.fit(X_train, y_train, **training_params)
        self.clf_meal = model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test.values, preds)
        return acc

    def fit(self, train):
        fit_city_acc = self.fit_city(train)
        meal_acc = self.fit_clf_meal(train)
        star_acc = self.fit_clf_star(train)
        dur_rmse, dur_mae = self.fit_reg_dur(train)
        price_rmse, price_mae = self.fit_reg_price(train)
        return {"meal_acc": meal_acc,
                "star_acc": star_acc,
                "dur_rmse": dur_rmse,
                "dur_mae": dur_mae,
                "price_rmse": price_rmse,
                "price_mae": price_mae}

    def predict(self, user):
        #TODO: add maping
        city = self.clf_city.predict(user)[0]
        price = self.reg_price.predict(user)
        star = self.clf_star.predict(user)[0]
        duration = np.around(self.reg_dur.predict(user))
        meal_type = self.clf_meal.predict(user)[0]
        return {"city": city,
                "price": price,
                "star": star,
                "duration": duration,
                "meal_type": meal_type}
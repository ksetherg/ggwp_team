import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor, CatBoostClassifier

class Reccommender():

    def __init__(self, reg_price=None, clf_star=None,
                       clf_meal=None, reg_dur=None, is_prod=False):
        self.clf_star = clf_star
        self.clf_meal = clf_meal
        self.reg_price = reg_price
        self.reg_dur = reg_dur
        self.is_prod = is_prod

    def fit_reg_price(self, train):
        X = train.loc[:, train.columns != 'Price']
        y = train['Price']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostRegressor()
        model.fit(X_train, y_train)
        self.reg_price = model

    def fit_reg_dur(self, train):
        X = train.loc[:, train.columns != 'N Nights']
        y = train['N Nights']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        
        model = CatBoostRegressor()
        model.fit(X_train, y_train)
        self.reg_dur = model

    def fit_clf_star(self, train):
        X = train.loc[:, train.columns != 'Star Rate']
        y = train['Star Rate']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostClassifier()
        model.fit(X_train, y_train)
        self.clf_star = model
    
    def fit_clf_meal(self, train):
        X = train.loc[:, train.columns != 'Meal Option']
        y = train['Meal Option']
        if self.is_prod:
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier()
        model.fit(X_train, y_train)
        self.clf_meal = model

    def fit(self, train):
        self.fit_clf_meal(train)
        self.fit_clf_star(train)
        self.fit_reg_dur(train)
        self.fit_reg_price(train)

    def predict(self, user):
        price = self.reg_price.predict(user)
        star = self.clf_star.predict(user)[0]
        duration = np.around(self.reg_dur.predict(user))
        meal_type = self.clf_meal.predict(user)[0]
        return {"price": price,
                "star": star,
                "duration": duration,
                "meal_type": meal_type}
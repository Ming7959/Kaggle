# coding=gbk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import norm, skew
import sklearn.metrics as metrics
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as XGB

train_data = pd.read_csv("train.csv")  # [1460 rows x 81 columns]
test_data = pd.read_csv("test.csv")  # [1459 rows x 80 columns]

# print(train_data.head())

# print(train_data.info())

# print(train_data['SalePrice'].describe())

# sns.distplot(train_data['SalePrice'])
# plt.show()

# print("Skewness: %f" % train_data['SalePrice'].skew())
# print("Kurtosis: %f" % train_data['SalePrice'].kurt())

train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
# sns.distplot(train_data['SalePrice'], fit=norm)
# plt.show()

# sns.boxplot(data=train_data, x='OverallQual', y='SalePrice')
# plt.show()

# plt.figure(figsize=(12,4),dpi=130)
# sns.boxplot(data=train_data,x='YearBuilt',y='SalePrice')
# plt.xticks(rotation=90)
# plt.show()

# sns.scatterplot(data=train_data, x='TotalBsmtSF', y='SalePrice')
# plt.show()

# sns.scatterplot(data=train_data, x='GrLivArea', y='SalePrice')
# plt.show()

# Let's check which features are the most corelated
# print(train_data.corr()["SalePrice"].sort_values(ascending=False)[1:])

# corrmat = train_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

# corr = train_data.corr()
# highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
# plt.figure(figsize=(10, 10))
# g = sns.heatmap(train_data[highest_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.show()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_data[cols])
# plt.show()

y_train = train_data['SalePrice']
test_id = test_data['Id']
all_data = pd.concat([train_data, test_data], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)

Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(25))

all_data.drop((missing_data[missing_data['Total'] > 5]).index, axis=1, inplace=True)
# print(all_data.isnull().sum().max())

total = all_data.isnull().sum().sort_values(ascending=False)
# print(total.head(19))

numeric_missed = ['BsmtFinSF1',
                  'BsmtFinSF2',
                  'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath',
                  'BsmtHalfBath',
                  'GarageArea',
                  'GarageCars']

for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)

categorical_missed = ['Exterior1st',
                      'Exterior2nd',
                      'SaleType',
                      'MSZoning',
                      'Electrical',
                      'KitchenQual']

for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])


all_data['Functional'] = all_data['Functional'].fillna('Typ')

all_data.drop(['Utilities'], axis=1, inplace=True)

# print(all_data.isnull().sum().max())

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
# print(high_skew)

for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data = pd.get_dummies(all_data)
# print(all_data.head())
# print(all_data.shape)

x_train =all_data[:len(y_train)]  # (1460, 219)
x_test = all_data[len(y_train):]  # (1459, 219)

scorer = make_scorer(mean_squared_error, greater_is_better=False)


def rmse_CV_train(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring ="neg_mean_squared_error", cv=kf))
    return rmse


# def rmse_CV_test(model):
#     kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_data.values)
#     rmse = np.sqrt(-cross_val_score(model, x_test, y_test,scoring ="neg_mean_squared_error", cv=kf))
#     return rmse


the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state=7, nthread=-1)
the_model.fit(x_train, y_train)

y_predict = np.floor(np.expm1(the_model.predict(x_test)))

# print(y_predict)

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_predict
sub.to_csv('mysubmission.csv', index=False)
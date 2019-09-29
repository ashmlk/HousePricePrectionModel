#%%
import matplotlib.pyplot as plt
import matplotlib as mplpy
import seaborn as sns 
import pandas as pd 
import numpy as np

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#import testing and training dataset
train = pd.read_csv('D:\\My Projects\\Python Projects\\house\\train.csv')
test = pd.read_csv('D:\\My Projects\\Python Projects\\house\\test.csv')

#Show number of columns and rows
train.shape

#show all the columns of our dataset
train.columns

#getting all the statistical information on house prices
train['SalePrice'].describe()

#select only datas that have numberical value 
train_corr = train.select_dtypes(include=[np.number])

#delete the id(it is not useful for determining correlation between data and is a noisy feature
del train_corr['Id']

#Heated map showing correlation between all numerical values
corr = train_corr.corr()
plt.subplots(figsize=(27,12))
sns.heatmap(corr, annot=True)

#Top 60% of correlative data
top_feature = corr.index[abs(corr['SalePrice']>0.6)]
plt.subplots(figsize=(16, 10))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

#Listing most correlated features
most_corr = pd.DataFrame(top_feature)
most_corr.columns = ['Most Correlated Features']
most_corr

#different plots based on different parameters 
#Box plot based on Quality and Price
sns.barplot(train.OverallQual, train.SalePrice)
plt.show()

#Pairgrid
col = ['SalePrice', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(train[col], height=3, kind='scatter')

#Scatter plot based on total sqft, YearsBuilt, Neighborhood, Kitchen Qual, Garage Cars, Pool Area
fig = plt.figure(figsize=(30, 20))
sns.set(style='ticks')
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

fig2 = fig.add_subplot(231); 
sns.scatterplot(train['TotalSF'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')

fig3 = fig.add_subplot(232); 
sns.scatterplot(train['YearBuilt'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')

fig4 = fig.add_subplot(233); 
sns.scatterplot(train['Neighborhood'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')

fig5 = fig.add_subplot(234); 
sns.scatterplot(train['KitchenQual'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')

fig6 = fig.add_subplot(235); 
sns.scatterplot(train['GarageCars'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')

fig7 = fig.add_subplot(236); 
sns.scatterplot(train['PoolArea'], train['SalePrice'], hue=train['OverallQual'], palette= 'Spectral')


#Delete the Id feature as it is not correlated
train_Id = train['Id']
del train['Id']

#Feature correlation to Sales price
corr_1 = train.corr()
corr_1.sort_values(['SalePrice'], ascending=False, inplace=True)
corr_1.SalePrice

#We have missing values, lets find them
train.columns[train.isnull().any()]

#Find number of each null value and their names
# and add them to two objects for plotting
data_not_available = train.isnull().sum()
data_not_available = data_not_available[data_not_available>0]
data_not_available = data_not_available.to_frame()
data_not_available.columns = ['count']
data_not_available.index.names= ['name']
data_not_available['name'] = data_not_available.index

#Plot the graph
plt.figure(figsize=(15, 7))
sns.set(style='ticks')
sns.barplot(x='name', y='count', data=data_not_available)
plt.show()

#filling in missing values
train = train.drop(['Utilities'], axis=1)
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('MasVnrType', 'PoolQC', 'MiscFeature', 'Alley', 'FireplaceQu', 'Fence', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MSSubClass'):
    train[col] = train[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    train[col] = train[col].fillna(0)
for col in ('MSZoning','Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
    train[col] = train[col].fillna(train[col].mode()[0])

#Checking to see if we have missing data
plt.figure(figsize=(12, 3))
sns.heatmap(train.isnull())

#We find the type 'Object' values
train.select_dtypes(include=['object']).columns

#We change the strings retrived from above to numeric values
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual',
        'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')

#From string to integer
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lb = LabelEncoder()
    lb.fit(list(train[c].values))
    train[c] = lb.transform(list(train[c].values))

#Import libraries for classification and regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Training and test set
X = train.values
y = train['SalePrice'].values
del train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

#Linear regression
from sklearn.linear_model import LinearRegression
li_r = LinearRegression()
li_r.fit(X_train, y_train)
prediction_lir = li_r.predict(X_test)
print('Linear regression accuracy is: ', li_r.score(X_test, y_test)*100)

#Logistics regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
prediction_lr = lr.predict(X_test)
print('Logistic regression accuracy is: ', lr.score(X_test, y_test)*100)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
prediction_dt = dt.predict(X_test)
print('DT accuracy is: ', accuracy_score(prediction_dt, y_test)*100)

#kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
prediction_knn = knn.predict(X_test)
print('kNN accuracy is: ', accuracy_score(prediction_knn, y_test)*100)

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, max_depth = 2)
gbc.fit(X_train, y_train)
prediction_gbc = gbc.predict(X_test)
print('GBC accuracy is: ', accuracy_score(prediction_gbc, y_test)*100)

prediction = pd.DataFrame(prediction_dt, prediction_gbc, prediction_knn, prediction_lir, prediction_lr, columns=['DecisionTree', 'GBC', 'kNN', 'LinearRegression', 'LogisticRegression'])
prediction.to_csv('results.csv')

#%%

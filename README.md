# Iowa-Housing-Prices-ML
Reading data file and understanding statistics using pandas from analysing housing prices in Iowa data-1st part of Machine Learning Course

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
iowa_file_path='../input/home-data-for-ml-course/train.csv'
iowa_data=pd.read_csv(iowa_file_path)
home_data.describe()
y=home_data.SalePrice
feature_columns=['LotArea','YearBuilt,'1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']
X=home_data[feature_colums]

iowa_model=DecisionTreeRegressor()
iowa_model.fit(X,y)
print('First in-sample predictions:',iowa_model.predict(X.head()))
print('Actual target values for those homes:',y.head().tolist())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_X,val_x,train_y,val_y=train_test_split(X,y,random state=1)

iowa_model=DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X,train_y)
val_predicitions=iowa_model.predict(val_X)
val_mae=mean_absolute_error(val_y,val_predictions)
print(val_mae)



Answers:
avg_lot_size=10517
newest_home_age=15
step_2.check()
#Would I trust my data? The newest house is 15 years old which isn't that recent, if they havent built any new houses since then, then my data is accurate. However if it is because new data hasn't been recorded, then I would extrapolate my data, alongside comparing what happens to housing prices the older they get.

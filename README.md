# Kaggle's Intro to Machine Learning (Python): Iowa-Housing-Prices-ML
Predicting housing prices with property features in Iowa dataset using ML models; simple like Decision Trees and advanced like Random Forests 
Libraries Used: pandas,sickit_learn

Results: The Random Forest model resulted in a validation Mean Absolute Error(mae) of 21857.15912981083
The Decision Tree model resulted in a validation mae for max leaf nodes: 27,283
```python
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path='../input/home-data-for-ml-course/train.csv'
iowa_data=pd.read_csv(iowa_file_path)
home_data.describe()
y=home_data.SalePrice
feature_columns=['LotArea','YearBuilt,'1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']
X=home_data[feature_colums]

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)

iowa_model=DecisionTreeRegressor(random state=1)
iowa_model.fit(X,y)
print('First in-sample predictions:',iowa_model.predict(X.head()))
print('Actual target values for those homes:',y.head().tolist())

#Avoiding underfitting and overfitting; checking mae and landed at 100 for depth of regression tree
iowa_model=DecisionTreeRegressor(max_leaf_nodes=100,random_state=1))
iowa_model.fit(train_X,train_y)
val_predicitions=iowa_model.predict(val_X)
val_mae=mean_absolute_error(val_y,val_predictions)
print('Validation MAE for best value of max_leaf_nodes:{:,.0f}'.format(val_mae))

from sklearn.ensemble import RandomForestRegressor
fr_model=RandomForestRegressor(random_state=1)
rf-model.fit(train_X,train_y)
rf_val_predictions=rf_model.predict(val_X)
rf_val_mae=mean_absolute_error(val_y,rf_val_predictions)
print('Validation MAE for Random Forest Model: {}'.format(rf_val_mae))




#Would I trust my data? The newest house is 15 years old which isn't that recent, if they havent built any new houses since then, then my data is accurate. However if it is because new data hasn't been recorded, then I would extrapolate my data, alongside comparing what happens to housing prices the older they get.

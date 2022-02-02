# Importing Libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Fetching Dataset
dataset= pd.read_csv(r"C:\Users\dell\PycharmProjects\Boston Houses Price Prediction\housing.csv")

# Dividing the data into features and labels
x= dataset.iloc[:,0:3]         # Input
y=dataset.iloc[:,-1]           # Output

# Dividing the data into Training and Test data
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

# Training our Model and Testing it
M2=RandomForestRegressor()
M2.fit(x_train,y_train)
predict=M2.predict(x_test)
print(predict)

# Checking our model and its R2 Score
R2_Score= r2_score(y_test,predict)
print(R2_Score)

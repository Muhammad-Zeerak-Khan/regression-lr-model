# importing the necessary dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#Reading the data
data =pd.read_csv('Admission_Prediction.csv')

#Handling the missing values
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())

#Dropping the unnecessary column
data= data.drop(columns = ['Serial No.'])

#Dependent and Independent Columns
y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])

#Scaling the features

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)


#Saving the standard scaler model for transformation of test data
file="standard_scaler_transformation.pickle"
pickle.dump(X_scaled,open(file,'wb'))

#Train_train_split
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)

#Calling the Regression model
regression = LinearRegression()

#Applying the model to the dataset
regression.fit(x_train,y_train)

# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))


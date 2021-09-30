import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# from google.colab import drive
# drive.mount('/content/drive')

dataset = pd.read_csv("Salary_Data.csv")

dataset.head()

# dataset.shape

dataset.describe()

a=dataset['YearsExperience']
print(type(a))

X = dataset.iloc[: , :-1].values   #[: , :]  #independant variable
y = dataset.iloc[:,-1].values   # dependant

type(X)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

X_train

y_train

X_test

y_test

from sklearn.linear_model import LinearRegression
regression_model=LinearRegression()
regression_model.fit(X_train,y_train)

#y=b0+b1X
# print(regression_model.intercept_)  #b0
# print(regression_model.coef_)  #b1


y_pred=regression_model.predict(X_test)
y_pred

y_test



# plt.scatter(X_train,y_train,color="red")
# plt.plot(X_train,regression_model.predict(X_train))

# plt.xlabel("Years of Experinace")
# plt.ylabel("Salary")

# plt.scatter(X_test,y_test,color="red")
# plt.plot(X_test,regression_model.predict(X_test))

# plt.xlabel("Years of Experinace")
# plt.ylabel("Salary")

from sklearn import metrics   #performance eveluation of matrics

print(metrics.mean_absolute_error(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred)) #MSE

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))  #RSME

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

import streamlit as st
def predicted_salary(years_of_experience):
    years_of_experience =  [[years_of_experience]]
    salary = regression_model.predict(years_of_experience)
    return salary

def display_graph():
    st.subheader("Training Data set")
    st.line_chart(pd.DataFrame(X_train, y_train))

    st.subheader("Testing Data set")
    st.line_chart(pd.DataFrame(X_test, y_pred))
import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("sales_20.csv")

X=data[['Month']]
y=data['Sales']

model=LinearRegression()
model.fit(X,y)

future_month=int(input("Enter future month: "))

prediction=model.predict([[future_month]])

print("Predicted Sales:",prediction[0])

import pandas as pd
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv("loan_19.csv")

X=data[['Age','Income','CreditScore']]
y=data['LoanApproved']

model=GaussianNB()
model.fit(X,y)

age=int(input("Age: "))
income=int(input("Income: "))
score=int(input("Credit Score: "))

pred=model.predict([[age,income,score]])

print("Loan Approval Prediction:",pred[0])

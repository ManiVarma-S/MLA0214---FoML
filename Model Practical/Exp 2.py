import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Step 1: Dataset
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old'],
    'BrowsingTime': ['High', 'Low', 'High', 'Low'],
    'PastPurchase': ['Yes', 'No', 'Yes', 'No'],
    'Buy': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Encoding (IMPORTANT)
df_knn = df.copy()
le = LabelEncoder()

for col in df_knn.columns:
    df_knn[col] = le.fit_transform(df_knn[col])

# Step 3: Features & Target
X = df_knn[['Age', 'BrowsingTime', 'PastPurchase']]
y = df_knn['Buy']

# Step 4: Model
model = LogisticRegression()
model.fit(X, y)

# Step 5: Prediction
test = [[0, 0, 1]]
print(model.predict(test))

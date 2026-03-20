import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ===== STEP 1: DATASET =====
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old'],
    'BrowsingTime': ['High', 'Low', 'High', 'Low'],
    'PastPurchase': ['Yes', 'No', 'Yes', 'No'],
    'Buy': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# ===== STEP 2: ENCODING (VERY IMPORTANT) =====
df_knn = df.copy()

le = LabelEncoder()
for col in df_knn.columns:
    df_knn[col] = le.fit_transform(df_knn[col])

print("\nEncoded Data:\n", df_knn)

# ===== STEP 3: FEATURES =====
X = df_knn[['Age', 'BrowsingTime', 'PastPurchase']]
y = df_knn['Buy']

# ===== STEP 4: MODEL =====
model = LogisticRegression()
model.fit(X, y)

# ===== STEP 5: PREDICTION =====
test = [[0, 0, 1]]
prediction = model.predict(test)

print("\nPrediction:", prediction)

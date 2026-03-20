import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== STEP 1: DATASET =====
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old'],
    'BrowsingTime': ['High', 'Low', 'High', 'Low'],
    'PastPurchase': ['Yes', 'No', 'Yes', 'No'],
    'Buy': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# ===== STEP 2: ENCODING =====
df_knn = df.copy()
le = LabelEncoder()

for col in df_knn.columns:
    df_knn[col] = le.fit_transform(df_knn[col])

# ===== STEP 3: FEATURES =====
X = df_knn[['Age', 'BrowsingTime', 'PastPurchase']]
y = df_knn['Buy']

# ===== STEP 4: KNN MODEL =====
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn_pred = knn.predict(X)

# ===== STEP 5: LOGISTIC MODEL =====
lr = LogisticRegression()
lr.fit(X, y)
lr_pred = lr.predict(X)

# ===== STEP 6: EVALUATION =====
print("===== KNN RESULTS =====")
print("Accuracy:", accuracy_score(y, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y, knn_pred))

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", accuracy_score(y, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y, lr_pred))

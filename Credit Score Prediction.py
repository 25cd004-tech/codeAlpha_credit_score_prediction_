import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('credit_score_dataset.csv')
print(data.head())
data = data.drop("customer_id", axis=1)
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
X=data.drop("credit_rating", axis=1)
y=data['credit_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("\nLOGISTIC REGRESSION")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred,average='weighted'))
print("Precision:",precision_score(y_test,y_pred,average='weighted'))
print("Recall:",recall_score(y_test,y_pred,average='weighted'))
model1 =DecisionTreeClassifier(random_state=42)
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
print("\nDECISION TREE CLASSIFIER")
print("Accuracy:",accuracy_score(y_test,y_pred1))
print("Classification Report:\n",classification_report(y_test,y_pred1))
print("F1 Score:",f1_score(y_test,y_pred1,average='weighted'))
print("Precision:",precision_score(y_test,y_pred1,average='weighted'))
print("Recall:",recall_score(y_test,y_pred1,average='weighted'))  
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train,y_train)
y_pred3 = model2.predict(X_test)
print("\nRANDOM FOREST CLASSIFIER")
print("Accuracy:",accuracy_score(y_test,y_pred3))
print("Classification Report:\n",classification_report(y_test,y_pred3))
print("F1 Score:",f1_score(y_test,y_pred3,average='weighted'))
print("Precision:",precision_score(y_test,y_pred3,average='weighted'))
print("Recall:",recall_score(y_test,y_pred3,average='weighted'))  
importance = pd.DataFrame({"Feature":X.columns,"Importance":model2.feature_importances_})
importance = importance.sort_values(by="Importance",ascending=False)
print("\nFeature Importance:\n",importance)
print("\nFEATURE IMPORTANCE PLOT:")
print(importance)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
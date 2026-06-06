import pandas as pd
from xgboost import XGBClassifier, train
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('heart_disease_dataset_1.csv')
le = LabelEncoder()
data['gender']= le.fit_transform(data['gender'])
X= data.drop('heart_disease',axis=1)
y= data['heart_disease']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# XGBoost Classifier
xgc_model = XGBClassifier(random_state=42)
xgc_model.fit(X_train, y_train)
y_pred_xgc = xgc_model.predict(X_test)
print("XGBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_xgc))
print("XGBoost Classification Report:\n",classification_report(y_test, y_pred_xgc))     
# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:\n",classification_report(y_test, y_pred_dt))
# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n",classification_report(y_test, y_pred_rf))
# Logistic Regression
lr_model = LogisticRegression(max_iter=50000, random_state=42)
lr_model.fit(X_train,y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n",classification_report(y_test, y_pred_lr))
importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
print("Feature Importance:\n", importance.sort_values(by='Importance', ascending=False).head(10))
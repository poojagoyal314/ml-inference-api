from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib


X, y = load_iris(return_X_y=True)                                
model = LogisticRegression(max_iter=200).fit(X,y)                                           
joblib.dump(model, "model.joblib")
print("saved model.joblib")


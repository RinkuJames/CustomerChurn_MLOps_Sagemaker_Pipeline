import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("/opt/ml/processing/model/model.pkl")
test_df = pd.read_csv("/opt/ml/processing/test/test.csv")

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

report = {
    "metrics": {
        "accuracy": {
            "value": accuracy
        }
    }
}

with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(report, f)

print("Evaluation complete.")

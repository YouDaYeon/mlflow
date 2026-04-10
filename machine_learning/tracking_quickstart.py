# MLflow Tracking Quickstart
# https://mlflow.org/docs/latest/ml/getting-started/quickstart/#step-1---set-up-mlflow

# Step 1. Set up MLflow
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # 웹 대시보드에 기록
mlflow.set_experiment("MLflow Quickstart")


# Step 2. Prepare training data
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# # Step 3. Train a model with MLflow Autologging
# import mlflow

# # Enable autologging for scikit-learn
# mlflow.sklearn.autolog()

# # Just train the model normally
# lr = LogisticRegression(**params)
# lr.fit(X_train, y_train)

# Step 5. start_run() 블록 추가
with mlflow.start_run():
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # 추가 기록(선택)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)

    print(f"학습 완료! Accuracy: {acc}")

    mlflow.set_tag("Training Info", "Basic LR model for iris data")


# Step 6. 모델 load 추론

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result.head(4))
# MLflow Hyperparameter Tuning
# https://mlflow.org/docs/latest/ml/getting-started/hyperparameter-tuning/

# pip install mlflow optuna

#Step 1: Create a new experiment
import mlflow

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # 웹 대시보드에 기록
mlflow.set_experiment("Hyperparameter Tuning Experiment")


# Step 2: Prepare Your Data
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# Step 3-1: Define the objective function
import mlflow
import optuna
import sklearn


def objective(trial):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=10)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)
        params = {
            "max_depth": rf_max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
        }
        # Log current trial's parameters
        mlflow.log_params(params)

        regressor_obj = sklearn.ensemble.RandomForestRegressor(**params)
        regressor_obj.fit(X_train, y_train)

        y_pred = regressor_obj.predict(X_val)
        error = sklearn.metrics.mean_squared_error(y_val, y_pred)
        # Log current trial's error metric
        mlflow.log_metrics({"error": error})
        

        # Log the model file
        mlflow.sklearn.log_model(regressor_obj, name="model")
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        print(f"Trial {trial.number} error: {error}")
        return error


# Step 3-2: Run the hyperparameter tuning study
# Create a parent run that contains all child runs for different trials
with mlflow.start_run(run_name="study") as run:
    # Log the experiment settings
    n_trials = 30
    mlflow.log_param("n_trials", n_trials)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Log the best trial and its run ID
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metrics({"best_error": study.best_value})
    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)


# # Step 5: Register the best model using the model 
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # 웹 대시보드에 기록

# mlflow.register_model(
#     model_uri="runs:/9886c35b88ed4acbb447372ba3c49af5/model",
#     name="housing-price-predictor",
# )

# # > Successfully registered model 'housing-price-predictor'.
# # > Created version '1' of model 'housing-price-predictor'.
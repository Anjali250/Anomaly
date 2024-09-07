
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import argparse
import os

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="AnomaData.xlsx")
    parser.add_argument("--test-file", type=str, default="AnomaData.xlsx")

    args, _ = parser.parse_known_args()

    print("SKLearn Version:", sklearn.__version__)
    print("Joblib Version:", joblib.__version__)

    print("[INFO] Reading data")
    train_df = pd.read_excel(os.path.join(args.train, args.train_file))
    test_df = pd.read_excel(os.path.join(args.test, args.test_file))

    X = train_df.drop(['y', 'time'], axis=1)
    y = train_df['y']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Selection and Training
    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=args.random_state), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    # Save the best model
    best_model = grid_search.best_estimator_
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(best_model, model_path)
    print("Model persisted at " + model_path)

    # Predicting and Evaluation
    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print("Total Rows are:", X_test.shape[0])
    print("[TESTING] Model Accuracy is:", test_acc)
    print("[TESTING] Testing Report:")
    print(test_rep)

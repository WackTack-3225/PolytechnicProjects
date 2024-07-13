import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import configparser

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def load_data(file_path):
    print(f"Reading file from {file_path}")
    return pd.read_csv(file_path)

def split_data(data, test_size):
    print("Data Modelling Stage1: train_test_split")
    train, test = train_test_split(data, test_size=test_size)
    y = train['AdoptionSpeed']
    train = train.drop(['AdoptionSpeed'], axis=1)
    y_test = test['AdoptionSpeed']
    test = test.drop(['AdoptionSpeed'], axis=1)
    return train, test, y, y_test

def train_xgboost(train, y, max_depth, n_estimators):
    print("Data Modelling Stage2: train using XGBoost")
    model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(train, y)
    return model

def train_random_forest(train, y, max_depth, n_estimators, random_state):
    print("Data Modelling Stage4: train using RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(train, y)
    return model

def train_logistic_regression(train, y, max_iter):
    print("Data Modelling Stage6: train using Logistic Regression")
    model = LogisticRegression(max_iter=max_iter, multi_class='multinomial', solver='lbfgs')
    model.fit(train, y)
    return model

def evaluate_model(model, test, y_test, model_name):
    print(f"Data Modelling Stage: output {model_name} results")
    predictions = model.predict(test)
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy * 100}%")
    return accuracy

def load_best_model(file_path, test, y_test):
    if os.path.exists(file_path):
        best_model = XGBClassifier()
        best_model.load_model(file_path)
        y_pred_best = best_model.predict(test)
        best_accuracy = accuracy_score(y_test, y_pred_best)
        return best_model, best_accuracy
    else:
        return None, 0

def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def main():
    config = read_config('../../parameter.env')
    file = config.get('PARAMETERS','PROCESSED_FILE')
    max_depth = int(config.get('PARAMETERS','MAX_DEPTH'))
    max_depth_tree = int(config.get('PARAMETERS','MAX_DEPTH_TREE'))
    n_estimators = int(config.get('PARAMETERS','N_ESTIMATORS'))
    n_estimators_tree = int(config.get('PARAMETERS','N_ESTIMATORS_TREE'))
    random_state_tree = int(config.get('PARAMETERS','RANDOM_STATE_TREE'))
    max_iter_lr = int(config.get('PARAMETERS','MAX_ITER_LR'))
    test_size = float(config.get('PARAMETERS', 'TEST_SIZE'))

    pets_data = load_data(file)
    train, test, y, y_test = split_data(pets_data, test_size)

    xgb_model = train_xgboost(train, y, max_depth, n_estimators)
    xgb_accuracy = evaluate_model(xgb_model, test, y_test, "XGBoost")

    rf_model = train_random_forest(train, y, max_depth_tree, n_estimators_tree, random_state_tree)
    rf_accuracy = evaluate_model(rf_model, test, y_test, "RandomForest")

    lr_model = train_logistic_regression(train, y, max_iter_lr)
    lr_accuracy = evaluate_model(lr_model, test, y_test, "Logistic Regression")

    best_xgb_model, best_xgb_accuracy = load_best_model('../../saved_model/saved_model_xgb.json', test, y_test)
    print(f'Current best XGBoost accuracy: {best_xgb_accuracy * 100}%')

    if xgb_accuracy > best_xgb_accuracy:
        xgb_model.save_model('../../saved_model/saved_model_xgb.json')
        print(f'New best XGBoost model saved with accuracy: {xgb_accuracy * 100}%')
    else:
        print(f'XGBoost model not improved, best accuracy remains: {best_xgb_accuracy * 100}%')

    if rf_accuracy > best_xgb_accuracy:
        save_model(rf_model, '../../saved_model/saved_model_rf.pkl')
        print(f'RandomForest model saved with accuracy: {rf_accuracy * 100}%')
    else:
        print(f'RandomForest model not improved, best accuracy remains: {best_xgb_accuracy * 100}%')

    if lr_accuracy > best_xgb_accuracy:
        save_model(lr_model, '../../saved_model/saved_model_lr.pkl')
        print(f'Logistic Regression model saved with accuracy: {lr_accuracy * 100}%')
    else:
        print(f'Logistic Regression model not improved, best accuracy remains: {best_xgb_accuracy * 100}%')

if __name__ == "__main__":
    main()

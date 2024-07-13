try:
    import logging
    from typing import Dict, Tuple
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier 
    from sklearn.metrics import classification_report, accuracy_score, recall_score

except Exception as e:
    print("Error Occured:", e)


def split_data(data: pd.DataFrame, parameters: Dict, 
               parameters2: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """


    X = data.drop(columns=['repurchase'])
    y = data["repurchase"]


    # Split the data into train and test sets using stratified sampling
    (X_train, X_test, 
     y_train, y_test) = train_test_split(X, y, 
                                         test_size=parameters["train_test_size"], 
                                         stratify=y, 
                                         random_state=parameters["random_seed"])


    # Further split the training set into training and validation sets using stratified sampling
    (X_train, X_val, 
     y_train, y_val) = train_test_split(X_train, y_train, 
                                        test_size=parameters["validate_size"], 
                                        stratify=y_train, 
                                        random_state=parameters["random_seed"])
    # 0.25 x 0.8 = 0.2 of the original dataset

    # Selection of the scaling tool
    if parameters2["type"] == "SMOTE":
        data_scalar = SMOTE(random_state=parameters["random_seed"])
    elif parameters2["type"] == "NearMiss":
        data_scalar = NearMiss(random_state=parameters["random_seed"])
    else:
        data_scalar = RandomUnderSampler(random_state=parameters["random_seed"])

    (X_train_resampled, 
     y_train_resampled) = data_scalar.fit_resample(X_train, y_train)

    # Pack to DF
    y_val = y_val.to_frame(name='y_val')
    y_train_resampled = y_train_resampled.to_frame(name='y_train_resampled')
    y_test = y_test.to_frame(name='y_test')

    # Create dummy code for the code to pass through in the correct order
    step = 1

    return X_train_resampled, y_train_resampled, X_test, y_test, X_val, y_val, step


def train_base_model(X_train: pd.DataFrame, y_train: pd.Series,
                    parameter: Dict, step: int) -> BaseEstimator:
    """Trains base model based on selected model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for target variable.

    Returns:
        Trained model.
    """
    # Unpack to series
    y_train = y_train['y_train_resampled']

    # Selection of model
    if parameter["type"] == 'RF':
        classifier = RandomForestClassifier()
    else:
        classifier = XGBClassifier()

    classifier.fit(X_train, y_train)

    return classifier, step

def fine_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                parameter: Dict, parameter2: Dict, step:int) -> BaseEstimator:
    # Unpack back to series
    y_train = y_train['y_train_resampled']
    
    # Selection of model
    if parameter["type"] == 'RF':
        classifier_fine = RandomForestClassifier(
            n_estimators = parameter2["RF_n_estimator"], 
            criterion = parameter2["RF_criterion"], 
            max_depth = parameter2["RF_max_depth"], 
            max_features = parameter2["RF_max_features"]
        )
    else:
        classifier_fine = XGBClassifier()
    classifier_fine.fit(X_train, y_train)

    return classifier_fine, step


def validate_model(classifier: BaseEstimator, 
                   X: pd.DataFrame, y: pd.Series, step) -> Dict:
    """Evaluates the classifier and logs the results.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for target variable.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Unpack back to series
    y = y['y_val']

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info("Model accuracy: %.3f", accuracy)
    logger.info("Classification Report:\n%s", report)
    
    return {"accuracy": accuracy, "classification_report": report}, step

def select_best_classifier(classifier1: BaseEstimator, 
                           classifier2: BaseEstimator, 
                           X: pd.DataFrame, y: pd.Series, 
                           step: int) -> BaseEstimator:
    """Evaluates the classifier and returns the best classifer based on recall.
    """
    y_pred_a = classifier1.predict(X)
    y_pred_b = classifier2.predict(X)

    # Calculate the recall scores
    recall1 = recall_score(y, y_pred_a) 
    recall2 = recall_score(y, y_pred_b)

    # Compare the recall scores and return the better classifier
    if recall1 > recall2:
        better_classifier = classifier1
    else:
        better_classifier = classifier2
    
    return better_classifier, step

def evaluate_model(classifier1: BaseEstimator, 
                   classifier2: BaseEstimator, 
                   X: pd.DataFrame, y: pd.Series, step:int) -> Dict:
    """Evaluates the classifier and logs the results.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for target variable.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Unpack back to series
    y = y['y_test']

    classifier, step = select_best_classifier(classifier1, classifier2, X, y, step)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info("Model accuracy: %.3f", accuracy)
    logger.info("Classification Report:\n%s", report)
    
    step = step

    return {"accuracy": accuracy, 
            "classification_report": report}
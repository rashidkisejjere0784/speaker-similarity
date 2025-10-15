import joblib


# load model
catboost_model = joblib.load('models/cat_model_split_wavlm.joblib')
scaler = joblib.load('models/scaler wavlm.joblib')

def get_prediction(features):
    """
    Get prediction from the CatBoost model.

    Args:
        features (np.ndarray): Input feature vector.

    Returns:
        np.ndarray: Predicted class probabilities.
    """
    # Scale features
    features = scaler.transform(features.reshape(1, -1))
    # Get prediction
    preds = catboost_model.predict_proba(features)
    return preds
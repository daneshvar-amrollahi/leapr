#!/usr/bin/env python3
import logging
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from feature_engine import prepare_supervised_data

logger = logging.getLogger(__name__)


def evaluate_regression_model(model, X, y_true) -> dict[str, float]:
    """Evaluate a regression model."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = math.sqrt(mse)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rho": correlation,
        "y_true": np.array(y_true).tolist(),
        "y_pred": np.array(y_pred).tolist(),
    }


def evaluate_classification_model(model, X, y_true) -> dict[str, float]:
    """Evaluate a classification model."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    if len(set(y_true)) == 2:
        f1 = f1_score(y_true, y_pred, average="binary")  # binary for 2-class problems
    else:
        f1 = None

    return {
        "accuracy": accuracy,
        "f1": f1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def evaluate_multilabel_model(model, X, y_true, option_set_vocab: dict) -> dict[str, float]:
    """Evaluate a multi-label classification model for SMT domain.
    
    For each instance:
    - Predict probabilities for all option sets
    - Select the option set with highest probability (argmax)
    - Check if that option set is in the true label set (any true label = 1)
    
    Args:
        model: Multi-label classifier (MultiOutputClassifier)
        X: Feature matrix
        y_true: Multi-hot encoded labels (n_samples, n_option_sets)
        option_set_vocab: Mapping from option_set string -> index
        
    Returns:
        Dictionary with accuracy and predictions
    """
    # Get probabilities for each option set
    # MultiOutputClassifier.predict_proba returns a list of arrays, one per output
    y_proba_list = model.predict_proba(X)
    
    # Extract probability of class 1 (option set being valid) for each classifier
    # y_proba_list[i] has shape (n_samples, 2) with [:, 1] being prob of class 1
    probs_class_1 = np.array([proba[:, 1] for proba in y_proba_list]).T  # Shape: (n_samples, n_option_sets)
    
    # For each sample, find which option set to select
    idx_to_option = {i: opt for opt, i in option_set_vocab.items()}
    
    correct = 0
    total = y_true.shape[0]
    
    for i in range(total):
        true_labels = y_true[i]  # Binary vector
        prob_vector = probs_class_1[i]  # Probability vector
        
        # Select the option set with highest probability (argmax)
        selected_idx = np.argmax(prob_vector)
        
        # Check if this option is in the true set
        if true_labels[selected_idx] == 1:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Create binary predictions based on argmax selection for logging
    y_pred_binary = np.zeros_like(y_true)
    for i in range(total):
        selected_idx = np.argmax(probs_class_1[i])
        y_pred_binary[i, selected_idx] = 1
    
    return {
        "accuracy": accuracy,
        "f1": None,  # Could compute multi-label F1 if needed
        "y_true": y_true.tolist(),
        "y_pred": y_pred_binary.tolist(),
    }


def prepare_train_valid_split(
    features: list[str],
    train_positions: list,
    valid_positions: list,
    domain_name: str = "chess",
):
    """Prepare training and validation data splits."""
    # For SMT domain, build vocabulary from training set only
    if domain_name == "smt_solver":
        from feature_engine import build_option_set_vocabulary
        
        # Build vocabulary from training data only
        vocab = build_option_set_vocabulary(train_positions)
        
        # Prepare train and valid separately with same vocabulary
        X_train, y_train = prepare_supervised_data(
            features, train_positions, domain_name, option_set_vocab=vocab
        )
        X_valid, y_valid = prepare_supervised_data(
            features, valid_positions, domain_name, option_set_vocab=vocab
        )
        
        return X_train, y_train, X_valid, y_valid
    else:
        # Original behavior for other domains
        X, y = prepare_supervised_data(
            features, train_positions + valid_positions, domain_name
        )
        X_train, y_train = X[: len(train_positions)], y[: len(train_positions)]
        X_valid, y_valid = X[len(train_positions) :], y[len(train_positions) :]
        return X_train, y_train, X_valid, y_valid

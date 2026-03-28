import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_curve

def load_server_assets():
    """Loads the test dataset and the base model."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load test data
    X_test = np.load(os.path.join(base_path, "../data/processed/X_test.npy"))
    y_test = np.load(os.path.join(base_path, "../data/processed/y_test.npy"), allow_pickle=True)
    
    # Convert labels to 0 (Benign) and 1 (Attack)
    is_benign = np.array([str(l).strip().upper() == 'BENIGN' for l in y_test])
    y_true_numeric = np.array([0 if b else 1 for b in is_benign])
    
    # Load base model
    model_path = os.path.join(base_path, "../models/autoencoder.keras")
    model = load_model(model_path)
    
    return model, X_test, y_true_numeric

def evaluate_global_metrics(model, X_test, y_true):
    """Calculates global metrics using the optimal dynamic threshold."""
    recons = model.predict(X_test, batch_size=4096, verbose=0)
    mses = np.mean(np.power(X_test - recons, 2), axis=1)
    
    # Dynamically find the best threshold using PR Curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, mses)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    best_preds = (mses > best_threshold).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, best_preds),
        "F1 Score": f1_scores[best_idx],
        "Precision": precisions[best_idx],
        "Recall": recalls[best_idx]
    }
    
    threats_detected = sum(best_preds)
    return metrics, threats_detected
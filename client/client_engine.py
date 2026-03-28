import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

def load_and_calibrate_client():
    """Loads assets and runs the exact Jupyter PR-Curve math for threshold calibration."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(base_path, "../models/autoencoder.keras"))
    X_train = np.load(os.path.join(base_path, "../data/processed/X_train.npy"))
    X_test = np.load(os.path.join(base_path, "../data/processed/X_test.npy"))
    y_test = np.load(os.path.join(base_path, "../data/processed/y_test.npy"), allow_pickle=True)
    
    is_benign = np.array([str(l).strip().upper() == 'BENIGN' for l in y_test])
    y_true_numeric = np.array([0 if b else 1 for b in is_benign])
    normal_data = X_test[is_benign]
    attack_data = X_test[~is_benign]
    
    print("Calibrating Optimal Threshold (Using Precision-Recall Curve)...")
    metrics, optimal_threshold = evaluate_full_metrics(model, X_test, y_true_numeric)
    print(f"Optimal Threshold Locked: {optimal_threshold:.6f} (Max F1: {metrics['F1 Score']:.4f})")
    
    return model, X_train, X_test, y_true_numeric, normal_data, attack_data, optimal_threshold

def evaluate_full_metrics(model, X_test, y_true):
    """Calculates metrics AND finds the new dynamic tripwire for the updated brain."""
    recons = model.predict(X_test, batch_size=4096, verbose=0)
    mses = np.mean(np.power(X_test - recons, 2), axis=1)
    
    # Recalculate the PR-Curve for the new brain
    precisions, recalls, thresholds = precision_recall_curve(y_true, mses)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    new_threshold = thresholds[best_idx]
    
    preds = (mses > new_threshold).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "F1 Score": f1_scores[best_idx],
        "Precision": precisions[best_idx],
        "Recall": recalls[best_idx]
    }
    return metrics, new_threshold
import json
import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("\n" + "="*70)
    print("[Train Model ERROR] scikit-learn is required for the training phase.")
    print("Please install it by running:")
    print("    pip install scikit-learn")
    print("="*70 + "\n")
    sys.exit(1)

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.json")
    weights_path = os.path.join(os.path.dirname(__file__), "model_weights.json")

    if not os.path.exists(dataset_path):
        print(f"[Train Model ERROR] Dataset file not found at {dataset_path}")
        print("Please run data collection first: python core/collect_data.py")
        return

    # 1. Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    if len(data) < 20:
        print(f"[Train Model ERROR] Dataset has only {len(data)} samples. Please collect at least 100+ total samples.")
        return

    print(f"[Train Model] Loaded dataset containing {len(data)} total samples.")

    # 2. Extract features and labels
    X = []
    y = []
    classes = ["rock", "paper", "scissors", "unknown"]

    for item in data:
        X.append(item["features"])
        y.append(classes.index(item["label"]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Print class distribution
    print("\nClass Distribution:")
    for c_idx, c_name in enumerate(classes):
        count = np.sum(y == c_idx)
        print(f"  {c_name.upper():10} : {count} samples")

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # 4. Initialize and train MLP
    # Hidden layers of (32, 16) provides excellent non-linear classification
    # while running in <0.05ms inside raw NumPy.
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )

    print("Fitting model...")
    mlp.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Extract and save weights
    print("\nExporting weights...")
    W1 = mlp.coefs_[0].tolist()
    b1 = mlp.intercepts_[0].tolist()
    W2 = mlp.coefs_[1].tolist()
    b2 = mlp.intercepts_[1].tolist()
    W3 = mlp.coefs_[2].tolist()
    b3 = mlp.intercepts_[2].tolist()

    export_data = {
        "classes": classes,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    try:
        with open(weights_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"[Train Model] Successfully saved model weights to {weights_path}")
    except Exception as e:
        print(f"[Train Model ERROR] Failed to save weights: {e}")

if __name__ == "__main__":
    main()

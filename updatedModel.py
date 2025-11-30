import preprocess as pp
import numpy as np
import pandas as pd

'''
AI DISCLAIMER NOTE: 
2 prompts which are noted in the code below, carbon usage for this file:

2*4.32g = 8.64g CO2

'''


# ============================================================
# 1. Split Data
# ============================================================

def splitData(df, top_crimes):
    train_mask = df['week_year'] <= 2015
    test_mask = df['week_year'] > 2015

    rollingFeatures = [f'{crime}_rolling_2w' for crime in top_crimes]
    targetFeatures = top_crimes

    X_train = df.loc[train_mask, rollingFeatures].reset_index(drop=True)
    y_train = df.loc[train_mask, targetFeatures].reset_index(drop=True)

    X_test = df.loc[test_mask, rollingFeatures].reset_index(drop=True)
    y_test = df.loc[test_mask, targetFeatures].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

# ============================================================
# 2. Utility Functions
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# ============================================================
# 3. Logistic Regression Training with L1 + L2 + Class Weights
# ============================================================

def train_single_logreg_regularized(X_train, y_train, lr=0.01, n_iter=800, lambda_l1=0.01, lambda_l2=0.01, class_weight=1.0, verbose=False):
    """
    Trains logistic regression with L1 and L2 regularization and class weighting, was GPT assisted.
    """
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0

    y_train = y_train.astype(float)
    weight_vec = np.ones_like(y_train) * class_weight

    for i in range(n_iter):
        linear = np.dot(X_train, weights) + bias
        y_pred = sigmoid(linear)

        error = (y_pred - y_train) * weight_vec

        # Gradients
        dw = np.dot(X_train.T, error) / n_samples
        db = np.mean(error)

        # Regularization
        dw += lambda_l2 * 2 * weights + lambda_l1 * np.sign(weights)

        weights -= lr * dw
        bias -= lr * db

        if verbose and i % 100 == 0:
            loss = binary_cross_entropy(y_train, y_pred)
            reg_loss = lambda_l2 * np.sum(weights**2) + lambda_l1 * np.sum(np.abs(weights))
            print(f"Iter {i}, Loss: {loss:.4f}, Reg: {reg_loss:.4f}")

    return weights, bias

def predict_logreg(X, weights, bias, threshold=0.5):
    proba = sigmoid(np.dot(X, weights) + bias)
    return proba, (proba >= threshold).astype(int)

# ============================================================
# 4. Multi-label Training
# ============================================================

def train_multi_label(X_train, X_test, y_train, y_test, top_crimes,
                      lr=0.01, n_iter=800, lambda_l1=0.01, lambda_l2=0.01):
    """
    Trains logistic regression with L1 and L2 regularization and class weighting, was GPT assisted.
    """
    results = {}
    y_pred_full = pd.DataFrame(index=y_test.index)

    X_train_np = X_train.values
    X_test_np = X_test.values

    weights = {}
    biases = {}

    for crime in top_crimes:
        print(f"Training {crime}...")
        # Compute class weight
        pos_count = y_train[crime].sum()
        neg_count = len(y_train) - pos_count
        if pos_count == 0:
            class_weight = 1.0
        else:
            class_weight = neg_count / pos_count

        w, b = train_single_logreg_regularized(
            X_train_np,
            y_train[crime].values,
            lr=lr,
            n_iter=n_iter,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            class_weight=class_weight,
            verbose=True
        )
        weights[crime] = w
        biases[crime] = b

        # Threshold tuning based on prevalence
        threshold = max(0.05, y_train[crime].mean())
        _, pred_binary = predict_logreg(X_test_np, w, b, threshold=threshold)
        y_pred_full[crime] = pred_binary

        # Accuracy
        acc = (pred_binary == y_test[crime].values).mean()
        results[crime] = acc
        print(f"{crime:40s} Accuracy: {acc:.3f}")

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy'])
    return results_df, y_pred_full

# ============================================================
# 5. Main Execution
# ============================================================

def main():
    df = pp.data.copy()
    top_crimes = pp.top_crimes.copy()

    X_train, y_train, X_test, y_test = splitData(df, top_crimes)

    results_df, y_pred_full = train_multi_label(
        X_train, X_test, y_train, y_test, top_crimes,
        lr=0.05, n_iter=1000, lambda_l1=0.02, lambda_l2=0.02
    )

    # Include identifiers
    y_pred_full['grid_id'] = df.loc[y_test.index, 'grid_id'].values
    y_pred_full['week_year'] = df.loc[y_test.index, 'week_year'].values
    y_pred_full['week_number'] = df.loc[y_test.index, 'week_number'].values

    # Save outputs
    y_pred_full.to_csv("data/predicted_crime_rates.csv", index=False)
    y_test.to_csv("data/ground_truth.csv", index=False)
    print("Saved predictions to data/predicted_crime_rates.csv")

    return y_pred_full, results_df

if __name__ == "__main__":
    predictions, results_df = main()

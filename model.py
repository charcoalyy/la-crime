import preprocess as pp
import numpy as np
import pandas as pd

# ============================================================
# 1. Split Data
# ============================================================

def splitData(df, top_crimes):
    train_mask = df['week_year'] <= 2015
    test_mask = df['week_year'] > 2015

    rollingFeatures = [f'{crime}_rolling_2w' for crime in top_crimes]
    targetFeatures = top_crimes

    missingFeatures = [c for c in rollingFeatures if c not in df.columns]
    missingTargets = [c for c in targetFeatures if c not in df.columns]
    if missingFeatures:
        raise ValueError(f"Missing rolling features in data: {missingFeatures}")
    if missingTargets:
        raise ValueError(f"Missing target features in data: {missingTargets}")

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
# 3. Logistic Regression Training and Prediction
# ============================================================

def logisticTrainingFitSingleLabel(X_train, y_train, learningRate, iterations, verbose=False):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0

    for i in range(iterations):
        linear_model = np.dot(X_train, weights) + bias
        y_predicted = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
        db = (1 / n_samples) * np.sum(y_predicted - y_train)

        weights -= learningRate * dw
        bias -= learningRate * db

        if verbose and i % 100 == 0:
            loss = binary_cross_entropy(y_train, y_predicted)
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return weights, bias

def predictBinaryLogistic(X, weights, bias, threshold=0.5):
    y_pred_proba = sigmoid(np.dot(X, weights) + bias)
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    return y_pred_proba, y_pred_binary

def logisticTrainingFitMultiLabel(X_train, y_train, learningRate, iterations, verbose=False):
    weights = {}
    biases = {}
    for label in y_train.columns:
        if verbose:
            print(f"Training label: {label}")
        weights[label], biases[label] = logisticTrainingFitSingleLabel(
            X_train, y_train[label], learningRate, iterations, verbose
        )
    return weights, biases

def predictMultiLabelLogistic(X, weights, biases, threshold=0.5):
    y_pred_proba = pd.DataFrame()
    y_pred_binary = pd.DataFrame()
    for label in weights.keys():
        proba, binary = predictBinaryLogistic(X, weights[label], biases[label], threshold)
        y_pred_proba[label] = proba
        y_pred_binary[label] = binary
    return y_pred_proba, y_pred_binary

# ============================================================
# 4. Train and Evaluate Multi-label Logistic Regression
# ============================================================

def trainMultiLabelLogisticRegression(X_train, X_test, y_train, y_test, top_crimes, lr=0.1, n_iter=800):
    labelsUnique = y_train.nunique()
    trainableLabels = labelsUnique[labelsUnique > 1].index.tolist()
    constantLabels = labelsUnique[labelsUnique <= 1].index.tolist()

    if not trainableLabels:
        print("No trainable labels found (all constant).")
        return None, None, constantLabels

    X_train_np = X_train.values
    X_test_np = X_test.values

    weights, biases = logisticTrainingFitMultiLabel(
        X_train_np, y_train[trainableLabels], learningRate=lr, iterations=n_iter, verbose=True
    )

    y_pred_proba, y_pred_binary = predictMultiLabelLogistic(X_test_np, weights, biases, threshold=0.5)

    # Rebuild full prediction including constant labels
    y_pred_full = pd.DataFrame(index=y_test.index)
    for col in top_crimes:
        if col in y_pred_binary.columns:
            y_pred_full[col] = y_pred_binary[col]
        elif col in constantLabels:
            y_pred_full[col] = int(y_train[col].iloc[0])
        else:
            y_pred_full[col] = 0

    # Simple accuracy per label
    results = {}
    for col in top_crimes:
        y_true = y_test[col].values
        y_pred = y_pred_full[col].values
        acc = (y_true == y_pred).mean()
        results[col] = acc
        print(f"{col:40s} Accuracy: {acc:.3f}")

    return (weights, biases), pd.DataFrame.from_dict(results, orient='index', columns=['accuracy']), y_pred_full

# ============================================================
# 5. Main Execution
# ============================================================

def main():
    df = pp.data.copy()
    top_crimes = pp.top_crimes.copy()

    # Split train/test
    X_train, y_train, X_test, y_test = splitData(df, top_crimes)

    # Train and predict
    (weights, biases), results_df, y_pred_full = trainMultiLabelLogisticRegression(
        X_train, X_test, y_train, y_test, top_crimes, lr=0.1, n_iter=800
    )

    # Include identifiers for reference
    y_pred_full['grid_id'] = df.loc[y_test.index, 'grid_id'].values
    y_pred_full['week_year'] = df.loc[y_test.index, 'week_year'].values
    y_pred_full['week_number'] = df.loc[y_test.index, 'week_number'].values

    # Save predictions for later visualization
    y_pred_full.to_csv("data/predicted_crime_rates.csv", index=False)
    y_test.to_csv("data/ground_truth.csv", index=False)
    print("Predicted crime rates saved to data/predicted_crime_rates.csv")

    return y_pred_full, results_df

# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":
    predictions, results_df = main()
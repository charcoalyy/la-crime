import preprocess as pp
import numpy as np
import pandas as pd

'''
AI DISCLAIMER NOTE (ChatGPT Model 5): 
11 prompts which are noted in the code below, carbon usage for this file:

11*4.32g = 47.52g CO2
'''

# ============================================================
# 1. Split Data
# ============================================================

def splitData(df, top_crimes):
    """
    Splits the dataframe into training and testing sets based on week_year. 
    
    3 prompts used: 

    Prompted ChatGPT to combine explicit feature selection and automatic numeric column inclusion 
    into a single feature set for model training.

    Prompted ChatGPT to implement one-hot encoding for the 'season' categorical variable,
    ensuring the first category is dropped to avoid multicollinearity.

    Prompted ChatGPT to create X_train, y_train, X_test, y_test splits based on week_year <= 2015 for training and > 2015 for testing.

    """

    #One-hot encode season
    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    seasonFeatures = [col for col in df.columns if col.startswith('season_')]

    train_mask = df['week_year'] <= 2015
    test_mask = df['week_year'] > 2015

    rollingFeatures = [f'{crime}_rolling_2w' for crime in top_crimes]
    crimeCountFeatures = top_crimes
    otherFeatures = ['neigh_activity_score', 'is_holiday', 'week_number']

    excluded = set(top_crimes + ['grid_id', 'week_year', 'week_number'])
    auto_numeric = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded]

    feature_set = list(set(rollingFeatures + crimeCountFeatures + otherFeatures + seasonFeatures + auto_numeric))

    X_train = df.loc[train_mask, feature_set].reset_index(drop=True)
    X_test  = df.loc[test_mask, feature_set].reset_index(drop=True)
    y_train = df.loc[train_mask, top_crimes].reset_index(drop=True)
    y_test  = df.loc[test_mask, top_crimes].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

# ============================================================
# 2. Utility Functions
# ============================================================

def sigmoid(x):
    """
    Overflow-safe sigmoid that handles Python floats and numpy arrays, GPT

    1 prompt used:

    Prompted ChatGPT to implement an overflow-safe sigmoid function that can handle both Python floats and numpy arrays.
    
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out

def binary_cross_entropy(y, y_pred):
    """
    Binary cross-entropy loss function.

    1 prompt used:

    Prompted ChatGPT to implement a binary cross-entropy loss function that safely handles edge cases to avoid log(0) errors.
    """
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


# ============================================================
# 3. Logistic Regression Training with L1 + L2 + Class Weights
# ============================================================

def train_single_logreg_regularized(X_train, y_train, lr=0.01, n_iter=800, lambda_l1=0.01, lambda_l2=0.01, class_weight=1.0, verbose=False):
    """
    Trains logistic regression with L1 and L2 regularization and class weighting.

    2 prompts used:

    Prompted ChatGPT to implement logistic regression training with both L1 and L2 regularization, including class weighting to handle imbalanced datasets.

    Prompted ChatGPT to format print statements for loss and regularization values during training for better readability.
    """
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    weight_vec = np.ones_like(y_train) * class_weight

    for i in range(n_iter):
        linear = np.dot(X_train, weights) + bias
        y_pred = sigmoid(linear)

        error = (y_pred - y_train) * weight_vec

        # Gradients
        dw = np.dot(X_train.T, error) / n_samples
        db = float(np.mean(error))

        # Regularization
        dw += lambda_l2 * 2 * weights + lambda_l1 * np.sign(weights)

        # Gradient update
        weights -= lr * dw
        bias -= lr * db

        if verbose and i % 100 == 0:
            loss = binary_cross_entropy(y_train, y_pred)
            reg_loss = lambda_l2 * np.sum(weights**2) + lambda_l1 * np.sum(np.abs(weights))
            print(f"Iter {i}, Loss: {loss:.4f}, Reg: {reg_loss:.4f}")

    return weights, bias

def predict_logreg(X, weights, bias, threshold=0.5):
    X = np.asarray(X, dtype=float)
    linear = np.dot(X, weights) + bias
    proba = sigmoid(linear)
    return proba, (proba >= threshold).astype(int)


# ============================================================
# 4. Multi-label Training
# ============================================================

def train_multi_label(X_train, X_test, y_train, y_test, top_crimes, lr=0.01, n_iter=800, lambda_l1=0.01, lambda_l2=0.01):
    
    """
    Trains multi-label logistic regression models for each crime type with regularization and class weighting.

    3 prompts used:

    Prompted ChatGPT to implement multi-label logistic regression training, iterating over each crime type and applying class weighting based on label prevalence.

    Prompted ChatGPT to include threshold tuning based on label prevalence for each crime type during prediction.

    Prompted ChatGPT to calculate and report accuracy for each crime type after predictions are made.
    """

    results = {}
    y_pred_full = pd.DataFrame(index=y_test.index)

    X_train_np = np.asarray(X_train, dtype=float)
    X_test_np = np.asarray(X_test, dtype=float)

    weights = {}
    biases = {}

    for crime in top_crimes:
        print(f"Training {crime}...")

        pos_count = y_train[crime].sum()
        neg_count = len(y_train) - pos_count
        if pos_count == 0 or neg_count == 0:
            print(f"Skipping {crime} due to zero variance.")
            weights[crime] = np.zeros(X_train_np.shape[1])
            biases[crime] = 0.0
            y_pred_full[crime] = np.zeros(len(y_test), dtype=int)
            results[crime] = 1.0 if pos_count == 0 else 0.0
            continue

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
    """
    Main execution function.

    1 prompt used:
    Prompted ChatGPT to create a main function that orchestrates data loading, model training, prediction, and saving of results.
    """
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

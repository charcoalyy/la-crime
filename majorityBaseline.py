import preprocess as pp
import numpy as np
import pandas as pd

'''
AI DISCLAIMER NOTE: 
2 prompts which are noted in the code below, carbon usage for this file:

2*4.32g = 8.64g CO2

'''



# ============================================================================
# 1. Majority Baseline (per-label)
# ============================================================================

def train_majority_baseline(X_train, y_train, top_crimes):
    """
    Computes the majority class (0 or 1) for each crime label, was GPT assisted.
    """
    majority_class = {}

    for crime in top_crimes:
        # If more than 50% of the values are 1 → predict 1 always
        majority_label = int(y_train[crime].mean() >= 0.5)
        majority_class[crime] = majority_label

    return majority_class


def predict_majority_baseline(X, majority_class, top_crimes):
    """
    Returns a DataFrame of binary predictions with shape (n_samples, n_labels), was GPT assisted.
    """
    n_samples = X.shape[0]
    preds = pd.DataFrame(index=range(n_samples))

    for crime in top_crimes:
        preds[crime] = [majority_class[crime]] * n_samples

    return preds


# ============================================================================
# 2. Data Split (unchanged from your model.py)
# ============================================================================

def splitData(df, top_crimes):
    train_mask = df["week_year"] <= 2015
    test_mask = df["week_year"] > 2015

    rollingFeatures = [f"{crime}_rolling_2w" for crime in top_crimes]

    X_train = df.loc[train_mask, rollingFeatures].reset_index(drop=True)
    y_train = df.loc[train_mask, top_crimes].reset_index(drop=True)

    X_test = df.loc[test_mask, rollingFeatures].reset_index(drop=True)
    y_test = df.loc[test_mask, top_crimes].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


# ============================================================================
# 3. Main Execution
# ============================================================================

def main():
    df = pp.data.copy()
    top_crimes = pp.top_crimes.copy()

    # Split data exactly the same way as the original model
    X_train, y_train, X_test, y_test = splitData(df, top_crimes)

    # ===== Train majority baseline =====
    majority_class = train_majority_baseline(X_train, y_train, top_crimes)

    # ===== Predict on test set =====
    y_pred_full = predict_majority_baseline(X_test, majority_class, top_crimes)

    # ===== Compute accuracy per label =====
    results = {}
    for crime in top_crimes:
        acc = (y_pred_full[crime].values == y_test[crime].values).mean()
        results[crime] = acc
        print(f"{crime:40s} Accuracy: {acc:.3f}")

    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["accuracy"])

    # ===== Add identifiers to predictions (same format as model.py) =====
    y_pred_full["grid_id"] = df.loc[y_test.index, "grid_id"].values
    y_pred_full["week_year"] = df.loc[y_test.index, "week_year"].values
    y_pred_full["week_number"] = df.loc[y_test.index, "week_number"].values

    # ===== Save results =====
    y_pred_full.to_csv("data/predicted_crime_rates.csv", index=False)
    y_test.to_csv("data/ground_truth.csv", index=False)

    print("\nSaved predictions to data/predicted_crime_rates.csv")

    return y_pred_full, results_df


if __name__ == "__main__":
    predictions, results_df = main()

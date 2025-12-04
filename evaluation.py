import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

'''
AI DISCLAIMER NOTE (ChatGPT Model 5):
6 prompts which are noted in the code below, carbon usage for this file:

6 * 4.32g = 25.92g CO2
'''

# ============================
# Helper functions
# ============================

def load_data():
    """
    1 prompt used:
    Asked ChatGPT to design a helper for loading GT/pred CSVs and auto-detecting top crime labels.
    """
    y_true = pd.read_csv("data/ground_truth.csv")
    y_pred = pd.read_csv("data/predicted_crime_rates.csv")
    top_crimes = [c for c in y_true.columns if c not in ['grid_id', 'week_year', 'week_number']]
    return y_true, y_pred, top_crimes


# ============================
# Error analysis functions
# ============================

def evaluate_multilabel(y_true, y_pred, top_crimes):
    '''
    1 prompt used:
    Prompted ChatGPT to compute precision/recall/F1/accuracy per label and aggregate a macro-average row.
    '''
    results = {}
    for label in top_crimes:
        if label in y_true.columns and label in y_pred.columns:
            precision = precision_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            recall = recall_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            f1 = f1_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            acc = accuracy_score(y_true[label], y_pred[label])
            results[label] = [precision, recall, f1, acc]

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['precision', 'recall', 'f1', 'accuracy'])
    results_df['macro_avg'] = results_df.mean(axis=1)
    results_df.loc['Overall'] = results_df.mean(numeric_only=True)
    return results_df


def confusion_matrices(y_true, y_pred, labels):
    """
    1 prompt used:
    Used ChatGPT to learn how to decompose sklearn confusion_matrix into TN/FP/FN/TP ratios.
    """
    matrices = {}
    
    for label in labels:
        cm = confusion_matrix(y_true[label], y_pred[label], labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        matrices[label] = {
            'TN': tn / total,
            'FP': fp / total,
            'FN': fn / total,
            'TP': tp / total
        }
    
    df_matrices = pd.DataFrame.from_dict(matrices, orient='index')
    return df_matrices


def cooccurrence_errors(y_true, y_pred, labels):
    """
    1 prompt used:
    ChatGPT suggested using dot-products of error matrices to compute co-misprediction counts.
    """
    errors = (y_true[labels] != y_pred[labels]).astype(int)
    co_error_matrix = errors.T.dot(errors)
    np.fill_diagonal(co_error_matrix.values, 0)
    return co_error_matrix


def high_error_samples(y_true, y_pred, labels, top_n=20):
    """
    1 prompt used:
    Asked ChatGPT to rank rows by number of mispredicted labels and return top-N.
    """
    errors = (y_true[labels] != y_pred[labels]).astype(int)
    errors['num_errors'] = errors.sum(axis=1)
    high_err_df = errors.sort_values('num_errors', ascending=False).head(top_n)
    return high_err_df


# ============================
# Main
# ============================

def main():
    """
    1 prompt used:
    ChatGPT assisted in structuring a clean main() workflow saving all metrics to CSV.
    """
    y_true, y_pred, top_crimes = load_data()
    results_df = evaluate_multilabel(y_true, y_pred, top_crimes)

    print("\n========== Evaluation Metrics ==========\n")
    print(results_df.round(3))
    results_df.to_csv("data/error_analysis/evaluation_metrics.csv", index=True)

    print("\n========== Confusion Matrices ==========\n")
    conf_df = confusion_matrices(y_true, y_pred, top_crimes)
    print(conf_df)
    conf_df.to_csv("data/error_analysis/confusion_matrices.csv", index=True)

    print("\n========== Co-occurrence Errors ==========\n")
    co_errors_df = cooccurrence_errors(y_true, y_pred, top_crimes)
    print(co_errors_df)
    co_errors_df.to_csv("data/error_analysis/cooccurrence_errors.csv", index=True)

    print("\n========== High Error Samples ==========\n")
    high_err_df = high_error_samples(y_true, y_pred, top_crimes)
    print(high_err_df)
    high_err_df.to_csv("data/error_analysis/high_error_samples.csv", index=True)

if __name__ == "__main__":
    main()
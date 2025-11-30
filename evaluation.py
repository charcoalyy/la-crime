import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

'''
AI DISCLAIMER NOTE: 
5 prompt which is noted in the code below, carbon usage for this file:

5*4.32g = 21.6g CO2

'''

# ============================
# Helper functions
# ============================

def load_data():
    y_true = pd.read_csv("data/ground_truth.csv")
    y_pred = pd.read_csv("data/predicted_crime_rates.csv")
    top_crimes = [c for c in y_true.columns if c not in ['grid_id', 'week_year', 'week_number']]
    return y_true, y_pred, top_crimes

# ============================
# Error analysis functions
# ============================

def evaluate_multilabel(y_true, y_pred, top_crimes):
    '''
    ChatGPT assistance was used to calculate evaluation metrics for multilabel classification.
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
    returns dictionary of confusion matrices (key = each class, value is TN/FP/FN/TP 2x2 array)
    """
    matrices = {}
    
    for label in labels:
        ''' AI: Learned how to use confusion_matrix using ChatGPT assistance '''
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
    """returns df showing how often 2 labels are both mispredicted in the same row"""
    
    errors = (y_true[labels] != y_pred[labels]).astype(int)
    co_error_matrix = errors.T.dot(errors)
    np.fill_diagonal(co_error_matrix.values, 0)
    return co_error_matrix

def high_error_samples(y_true, y_pred, labels, top_n=20):
    """returns the top_n rows with the most # of mispredicted labels"""
    
    errors = (y_true[labels] != y_pred[labels]).astype(int)
    errors['num_errors'] = errors.sum(axis=1)
    high_err_df = errors.sort_values('num_errors', ascending=False).head(top_n)
    return high_err_df

# ============================
# Main
# ============================

def main():
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
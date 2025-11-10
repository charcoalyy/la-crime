import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_multilabel(y_true, y_pred, top_crimes):
    results = {}
    for label in top_crimes:
        if label in y_true.columns and label in y_pred.columns:
            yt = y_true[label].values
            yp = y_pred[label].values
            precision = precision_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            recall = recall_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            f1 = f1_score(y_true[label], y_pred[label], average='macro', zero_division=0)
            acc = accuracy_score(y_true[label], y_pred[label])
            results[label] = [precision, recall, f1, acc]
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['precision', 'recall', 'f1', 'accuracy'])
    results_df['macro_avg'] = results_df.mean(axis=1)
    results_df.loc['Overall'] = results_df.mean(numeric_only=True)
    return results_df

def load_data():
    y_true = pd.read_csv("data/ground_truth.csv")
    y_pred = pd.read_csv("data/predicted_crime_rates.csv")
    top_crimes = [c for c in y_true.columns if c not in ['grid_id', 'week_year', 'week_number']]
    return y_true, y_pred, top_crimes

def main():
    y_true, y_pred, top_crimes = load_data()
    results_df = evaluate_multilabel(y_true, y_pred, top_crimes)
    print("\nEvaluation Results:\n")
    print(results_df.round(3))
    results_df.to_csv("data/evaluation_results.csv", index=True)
    print("\nSaved to data/evaluation_results.csv")

if __name__ == "__main__":
    main()
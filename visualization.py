import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

'''
AI DISCLAIMER NOTE: 
7 prompts which are noted in the code below, carbon usage for this file:

7*4.32g = 30.24g CO2

'''

# ============================
# Helper functions
# ============================

def get_paths(filename, base_dir="data/error_analysis"):
    ''' AI: File path helper function created using ChatGPT assistance '''
    """
    example:
        filename = "evaluation_metrics"
        returns ("data/error_analysis/evaluation_metrics.csv",
                 "data/error_analysis/plots/evaluation_metrics.png")
    """
    csv_path = os.path.join(base_dir, f"{filename}.csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{filename}.png")
    return csv_path, plot_path

# ============================
# Plotting Functions
# ============================

def plot_evaluation_metrics(filename="evaluation_metrics"):
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop("Overall", errors='ignore')
    
    ''' AI: Plot created using ChatGPT assistance '''
    plt.figure()
    df[['precision','recall','f1','accuracy']].plot(kind='bar', figsize=(14,7))

    plt.title("Evaluation Metrics per Crime Type")
    plt.ylabel("Score")
    plt.xlabel("Crime Type")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric")
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()

def plot_confusion_matrices(filename="confusion_matrices"):
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop("Overall", errors='ignore')
    
    ''' AI: Plot transformation created using ChatGPT assistance '''
    n = len(df)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, (label, row) in enumerate(df.iterrows()):
        cm = np.array([[row['TN'], row['FP']],
                       [row['FN'], row['TP']]])
        im = axes[i].imshow(cm, cmap='Blues', vmin=0, vmax=1)
        
        ''' AI: Cells annotated using ChatGPT assistance '''
        annotations = [['TN', 'FP'], ['FN', 'TP']]
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{annotations[r][c]}\n{cm[r,c]:.2f}", 
                             ha='center', va='center', color='red', fontsize=10)
        
        axes[i].set_title(label, fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.tight_layout()
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04)
    plt.savefig(plot_path)
    plt.close()

def plot_cooccurrence_errors(filename="cooccurrence_errors"):
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0)
    
    # normalized for plot readability
    df_norm = df.div(df.sum(axis=1), axis=0) * 100
    
    ''' AI: Plot created using ChatGPT assistance '''
    plt.figure(figsize=(16,12))
    sns.heatmap(
        df_norm,
        annot=True,
        fmt=".1f",
        cmap='Reds',
        cbar_kws={'label':'Co-occurrence (%)'},
        annot_kws={'fontsize':8}
    )
    
    plt.title("Co-occurrence of Mispredicted Crimes (Row-Normalized %)")
    plt.ylabel("Crime Type 1")
    plt.xlabel("Crime Type 2")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()

# ============================
# Main
# ============================

def main():
    plot_evaluation_metrics()
    plot_confusion_matrices()
    plot_cooccurrence_errors()

if __name__ == "__main__":
    main()

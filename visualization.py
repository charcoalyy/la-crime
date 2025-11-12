import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
AI DISCLAIMER NOTE: 
2 prompts which are noted in the code below, carbon usage for this file:

2*4.32g = 8.64g CO2

'''

def load_results():
    df = pd.read_csv("data/evaluation_results.csv", index_col=0)
    df = df.drop("Overall", errors='ignore')
    return df

def plot_bar_metrics(df):
    '''
    Plots a bar chart of the performance metrics for each crime type, ChatGPT was prompted to generate this.
    '''
    plt.figure(figsize=(12,6))
    df[['precision','recall','f1','accuracy']].plot(kind='bar', figsize=(14,7))
    plt.title("Crime Type Performance Metrics")
    plt.ylabel("Score")
    plt.xlabel("Crime Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("data/performance_bar_chart.png")
    plt.close()

def plot_heatmap(df): 
    '''
    Plots a heatmap of the performance metrics for each crime type, ChatGPT was prompted to generate this.
    '''
    plt.figure(figsize=(8,6))
    sns.heatmap(df[['precision','recall','f1','accuracy']], annot=True, cmap='Blues', fmt=".2f")
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("data/performance_heatmap.png")
    plt.close()

def main():
    df = load_results()
    plot_bar_metrics(df)
    plot_heatmap(df)
    print("Charts saved to data/performance_bar_chart.png and data/performance_heatmap.png")

if __name__ == "__main__":
    main()
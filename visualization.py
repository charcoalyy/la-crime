import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def load_predictions():
    df = pd.read_csv("data/predicted_crime_rates.csv")
    lat = df["grid_id"].str.extract(r"lat([0-9]+.[0-9]+)")[0].astype(float)
    lon = df["grid_id"].str.extract(r"lon(-[0-9]+.[0-9]+)")[0].astype(float)
    df["lat"] = lat
    df["lon"] = lon
    return df

def visualize_spatial_distribution(df):
    os.makedirs("data/maps", exist_ok=True)
    crime_cols = [c for c in df.columns if c not in ["grid_id", "week_year", "week_number", "lat", "lon"]]
    for crime in crime_cols:
        subset = df[df[crime] > 0]
        if subset.empty:
            continue
        plt.figure(figsize=(6,5))
        sns.kdeplot(
            data=subset,
            x="lon",
            y="lat",
            fill=True,
            cmap="Reds",
            thresh=0.05,
            levels=100
        )
        plt.scatter(subset["lon"], subset["lat"], s=5, color="black", alpha=0.3)
        plt.title(f"Crime Density Map — {crime}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(f"data/maps/{crime.replace('/', '').replace(' ', '_')}.png", dpi=150)
        plt.close()

def main():
    df = load_predictions()
    visualize_spatial_distribution(df)
    print("Crime maps saved to data/maps/")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("[visualization.py] geopandas / shapely not installed. Map overlays disabled.")

'''
AI DISCLAIMER NOTE (ChatGPT Model 5):
15 prompts which are noted in the code below, carbon usage for this file:

15 * 4.32g = 64.8g CO2
'''

# ============================
# Helper functions
# ============================

def get_paths(filename, base_dir="data/error_analysis"):
    """
    1 prompt used:
    Asked ChatGPT to create a helper function that constructs both CSV and PNG save paths 
    and auto-creates the plots directory.
    """
    csv_path = os.path.join(base_dir, f"{filename}.csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{filename}.png")
    return csv_path, plot_path


def get_la_shapefile(path="data/maps/la_shapefile/Neighborhood_Councils_(Certified).shp"):
    """
    1 prompt used:
    Requested ChatGPT to implement a robust shapefile loader including:
    - geopandas availability check
    - file existence check
    - clear error messages
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("Install geopandas + shapely for LA map overlays.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shapefile not found at {path}")
    return gpd.read_file(path)


def parse_grid_lat_lon(series):
    """
    1 prompt used:
    Asked ChatGPT to write a regex-based parser that extracts lat/lon from grid_id strings.
    """
    coords = series.str.extract(r"grid_lat(?P<lat>-?\d+\.\d+)_lon(?P<lon>-?\d+\.\d+)")
    coords["lat"] = coords["lat"].astype(float)
    coords["lon"] = coords["lon"].astype(float)
    return coords


def grid_points_geodf(df, crs="EPSG:4326"):
    """
    1 prompt used:
    ChatGPT assisted in converting latitude/longitude columns into a GeoDataFrame using shapely points.
    """
    coords = parse_grid_lat_lon(df["grid_id"])
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(coords["lon"], coords["lat"]),
        crs=crs,
    )


def get_crime_columns(df):
    """
    1 prompt used:
    Asked ChatGPT to generate helper function filtering out identifier columns.
    """
    exclude = {"grid_id", "week_year", "week_number"}
    return [c for c in df.columns if c not in exclude]


# ============================
# Error-analysis plotting
# ============================

def plot_evaluation_metrics(filename="evaluation_metrics"):
    """
    1 prompt used:
    Assistance requested for designing a bar-plot layout for precision/recall/F1/accuracy.
    """
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0).drop("Overall", errors="ignore")
    fig, ax = plt.subplots(figsize=(14, 7))
    df[["precision", "recall", "f1", "accuracy"]].plot(kind="bar", ax=ax)
    ax.set_title("Evaluation Metrics per Crime Type")
    ax.set_ylabel("Score")
    ax.set_xlabel("Crime Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def plot_confusion_matrices(filename="confusion_matrices"):
    """
    2 prompts used:
    - Prompted ChatGPT for constructing a multi-row/multi-column subplot grid based on #labels.
    - Prompted ChatGPT to overlay TN/FP/FN/TP text directly onto heatmap squares.
    """
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0).drop("Overall", errors="ignore")
    n = len(df)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, (label, row) in enumerate(df.iterrows()):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        im = axes[i].imshow(cm, cmap="Blues", vmin=0, vmax=1)

        labels = [["TN","FP"],["FN","TP"]]
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{labels[r][c]}\n{cm[r,c]:.2f}",
                             ha="center", va="center", color="red", fontsize=10)

        axes[i].set_title(label, fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04)
    plt.savefig(plot_path)
    plt.close(fig)


def plot_cooccurrence_errors(filename="cooccurrence_errors"):
    """
    1 prompt used:
    ChatGPT suggested normalizing rows by percentage and formatting a heatmap with annotation.
    """
    csv_path, plot_path = get_paths(filename)
    df = pd.read_csv(csv_path, index_col=0)
    df_norm = df.div(df.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        df_norm,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar_kws={"label": "Co-occurrence (%)"},
        annot_kws={"fontsize": 8}
    )
    plt.title("Co-occurrence of Mispredicted Crimes (Row-Normalized %)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# ============================
# Spatio-temporal LA map overlays
# ============================

def plot_weekly_la_map(
    week_year,
    week_number,
    crime_type=None,
    use_ground_truth=False,
    shapefile_path="data/maps/la_shapefile/Neighborhood_Councils_(Certified).shp",
    output_name=None,
):
    """
    3 prompts used:
    - Prompted ChatGPT to write logic for choosing GT vs predicted CSV.
    - Prompted ChatGPT to compute intensity values (single crime vs all crimes).
    - Prompted ChatGPT to scale point sizes based on relative intensity.
    """

    if not GEOPANDAS_AVAILABLE:
        return

    pred_path = "data/predicted_crime_rates.csv"
    true_path = "data/ground_truth.csv"
    df = pd.read_csv(true_path if use_ground_truth else pred_path)

    mask = (df["week_year"] == week_year) & (df["week_number"] == week_number)
    week_df = df.loc[mask].copy()
    if week_df.empty:
        print("No data for week.")
        return

    crime_cols = get_crime_columns(week_df)
    if crime_type is not None and crime_type in week_df.columns:
        week_df["intensity"] = week_df[crime_type]
        label = crime_type
    else:
        week_df["intensity"] = week_df[crime_cols].sum(axis=1)
        label = "All Crimes"

    gdf = grid_points_geodf(week_df)
    la = get_la_shapefile(shapefile_path)

    if la.crs != gdf.crs:
        """
        1 prompt used:
        Asked ChatGPT how to safely convert GeoDataFrame coordinate systems.
        """
        gdf = gdf.to_crs(la.crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    la.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.4)

    max_intensity = gdf["intensity"].max()
    size = 20 if max_intensity == 0 else 20 + 60*(gdf["intensity"]/max_intensity)

    gdf.plot(
        ax=ax,
        column="intensity",
        cmap="Reds",
        markersize=size,
        alpha=0.8,
        legend=True,
        legend_kwds={"label": label},
    )

    prefix = "Ground Truth" if use_ground_truth else "Predicted"
    ax.set_title(f"{prefix} Hotspots: {label}\nWeek {week_number}, {week_year}")
    ax.set_axis_off()

    out_dir = "data/maps/plots"
    os.makedirs(out_dir, exist_ok=True)
    if output_name is None:
        """
        1 prompt used:
        Asked ChatGPT to design automatic output filename formatting.
        """
        crime_name = crime_type.replace(" ", "_") if crime_type else "all_crimes"
        base = "gt" if use_ground_truth else "pred"
        output_name = f"{base}_{week_year}_{week_number}_{crime_name}"

    out_path = os.path.join(out_dir, f"{output_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved map: {out_path}")


# ============================
# Main
# ============================

def main():
    """
    1 prompt used:
    ChatGPT assisted in designing a clean orchestrator function calling all plots.
    """
    plot_evaluation_metrics()
    plot_confusion_matrices()
    plot_cooccurrence_errors()

    try:
        plot_weekly_la_map(week_year=2016, week_number=10, crime_type=None)
    except Exception as e:
        print(f"Map plotting skipped: {e}")

if __name__ == "__main__":
    main()
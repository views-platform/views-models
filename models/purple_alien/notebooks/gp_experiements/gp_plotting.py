# gp_plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_single_series(df: pd.DataFrame, series_id: int):
    series_df = df[df["series_id"] == series_id]

    plt.figure(figsize=(12, 6))
    plt.plot(series_df["timestep"], series_df["value"], label="Value", linewidth=2)
    plt.plot(series_df["timestep"], series_df["long_term"], label="Long-Term Trend", linestyle="--")
    plt.plot(series_df["timestep"], series_df["short_term"], label="Short-Term Trend", linestyle="--")
    plt.plot(series_df["timestep"], series_df["seasonal"], label="Seasonal", linestyle="--")
    plt.plot(series_df["timestep"], series_df["noise"], label="Noise", linestyle=":", alpha=0.7)
    
    plt.title(f"Time Series Decomposition (Series ID = {series_id})")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_series(df: pd.DataFrame, n: int = 4):
    unique_ids = df["series_id"].unique()[:n]
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    
    for i, series_id in enumerate(unique_ids):
        series_df = df[df["series_id"] == series_id]
        axes[i].plot(series_df["timestep"], series_df["value"], label=f"Series {series_id}")
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()


def set_plot_style(style="whitegrid"):
    sns.set(style=style)


def plot_signal_vs_noise_comparison(df: pd.DataFrame):
    """
    Compares a high-signal and low-signal time series using all decomposed components.
    """
    high_ids = df[df["signal_strength"] == "high"]["series_id"].unique()
    low_ids = df[df["signal_strength"] == "low"]["series_id"].unique()

    if len(high_ids) == 0 or len(low_ids) == 0:
        print("Not enough data to plot both high and low signal series.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for idx, (label, sid) in enumerate([("High", high_ids[0]), ("Low", low_ids[0])]):
        subdf = df[df["series_id"] == sid]
        axes[idx].plot(subdf["timestep"], subdf["value"], label="Value", linewidth=2)
        axes[idx].plot(subdf["timestep"], subdf["long_term"], label="Long-Term", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["short_term"], label="Short-Term", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["seasonal"], label="Seasonal", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["noise"], label="Noise", linestyle=":", alpha=0.6)
        axes[idx].set_title(f"{label}-Signal Series (ID: {sid})")
        axes[idx].legend(loc="upper right")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()

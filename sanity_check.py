import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

def plot_scenario(scenario_id: str, data_dir: Path):
    df = pd.read_csv(data_dir / f"{scenario_id}.csv")
    with open(data_dir / f"{scenario_id}_metadata.json") as f:
        meta = json.load(f)

    df["date"] = pd.to_datetime(df["date"])
    ground_truth = meta["ground_truth"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top: signups per channel
    for ch in df["channel"].unique():
        sub = df[df["channel"] == ch]
        axes[0].plot(sub["date"], sub["signups"], label=ch, marker="o", markersize=2)
    axes[0].set_ylabel("Signups")
    axes[0].legend()
    axes[0].set_title(f"{scenario_id}: {ground_truth['incident_type']} "
                       f"({ground_truth['affected_channel'] or 'all channels'})")

    # Bottom: conversion rate per channel
    for ch in df["channel"].unique():
        sub = df[df["channel"] == ch]
        axes[1].plot(sub["date"], sub["conversion_rate"], label=ch, marker="o", markersize=2)
    axes[1].set_ylabel("Conversion rate")
    axes[1].legend()

    # Mark the ground-truth incident start on both plots
    incident_date = pd.to_datetime(ground_truth["start_date"])
    for ax in axes:
        ax.axvline(incident_date, color="red", linestyle="--", alpha=0.7,
                    label="incident start" if ax == axes[0] else None)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.suptitle(ground_truth["description"], y=1.02, fontsize=10, wrap=True)
    plt.show()


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data" / "scenarios"
    plot_scenario("scenario_01", data_dir)   # traffic_collapse on referral
    plot_scenario("scenario_04", data_dir)   # pricing_change, all channels, gradual ramp
    plot_scenario("scenario_05", data_dir)   # spend_cut on paid — should look identical to scenario_01's shape
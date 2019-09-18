#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set("notebook")
sns.set_style("white")


def plot_memory_history():
    # Load CSVs
    processor_stats = pd.read_csv("../data/statistics/Processor memory and model memory - Processors+.csv")
    model_stats = pd.read_csv("../data/statistics/Processor memory and model memory - Model memory.csv")
    print(processor_stats.columns)
    print(model_stats.columns)

    # Extract model data
    models, model_dates, model_total_mem = [], [], []

    for i, row in model_stats.iterrows():
        if str(row["Exclude?"]).strip() == "1":
            continue
        try:
            mem = float(row["Total peak memory"])
            date = row["Date of paper (original arXiv v1, not arXiv revision dates)"]
            if str(date) == "nan" or "/" not in date:
                continue
            model_name = row["Model"].strip()
            if "," in model_name:
                model_name = model_name[:model_name.index(",")]
            if model_name in ("VGG19", "FCN32 (Pascal)", "FCN16 (Pascal)") or model_name.startswith("SE-"):
                continue
            models.append(model_name)
            model_dates.append(pd.to_datetime(date))
            model_total_mem.append(mem)
        except Exception as e:
            print("ERROR", e)
            continue

    # Extract processor data
    processor, pdates, pmem = [], [], []

    for i, row in processor_stats.iterrows():
        if str(row["Exclude?"]).strip() == "1":
            continue
        try:
            mem = float(row["Memory"])
            name = row["Model"]
            date = pd.to_datetime(row["Launch date"])
            processor.append(name)
            pdates.append(date)
            pmem.append(mem)
        except Exception as e:
            print("ERROR", e)
            continue

    # Plot
    list(zip(model_dates, model_total_mem))
    plt.plot_date(model_dates, model_total_mem)
    for model, x, y in zip(models, model_dates, model_total_mem):
        print(model, x, y)
        plt.annotate(model, (x+pd.DateOffset(months=1), y-0.2))
    plt.ylabel("Memory usage (GB)")

    plt.scatter(pdates, pmem, alpha=0.3, c='green')

    plt.savefig("../data/statistics/memory_history.pdf")
    plt.savefig("../data/statistics/memory_history.png")


def plot_model_memory_breakdown():
    pass


if __name__=="__main__":
    plot_memory_history()
    plot_model_memory_breakdown()

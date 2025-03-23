import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib

results_path = "./metrics_results/results.json"
methods =  ["BPR", "LightGCN", "NGCF", "MultiVAE", "Random"]
datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
datasets_formatted = ["ML-100K", "ML-1M", "ML-20M", "Gowalla", "Steam"]

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_data(exposure_data: List[float], dataset: str, method: str):
    plt.figure(figsize=(8, 5))
    items = [i for i in range(len(exposure_data))]
    exposure = sorted(exposure_data, reverse=True)
    plt.scatter(items, exposure, s=10)
    plt.xlabel('Items')
    plt.ylabel('Exposure')
    plt.suptitle(F'Exposure of Each Recommended Items')
    plt.title(f"{datasets_formatted[datasets.index(dataset)]} - {method}")
    folder = "./plots/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    plt.savefig(f"{folder}{dataset}-{method}.pgf")
    plt.close()


def load_data(path: str):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    raise FileNotFoundError(f"File {path} not found.")


if __name__ == "__main__":
    results = load_data(results_path)
    for dataset in datasets:
        if dataset in results:
            dataset_results = results[dataset]
            for method in methods:
                if method in dataset_results and "test_result" in dataset_results[method] and "plot_data" in dataset_results[method]["test_result"]:
                    plot_data(dataset_results[method]["test_result"]["plot_data"], dataset, method)
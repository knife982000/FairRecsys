import json
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt

results_path = "./metrics_results/results.json"
methods =  ["BPR", "LightGCN", "NGCF", "MultiVAE", "Random"]
datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
datasets_formatted = ["ML-100K", "ML-1M", "ML-20M", "Gowalla", "Steam"]


def plot_data(exposure_data: Dict[str, List[float]], dataset: str, fair_exposure: float):
    plt.figure(figsize=(6, 4))
    for method, data in exposure_data.items():
        exposure = sorted(data, reverse=True)
        items = [i for i in range(len(exposure))]
        plt.scatter(items, exposure, s=3, alpha=0.6, label=method)
    plt.axhline(y=fair_exposure, color='black', linestyle='--', label='Fair Exposure')
    plt.xlabel('Items')
    plt.ylabel('Exposure')
    plt.suptitle('Exposure of Each Recommended Items')
    plt.title(f"{datasets_formatted[datasets.index(dataset)]}")
    plt.legend()
    folder = "./plots/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    plt.savefig(f"{folder}{dataset}.png", dpi=300)
    plt.show()
    plt.close()


def load_data(path: str):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    raise FileNotFoundError(f"File {path} not found.")


def fill_table(dataset: str, k: int, results: Dict[str, Any]) -> str:
    def bold_max(values, is_exposure=False):
        def format_value(v):
            return f"{v:.4f}" if v >= 0.0001 else f"{v:.6f}"

        if is_exposure:
            closest_to_0_5 = min(values, key=lambda x: abs(x - 0.5))
            return [f"\\textbf{{{format_value(v)}}}" if v == closest_to_0_5 else format_value(v) for v in values]
        else:
            max_value = max(values)
            return [f"\\textbf{{{format_value(v)}}}" if v == max_value else format_value(v) for v in values]

    metrics = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "shannonentropy@10", "novelty", "exposure_50-50@10", "exposure_80-19@10", "exposure_90-9@10", "exposure_99-1@10"]
    models = ["Random", "BPR", "NGCF", "LightGCN", "MultiVAE"]
    table_data = {metric: [] for metric in metrics}

    for model in models:
        for metric in metrics:
            value = results["ml-1m"][model]["test_result"][metric]
            table_data[metric].append(value)

    for metric in metrics:
        is_exposure = "exposure" in metric
        table_data[metric] = bold_max(table_data[metric], is_exposure)

    table_str = f"\\begin{{table*}}\n \\caption{{Metrics results for the {dataset} dataset, where the k value is {k}.}}\n  \\label{{table:results_{dataset}}}\n  \\centering\n  \\begin{{tabular}}{{lcccccccccc}}\n    \\toprule\n    \\textbf{{Model}} & \\textbf{{Recall}} & \\textbf{{MRR}} & \\textbf{{NDCG}} & \\textbf{{Hit}} & \\textbf{{SE}} & \\textbf{{Novelty}} & \\textbf{{Exp 50-50}} & \\textbf{{Exp 80-20}} & \\textbf{{Exp 90-10}} & \\textbf{{Exp 99-1}} \\\\\n    \\midrule\n"
    for i, model in enumerate(models):
        table_str += f"    {model} & {table_data['recall@10'][i]} & {table_data['mrr@10'][i]} & {table_data['ndcg@10'][i]} & {table_data['hit@10'][i]} & {table_data['shannonentropy@10'][i]} & {table_data['novelty'][i]} & {table_data['exposure_50-50@10'][i]} & {table_data['exposure_80-19@10'][i]} & {table_data['exposure_90-9@10'][i]} & {table_data['exposure_99-1@10'][i]} \\\\\n"
    table_str += "    \\bottomrule\n  \\end{tabular}\n\\end{table*}"
    folder = "./plots/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    with open(f"{folder}{dataset}_table.tex", "w") as file:
        file.write(table_str)


if __name__ == "__main__":
    results = load_data(results_path)
    for dataset in datasets:
        if dataset in results:
            dataset_results = {}

            fill_table(dataset, 10, results)

            for method in methods:
                method_path = f"./metrics_results/{dataset}/{method}.json"
                if os.path.exists(method_path):
                    with open(method_path, "r") as file:
                        dataset_results[method] = json.load(file)
            method_data = {}
            fair_exposure = 0
            for method in methods:
                fair_exposure = dataset_results[method]["fair_exposure"]
                method_data[method] = dataset_results[method]["plot_data"] + [0] * (dataset_results[method]["num_items"] - len(dataset_results[method]["plot_data"]))
            plot_data(method_data, dataset, fair_exposure)
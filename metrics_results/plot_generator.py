import json
import os
from typing import Dict, List, Any
import plotly.graph_objects as go
import numpy as np

methods_normal = ["BPR", "LightGCN", "NGCF", "MultiVAE", "Random"]
methods_zipf = ["BPRZipf", "LightGCNZipf", "NGCFZipf", "MultiVAEZipf"]
methods_sampled = [
    "BPR_Combined_O1.5_U0.75",
    "BPR_Oversample_O1.5",
    "BPR_Undersample_U0.75",
    "LightGCN_Combined_O1.5_U0.75",
    "LightGCN_Oversample_O1.5",
    "LightGCN_Undersample_U0.75",
    "MultiVAE_Combined_O1.5_U0.75",
    "MultiVAE_Oversample_O1.5",
    "MultiVAE_Undersample_U0.75",
    "NGCF_Combined_O1.5_U0.75",
    "NGCF_Oversample_O1.5",
    "NGCF_Undersample_U0.75"
]
datasets = ["ml-1m", "gowalla-merged", "steam-merged"]


def format_dataset_name(dataset: str) -> str:
    if dataset == "ml-1m":
        return "ML-1M"
    elif dataset == "gowalla-merged":
        return "Gowalla"
    elif dataset == "steam-merged":
        return "Steam"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def plot_data(exposure_data: Dict[str, List[float]], dataset: str, fair_exposure: float, technique: str = ""):
    fig = go.Figure()
    highest_exposure = 0
    items = []
    for method, data in exposure_data.items():
        exposure = np.asarray([e if e > 0 else 1e-1 for e in sorted(data, reverse=True)])
        highest_exposure = max(highest_exposure, exposure[0])
        items = np.asarray(list(range(len(exposure))))
        fig.add_trace(go.Scattergl(x=items, y=exposure, mode='markers', name=method))
    fig.add_trace(go.Scatter(x=[0, len(items)], y=[fair_exposure, fair_exposure], mode='lines', name='Fair Exposure', line=dict(dash='dash', color='black')))
    y_max = 0
    for i in range(1, 100):
        if (10 ** (i + 0.5)) > highest_exposure:
            y_max = i + 0.5
            break
    fig.update_layout(
        xaxis_title='Items',
        yaxis_title='Exposure',
        legend_title='',
        xaxis_type='log',
        yaxis_type='log',
        yaxis_exponentformat='none',
        yaxis_range=[-1.1, y_max],
        yaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000, 100000],
        yaxis_ticktext=["0", "1", "10", "100", "1K", "10K", "100K"],
        xaxis_tickvals=[0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        xaxis_ticktext=["0", "1", "10", "100", "1K", "10K", "100K", "1M"],
        font=dict(size=18),
        margin=dict(l=0, r=0, t=2, b=2),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35 if dataset == "ml-1m" else -0.25,
            xanchor="center",
            x=0.5,
        )
    )
    folder = "./plots/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    fig.write_image(f"{folder}{dataset}{technique}.png")


def load_data(path: str):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    raise FileNotFoundError(f"File {path} not found.")


def fill_table(dataset: str, results: Dict[str, Any], technique: str = "") -> str:
    def bold_max(values, is_exposure=False):
        def format_value(v):
            return f"{v:.4f}" if v >= 0.0001 else f"{v:.6f}"

        if is_exposure:
            closest_to_0_5 = min(values, key=lambda x: abs(x - 0.5))
            return [f"\\textbf{{{format_value(v)}}}" if v == closest_to_0_5 else format_value(v) for v in values]
        else:
            max_value = max(values)
            return [f"\\textbf{{{format_value(v)}}}" if v == max_value else format_value(v) for v in values]

    metrics = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "shannonentropy@10", "novelty", "exposure_global_50-50@10", "exposure_global_80-20@10",
               "exposure_global_90-10@10", "exposure_global_99-1@10"]
    table_data = {metric: [] for metric in metrics}

    result_method = []
    for model in results.keys():
        if model not in results and (model == "MultiVAE" or model == "MultiVAEZipf"):
            continue
        for metric in metrics:
            value = results[model]["test_result"][metric]
            table_data[metric].append(value)
        result_method.append(model)

    for metric in metrics:
        is_exposure = "exposure" in metric
        table_data[metric] = bold_max(table_data[metric], is_exposure)

    table_str = f"\\begin{{table*}}\n \\caption{{Metrics results for the {format_dataset_name(dataset)} dataset.}}\n  \\label{{table:results_{dataset}}}\n  \\centering\n  \\begin{{tabular}}{{lcccccccccc}}\n    \\toprule\n    \\textbf{{Model}} & \\textbf{{Recall}} & \\textbf{{MRR}} & \\textbf{{NDCG}} & \\textbf{{Hit}} & \\textbf{{SE}} & \\textbf{{Novelty}} & \\textbf{{DE$_{{50-50}}$}} & \\textbf{{DE$_{{80-20}}$}} & \\textbf{{DE$_{{90-10}}$}} & \\textbf{{DE$_{{99-1}}$}} \\\\\n    \\midrule\n"
    for i, model in enumerate(result_method):
        model_formatted = model.replace("_", " ").replace("Combined", "Mixed").replace("Undersample U", "Undersample ").replace("Oversample O", "Oversample ")
        table_str += f"    {model_formatted} & {table_data['recall@10'][i]} & {table_data['mrr@10'][i]} & {table_data['ndcg@10'][i]} & {table_data['hit@10'][i]} & {table_data['shannonentropy@10'][i]} & {table_data['novelty'][i]} & {table_data['exposure_global_50-50@10'][i]} & {table_data['exposure_global_80-20@10'][i]} & {table_data['exposure_global_90-10@10'][i]} & {table_data['exposure_global_99-1@10'][i]} \\\\\n"
    table_str += "    \\bottomrule\n  \\end{tabular}\n\\end{table*}"
    folder = "./plots/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    with open(f"{folder}{dataset}{technique}_table.tex", "w") as file:
        file.write(table_str)


if __name__ == "__main__":
    #  "", "_zipf" or "_sample"
    technique = "_sample"
    if technique == "":
        methods = methods_normal
    elif technique == "_zipf":
        methods = methods_zipf
    elif technique == "_sample":
        methods = methods_normal
    else:
        raise ValueError(f"Unknown technique: {technique}")

    for dataset in datasets:
        dataset_results = {}

        for method in methods:
            method_path = f"./{dataset}/{method}.json"
            methods_to_plot = []
            if os.path.exists(method_path):
                with open(method_path, "r") as file:
                    data = json.load(file)
                    if technique != "_sample":
                        dataset_results[method] = data[dataset][method]
                        methods_to_plot.append(method)
                        continue
                    for key, value in data[dataset].items():
                        dataset_results[key] = value
                        methods_to_plot.append(key)
            for method_to_plot in methods_to_plot:
                method_path = f"./{dataset}/{method_to_plot}-plot.json"
                if os.path.exists(method_path):
                    with open(method_path, "r") as file:
                        data = json.load(file)
                        dataset_results[method_to_plot]["plot_data"] = data["plot_data"]
                        dataset_results[method_to_plot]["num_items"] = data["num_items"]
                        dataset_results[method_to_plot]["fair_exposure"] = data["fair_exposure"]
                    continue

        if technique == "_sample":
            methods = methods_sampled
        method_data = {}
        fair_exposure = 0
        for method in dataset_results.keys():
            fair_exposure = dataset_results[method]["fair_exposure"]
            method_data[method] = dataset_results[method]["plot_data"] + [-0.1] * (dataset_results[method]["num_items"] - len(dataset_results[method]["plot_data"]))
        plot_data(method_data, dataset, fair_exposure, technique)
        fill_table(dataset, dataset_results, technique)

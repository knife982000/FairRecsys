import plotly.graph_objects as go
from typing import Dict, List
import numpy as np
import os

datasets = {
    "ML-1M": {"path": "ml-1m/ml-1m.inter", "item_id": "item_id:token"},
    "Gowalla": {"path": "gowalla-merged/gowalla-merged.inter", "item_id": "item_id:token"},
    "Steam": {"path": "steam-merged/steam-merged.inter", "item_id": "product_id:token"},
}


def get_interaction_count(path: str, item_id: str) -> Dict[int, int]:
    interaction = {}
    with open(path, 'r') as file:
        item_id_index = find_item_id_index(file.readline(), item_id)
        for line in file:
            line = line.strip().split("\t")
            item_id = int(line[item_id_index])
            if item_id not in interaction:
                interaction[item_id] = 1
            else:
                interaction[item_id] += 1
    return interaction


def find_item_id_index(line: str, item_id: str) -> int:
    headers = line.split("\t")
    item_id_index = -1
    for i, header in enumerate(headers):
        if header == item_id:
            item_id_index = i
            break
    if item_id_index == -1:
        raise ValueError("No item_id column found")
    return item_id_index


def sort_interaction(interaction: Dict[int, int]) -> List[int]:
    return sorted(interaction.values(), reverse=True)


def plot_data(interactions: List[int], dataset: str):
    fig = go.Figure()
    items = np.asarray(list(range(len(interactions))))
    fig.add_trace(go.Scattergl(x=items, y=interactions, mode='markers'))
    y_max = 0
    for i in range(1, 100):
        if (10 ** (i + 0.5)) > interactions[0]:
            y_max = i + 0.5
            break
    fig.update_layout(
        xaxis_title='Items',
        yaxis_title='Interactions',
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
    fig.write_image(f"{folder}{dataset}-interactions.png".lower())


if __name__ == "__main__":
    for dataset, data in datasets.items():
        interactions = get_interaction_count(data["path"], data["item_id"])
        sorted_interactions = sort_interaction(interactions)
        plot_data(sorted_interactions, dataset)

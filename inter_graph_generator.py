import json

from RecboleRunner import RecboleRunner
from collections import Counter

from config import eval_config_file

datasets = [
    "ml-1m",
    "gowalla-merged",
    "steam-merged"
]

if __name__ == "__main__":
    for dataset in datasets:
        config_file = eval_config_file
        if dataset == "steam-merged":
            config_file.append("config_steam.yaml")

        recbole_runner = RecboleRunner("Random", dataset, config_file_list=config_file)
        _, _, _, _, _, test_data = recbole_runner.get_model_and_dataset()

        item_interaction_counts = Counter()
        for _, _, _, positive_i in test_data:
            item_ids = positive_i.tolist()
            item_interaction_counts.update(item_ids)
        sorted_interaction_counts = sorted(item_interaction_counts.values(), reverse=True)
        results = {
            "plot_data": sorted_interaction_counts,
        }
        with open(f"metrics_results/{dataset}.json", "w") as file:
            json.dump(results, file)

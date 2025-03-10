import argparse
import json
import os
from logging import getLogger
from typing import Optional, Any, Dict, List

from RecBole.recbole.quick_start.quick_start import load_data_and_model, run_recbole
from RecBole.recbole.utils import get_trainer, set_color

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = ["BPR", "LightGCN", "NGCF", "MultiVAE"]
datasets = ["ml-1m", "gowalla-merged", "yahoo-music", "amazon-books"]
config_dict = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "GiniIndex", "ItemCoverage"]
}


def is_model_trained(model: str) -> Optional[str]:
    """
    Check if the model has been trained on the specified dataset
    :param model: ``str`` The name of the model
    :return: ``str`` | ``None`` Name of the saved model if it exists
    """
    if os.path.isdir(config_dict["checkpoint_dir"]):
        saved_models = os.listdir(config_dict["checkpoint_dir"])
        for saved_model in saved_models:
            if saved_model.find(model) != -1:
                return saved_model
    return None


def run_and_train_model(model: str, dataset: str) -> Dict[str, Any]:
    return run_recbole(model=model, dataset=dataset, config_dict=config_dict, config_file_list=["config.yaml"])


def run_and_evaluate_model(model: str, dataset: str) -> Dict[str, Any]:
    """
    Run and evaluate the model on the specified dataset
    :param model: ``str`` The name of the model
    :param dataset: ``str`` The name of the dataset
    :return: ``Dict[str, Any]`` The evaluation results
    """
    config_dict["checkpoint_dir"] = model_folder + dataset
    trained_model = is_model_trained(model)
    if trained_model is None:
        return run_and_train_model(model, dataset)

    print(f"Model {model} has been trained on dataset {dataset}. Skipping training.")
    return evaluate_pre_trained_model(model_folder + dataset + "/" + trained_model)


def model_supports_metrics(model_metrics: List) -> bool:
    """
    Check if the model supports the selected metrics
    :param model_metrics: ``List`` The metrics supported by the model
    :return: ``bool`` True if the model supports the selected metrics, False otherwise
    """
    metrics = [metric for metric in config_dict["metrics"] if metric not in model_metrics]
    if len(metrics) > 0:
        print(f"Model doesn't support some selected metrics: {metrics}")
        return False
    return True


def evaluate_pre_trained_model(model_path: str) -> Dict[str, Any]:
    """
    Evaluate a pre-trained model
    :param model_path: ``str`` The path to the pre-trained model
    :return: ``Dict[str, Any]`` The evaluation results
    """
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)

    if not model_supports_metrics(config["metrics"]):
        return {"error": "Model doesn't support some selected metrics"}

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config["show_progress"])

    logger = getLogger()
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return {"test_result": test_result}


def save_metrics_results(model: str, dataset: str, results: Dict[str, Any]) -> None:
    """
    Save the evaluation results
    :param model: ``str`` The name of the model
    :param dataset: ``str`` The name of the dataset
    :param results: ``Dict[str,Any]`` The evaluation results
    :return: ``None``
    """
    path = f"{metrics_results_folder}{dataset}/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}{model}.json", "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate RecBole models")
    parser.add_argument("-d", "--dataset", type=str, help=f"Dataset to use: {datasets}")
    parser.add_argument("-m", "--method", type=str, help=f"Method to use: {methods}")
    args = parser.parse_args()

    if not args.dataset:
        print("Specify a dataset using flag -d")
        exit(1)
    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not supported. Supported datasets: {datasets}")
        exit(1)
    if not args.method:
        print("Specify a method using flag -m")
        exit(1)
    if args.method not in methods:
        print(f"Method {args.method} not supported. Supported methods: {methods}")
        exit(1)

    results = run_and_evaluate_model(args.method, args.dataset)
    save_metrics_results(args.method, args.dataset, results)

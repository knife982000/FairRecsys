import argparse
import json
import os
from logging import getLogger
from typing import Optional, Any, Dict, List
import torch.multiprocessing as mp
import torch.distributed as dist

import torch

from RecBole.recbole.config.configurator import Config
from RecBole.recbole.quick_start.quick_start import load_data_and_model, run_recbole, run_recboles
from RecBole.recbole.utils import get_trainer, set_color

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods =  ["BPR", "LightGCN", "NGCF", "MultiVAE", "Random"]
datasets = ["ml-100k", "ml-1m", "gowalla-merged", "steam-merged"]
config_dict = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty"]
}
config_file = ["config.yaml"]


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


def find_available_port(start: int, end: int) -> int:
    """
    Find an available port in the specified range
    :param start: ``int`` The start of the port range
    :param end: ``int`` The end of the port range
    """
    import socket
    for port in range(start, end + 1):
        # Try to connect to the port, if it fails (doesn't return 0), the port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                print(f"Binding to 127.0.0.1:{port} for multi-GPU training")
                return port
    raise RuntimeError("No available ports in the specified range")


def run_and_train_model_multi_gpu(model: str, dataset: str) -> Dict[str, Any]:
    """
    Run and train the model on the specified dataset using multiple GPUs on a single node.
    Based on run function from RecBole in "recbole.quick_start.quick_start"
    :param model: ``str`` The name of the model
    :param dataset: ``str`` The name of the dataset
    :param nproc: ``int`` The number of GPUs to use
    """
    queue = mp.get_context("spawn").SimpleQueue()

    kwargs = {
        "config_dict": config_dict,
        "queue": queue,
    }

    mp.spawn(
        run_recboles,
        args=(model, dataset, config_file, kwargs),
        nprocs=config_dict["nproc"],
        join=True,
    )

    res = None if queue.empty() else queue.get()
    return res


def run_and_evaluate_model(model: str, dataset: str) -> Dict[str, Any]:
    """
    Run and evaluate the model on the specified dataset
    :param model: ``str`` The name of the model
    :param dataset: ``str`` The name of the dataset
    :return: ``Dict[str, Any]`` The evaluation results
    """
    if torch.cuda.is_available():
        gpus = [f"{torch.cuda.get_device_name(i)} - {i}" for i in range(torch.cuda.device_count())]
        print(f"GPU(s) available({len(gpus)}): {gpus}")
        config_dict["nproc"] = len(gpus)
    else:
        print("No GPU available. Exiting.")
        exit(0)

    # Configuration for distributed training
    config_dict["world_size"] = config_dict["nproc"]
    config_dict["offset"] = 0
    config_dict["ip"] = "127.0.0.1"
    config_dict["port"] = str(find_available_port(5670, 5680))

    config_dict["checkpoint_dir"] = model_folder + dataset
    trained_model = is_model_trained(model)
    if trained_model is None:
        if len(gpus) == 1:
            return run_recbole(model, dataset, config_file, config_dict)
        else:
            return run_and_train_model_multi_gpu(model, dataset)

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
    dist.init_process_group(backend='nccl', init_method=f'tcp://{config_dict["ip"]}:{config_dict["port"]}', world_size=config_dict["nproc"], rank=0)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
    config["nproc"] = config_dict["nproc"]

    if not model_supports_metrics(config["metrics"]):
        return {"error": "Model doesn't support some selected metrics"}

    config_file_eval = ["config_eval.yaml"] if config_file == ["config.yaml"] else ["config_steam_eval.yaml"]
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_eval,
        config_dict=config_dict,
    )

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
    path = f"{metrics_results_folder}results.json"
    print(f"Saving results to {path}")
    os.makedirs(os.path.dirname(metrics_results_folder), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as file:
            all_results = json.load(file)
    else:
        all_results = {}

    if dataset not in all_results:
        all_results[dataset] = {}

    all_results[dataset][model] = results

    with open(path, "w") as file:
        json.dump(all_results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate RecBole models")
    parser.add_argument("-d", "--dataset", type=str, help=f"Dataset to use: {datasets}")
    parser.add_argument("-m", "--method", type=str, help=f"Method to use: {methods}")
    args = parser.parse_args()

    print(f"Called with args: {args}")
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

    if args.dataset == "steam-merged":
        config_file = ["config_steam.yaml"]

    # Fixing compatibility issues
    import numpy as np
    np.float = np.float64
    np.complex = np.complex128
    np.unicode = np.str_

    results = run_and_evaluate_model(args.method, args.dataset)
    save_metrics_results(args.method, args.dataset, results)
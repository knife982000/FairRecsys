import argparse

from config import methods, datasets, config_dictionary, config_file, eval_config_file
from RecboleRunner import RecboleRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate RecBole models")
    parser.add_argument("-d", "--dataset", type=str, help=f"Dataset to use: {datasets}")
    parser.add_argument("-m", "--method", type=str, help=f"Method to use: {methods.keys()}")
    parser.add_argument("-r", "--retrain", action="store_true", help="Ignore pre-trained model and retrain")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the selected model")
    parser.add_argument("-o", "--oversample", type=float, help=f"Ratio for oversampling", default=0.0)
    parser.add_argument("-u", "--undersample", type=float, help=f"Ratio for undersampling", default=0.0)
    parser.add_argument("-s", "--save_model_as", type=str, help=f"Name to save model as", default=None)
    parser.add_argument("-a", "--alpha_values", type=str, help="Comma-separated list of zipf_alpha values for grid search")
    parser.add_argument("-z", "--apply_zipf", action="store_true", help="Apply Zipf's penalty to the model")
    parser.add_argument("-mmr", "--mmr", action="store_true", help="Use MMR for reranking")
    parser.add_argument("-fe", "--find_entropy", action="store_true", help=f"Find the optimal entropy alpha value")

    args = parser.parse_args()

    if not args.dataset:
        print("Specify a dataset using flag -d or --dataset")
        exit(1)
    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not supported. Supported datasets: {datasets}")
        exit(1)
    if not args.method:
        print("Specify a method using flag -m or --method")
        exit(1)
    if args.method not in methods.keys():
        print(f"Method {args.method} not supported. Supported methods: {methods.keys()}")
        exit(1)

    if args.mmr:
        if args.evaluate:
            config_dictionary["apply_mmr"] = True
        else:
            print("MMR reranking can only be used during evaluation!")

    if args.evaluate:
        config_file = eval_config_file

    if args.apply_zipf:
        config_dictionary["apply_zipf"] = True

    if args.dataset == "steam-merged":
        config_file.append("config_steam.yaml")

    # Fixing compatibility issues
    import numpy as np

    np.float = np.float64
    np.complex = np.complex128
    np.unicode = np.str_

    print(f"\n------------- Running Recbole -------------\nArguments given: {args}\n")
    runner = RecboleRunner(model_name=args.method, dataset_name=args.dataset, config_file_list=config_file, config_dict=config_dictionary, retrain=args.retrain,
                           evaluate=args.evaluate, over_sample_ratio=args.oversample, under_sample_ratio=args.undersample, save_model_as=args.save_model_as)

    if args.alpha_values:
        alpha_values = [float(a) for a in args.alpha_values.split(",")]
        results = runner.grid_search_zipf_alpha(alpha_values)
        print("Grid search results:", results)
    elif args.find_entropy:
        best_alpha, results = runner.optimize_entropy_alpha()
        runner.save_model_as += f"_optimized_alpha"
        results["best_alpha"] = best_alpha
    else:
        results = runner.run_and_evaluate_model()
    runner.save_metrics_results(results)

import numpy as np

# Exposure is negative as they should be minimized
metric_weights = {
    "recall@10": 2.0,
    "mrr@10": 2.0,
    "ndcg@10": 2.0,
    "precision@10": 2.0,
    "hit@10": 2.0,
    "exposure_50-50@10": 0.0,
    "exposure_global_50-50@10": -1.0,
    "exposure_80-20@10": 0.0,
    "exposure_global_80-20@10": -1.0,
    "exposure_90-10@10": 0.0,
    "exposure_global_90-10@10": -1.0,
    "exposure_99-1@10": 0.0,
    "exposure_global_99-1@10": -1.0,
    "shannonentropy@10": 0.5,
    "novelty": 2.0,
}


def find_optimal_alpha(runner, alpha_start: float, alpha_end: float, step: float):
    """
    Find the optimal alpha value for the model based on a weighted score of metrics.

    :param runner: RecboleRunner instance
    :param alpha_start: Starting value for alpha
    :param alpha_end: Ending value for alpha
    :param step: Step size for alpha values
    :param metric_weights: Dictionary of weights for each metric
    :return: Optimal alpha value and corresponding results
    """
    alpha_values = np.arange(alpha_start, alpha_end + step, step)

    best_alpha = None
    best_score = float("-inf")
    best_result = None

    for alpha in alpha_values:
        runner.logger.info(f"Testing alpha={alpha}")
        runner.config_dict["entropy_alpha"] = alpha
        runner.retrain = True
        results = runner.run_and_evaluate_model()

        # Compute weighted score
        test_result = results.get("best_valid_result", {})
        score = sum(
            metric_weights.get(metric, 0) * test_result.get(metric, 0)
            for metric in metric_weights
        )

        runner.logger.info(f"Alpha={alpha}, Score={score}, Results={test_result}")

        if score > best_score:
            results["weighted_score"] = score
            best_score = score
            best_alpha = alpha
            best_result = test_result

    runner.logger.info(f"Best alpha: {best_alpha}, Best score: {best_score}")

    return best_alpha, best_result

from model import BPREntropy, NGCFEntropy, LightGCNEntropy, SSEPT, SSEPTEntropy

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = {"BPR": None, "LightGCN": None, "NGCF": None,"Random": None, "BPREntropy": BPREntropy, "NGCFEntropy": NGCFEntropy, "LightGCNEntropy": LightGCNEntropy, "SSEPT":SSEPT, "SSEPTEntropy":SSEPTEntropy,}

datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
config_dictionary = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty", "RecommendedGraph", "TailPercentage","JensenShannonDivergence"]
}
config_file = ["config.yaml"]
eval_config_file = ["eval_config.yaml"]

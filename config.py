from model import BPREntropy, BPREntropy2, NGCFEntropy, NGCFEntropy2, LightGCNEntropy, LightGCNEntropy2, SSEPT, SSEPTEntropy, SSEPTEntropy2

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = {"BPR": None, "LightGCN": None, "NGCF": None,"Random": None, 
           "BPREntropy": BPREntropy, "BPREntropy2": BPREntropy2,
           "NGCFEntropy": NGCFEntropy, "NGCFEntropy2": NGCFEntropy2,
           "LightGCNEntropy": LightGCNEntropy, "LightGCNEntropy2": LightGCNEntropy2, 
           "SSEPT":SSEPT, "SSEPTEntropy":SSEPTEntropy, "SSEPTEntropy2":SSEPTEntropy2,}

datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
config_dictionary = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty", "RecommendedGraph", "TailPercentage","JensenShannonDivergence"]
}
config_file = ["config.yaml"]
eval_config_file = ["eval_config.yaml"]

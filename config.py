from RecBole.recbole.model.general_recommender.bpr_mmr_sim import BPRMMRSim
from model.models.lightgcn_zipf import LightGCNZipf
from model.models.ngcf_zipf import NGCFZipf
from model.models.bpr_zipf import BPRZipf

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = {"BPR": None, "LightGCN": None, "NGCF": None,"Random": None, "BPRZipf": BPRZipf,
           "BPRMMRSim": BPRMMRSim, "LightGCNZipf": LightGCNZipf, "NGCFZipf": NGCFZipf}

datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
config_dictionary = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty", "RecommendedGraph"]
}
config_file = ["config.yaml"]
eval_config_file = ["eval_config.yaml"]

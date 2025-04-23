from RecBole.recbole.model.general_recommender.lightgcn_zipf import LightGCNZipf
from RecBole.recbole.model.general_recommender.multivae_zipf import MultiVAEZipf
from RecBole.recbole.model.general_recommender.ngcf_zipf import NGCFZipf
from RecBole.recbole.model.general_recommender.bpr_zipf import BPRZipf

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = {"BPR": None, "LightGCN": None, "NGCF": None, "MultiVAE": None, "Random": None, "BPRZipf": BPRZipf,
           "LightGCNZipf": LightGCNZipf, "NGCFZipf": NGCFZipf, "MultiVAEZipf": MultiVAEZipf }

datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
config_dictionary = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty", "RecommendedGraph"]
}
config_file = ["config.yaml"]
eval_config_file = ["eval_config.yaml"]

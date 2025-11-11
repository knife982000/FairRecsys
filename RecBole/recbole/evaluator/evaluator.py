# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
import numpy as np

class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)

        if "rec.user_group" in dataobject._data_dict:
            user_group = dataobject["rec.user_group"]
            user_group = user_group.cpu().numpy()
            groups = np.unique(user_group)
            groups.sort()
            for g in groups:
                g_dataobject = DataStruct()
                for key, value in dataobject.items():
                    if key.startswith("rec."):
                        g_dataobject.set(key, value[user_group == g])
                    else:
                        g_dataobject.set(key, value)
                for metric in self.metrics:
                    metric_val = self.metric_class[metric].calculate_metric(g_dataobject)
                    metric_val = {f"{g}:{k}": v for k, v in metric_val.items()}
                    result_dict.update(metric_val)

        if "boostrap" in self.config and self.config["boostrap"]:
            boostrap = self.evaluate_bootstrap(dataobject)
            result_dict.update(boostrap)
        return result_dict

    def compute_statistics(self, data, confidence_level):
        """calculate the statistics of the data. It is called at the end of each epoch

        Args:
            data (dict): It contains all the information needed for metrics.
            confidence_level (float): confidence level for confidence interval.

        Returns:
            collections.OrderedDict: such as ``{'hit@20_std': 0.0024, 'recall@20_std': 0.0007, 'hit@10_std': 0.0021, 'recall@10_std': 0.0006,
                                                'hit@20_avg': 0.544, 'recall@20_avg': 0.534, 'hit@10_avg': 0.343, 'recall@10_avg': 0.423,
                                                'hit@20_lowconf': 0.540, 'recall@20_lowconf': 0.530, 'hit@10_lowconf': 0.340, 'recall@10_lowconf': 0.420,
                                                'hit@20_highconf': 0.549, 'recall@20_highconf': 0.539, 'hit@10_highconf': 0.349, 'recall@10_highconf': 0.429}``


        """
        result_dict = OrderedDict()
        low_confidence = (1.0 - confidence_level) / 2.0
        high_confidence = 1.0 - low_confidence

        for k, v in data.items():
            result_dict[f"{k}_avg"] = np.mean(v)
            result_dict[f"{k}_std"] = np.std(v)
            result_dict[f"{k}_lowconf"] = np.quantile(v, low_confidence)
            result_dict[f"{k}_highconf"] = np.quantile(v, high_confidence)
        return result_dict
    

    def evaluate_bootstrap(self, org_dataobject: DataStruct):
        """calculate all the metrics with boostrap. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20_std': 0.0024, 'recall@20_std': 0.0007, 'hit@10_std': 0.0021, 'recall@10_std': 0.0006,
                                                'hit@20_avg': 0.544, 'recall@20_avg': 0.534, 'hit@10_avg': 0.343, 'recall@10_avg': 0.423,
                                                'hit@20_lowconf': 0.540, 'recall@20_lowconf': 0.530, 'hit@10_lowconf': 0.340, 'recall@10_lowconf': 0.420,
                                                'hit@20_highconf': 0.549, 'recall@20_highconf': 0.539, 'hit@10_highconf': 0.349, 'recall@10_highconf': 0.429}``

        """

        assert "_mask" not in vars(), "`_mask` is a reserved variable in `evaluate_bootstrap`, please change your code to avoid the name conflict."

        n_bootstrap = 1000 if "n_bootstrap" not in self.config else self.config["n_bootstrap"]
        seed = 42 if "bootstrap_random_seed" not in self.config else self.config["bootstrap_random_seed"]
        rng = np.random.default_rng(seed)
        confidence_level = 0.95 if "bootstrap_confidence_level" not in self.config else self.config["bootstrap_confidence_level"]
        file_run = None if "bootstrap_file" not in self.config else self.config["bootstrap_file"]
        
        def update_metrics(out_dict, in_dict):
            for key in in_dict:
                if key not in out_dict:
                    out_dict[key] = []
                out_dict[key].append(in_dict[key])
            return out_dict

        result_dict = OrderedDict()
        #Bootstrap sampling
        from tqdm import tqdm
        for _ in tqdm(range(n_bootstrap)):
            #subset of recommendations
            dataobject = DataStruct()

            for key, value in org_dataobject.items():
                if key.startswith("rec."):
                    if "_mask" not in vars():
                        n_users = value.shape[0]
                        _mask = rng.integers(0, n_users, n_users)
                        #mask = torch.tensor(indices, device=value.device)
                    dataobject.set(key, value[_mask])
                else:
                    dataobject.set(key, value)
            del _mask
            #Compute metrics
            for metric in self.metrics:
                metric_val = self.metric_class[metric].calculate_metric(dataobject)
                update_metrics(result_dict, metric_val)

            if "rec.user_group" in dataobject._data_dict:
                user_group = dataobject["rec.user_group"]
                user_group = user_group.cpu().numpy()
                groups = np.unique(user_group)
                groups.sort()
                for g in groups:
                    g_dataobject = DataStruct()
                    for key, value in dataobject.items():
                        if key.startswith("rec."):
                            g_dataobject.set(key, value[user_group == g])
                        else:
                            g_dataobject.set(key, value)
                    for metric in self.metrics:
                        metric_val = self.metric_class[metric].calculate_metric(g_dataobject)
                        metric_val = {f"{g}:{k}": v for k, v in metric_val.items()}
                        update_metrics(result_dict, metric_val)

        if file_run is not None:
            import pickle
            with open(file_run, "wb") as f:
                pickle.dump(result_dict, f)

        return self.compute_statistics(result_dict, confidence_level)
